"""Session transport helpers for SSH-backed request handling."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from io import BufferedIOBase
import logging
import subprocess
import threading
from time import perf_counter
from typing import Any, Callable
from p5rem.proxy import rFile

log = logging.getLogger(__name__)

# Import cache optionally - may not be available on remote servers
try:
	from .cache import P5RemCache, get_default_cache
except ImportError:
	P5RemCache = None  # type: ignore
	get_default_cache = None  # type: ignore

from .protocol import CHUNK_DATA, CHUNKS_DONE, ERROR, FILE_INFO, FILE_CLOSE, FILE_OPEN, GET_CHUNK, GET_CHUNKS, HEARTBEAT, LIST, LIST_RESULT, REDUCE, REDUCTION_RESULT, STAT, STAT_RESULT, VAR_INFO, VAR_OPEN, read_message, write_message

REQUEST_RESPONSE_TYPES = {
	LIST: LIST_RESULT,
	STAT: STAT_RESULT,
	FILE_OPEN: FILE_INFO,
	VAR_OPEN: VAR_INFO,
	GET_CHUNK: CHUNK_DATA,
	REDUCE: REDUCTION_RESULT,
	FILE_CLOSE: FILE_CLOSE,
	HEARTBEAT: HEARTBEAT,
}


class SessionError(RuntimeError):
	"""Base exception for session-layer failures."""


class ResponseError(SessionError):
	"""Raised when the remote server replies with an error message."""

	def __init__(self, message: str, *, response: Mapping[str, Any]):
		super().__init__(message)
		self.response = dict(response)


class UnexpectedResponseError(SessionError):
	"""Raised when a request receives the wrong response type."""

	def __init__(
		self,
		message: str,
		*,
		request_type: str,
		expected_types: tuple[str, ...],
		response: Mapping[str, Any],
	) -> None:
		super().__init__(message)
		self.request_type = request_type
		self.expected_types = expected_types
		self.response = dict(response)


class p5remSession:
	"""Persistent request/response session over binary streams."""

	def __init__(
		self,
		host: str | None = None,
		username: str | None = None,
		*,
		process: subprocess.Popen[bytes] | None = None,
		stdin: BufferedIOBase | None = None,
		stdout: BufferedIOBase | None = None,
		cache: P5RemCache | None = None,
		heartbeat_interval: float | None = None,
		heartbeat_max_failures: int = 3,
		heartbeat_failure_callback: Callable[[p5remSession, Exception], None] | None = None,
	) -> None:
		self.host = host
		self.username = username
		self._proc = process
		self._stdin = stdin if stdin is not None else getattr(process, "stdin", None)
		self._stdout = stdout if stdout is not None else getattr(process, "stdout", None)
		self._lock = threading.RLock()
		self._cache = cache if cache is not None else (get_default_cache() if (host is not None and get_default_cache is not None) else None)
		self._path_mtime: dict[str, float] = {}
		self._heartbeat_interval = heartbeat_interval
		self._heartbeat_max_failures = max(1, int(heartbeat_max_failures))
		self._heartbeat_failure_callback = heartbeat_failure_callback
		self._heartbeat_stop = threading.Event()
		self._heartbeat_thread: threading.Thread | None = None

		if self._stdin is None or self._stdout is None:
			raise ValueError("session requires readable stdout and writable stdin streams")

		log.info(
			"Session created (host=%s, cache=%s)",
			self.host or "local",
			type(self._cache).__name__ if self._cache is not None else "none",
		)

		if heartbeat_interval is not None and heartbeat_interval > 0:
			self.start_heartbeat(interval=heartbeat_interval)

	@property
	def process(self) -> subprocess.Popen[bytes] | None:
		"""Return the attached subprocess, if any."""

		return self._proc

	def __enter__(self) -> p5remSession:
		return self

	def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
		self.close()

	def request(
		self,
		message_type: str,
		/,
		*,
		expected_type: str | tuple[str, ...] | None = None,
		**fields: Any,
	) -> dict[str, Any]:
		"""Send one request and wait for one response."""

		with self._lock:
			write_message(self._stdin, message_type, **fields)
			response = read_message(self._stdout)

		if response["type"] == ERROR:
			raise ResponseError(response.get("message", "remote server returned an error"), response=response)

		expected_types = self._expected_types_for(message_type, expected_type)
		if expected_types is not None and response["type"] not in expected_types:
			raise UnexpectedResponseError(
				(
					f"request {message_type!r} expected response type "
					f"{expected_types} but received {response['type']!r}"
				),
				request_type=message_type,
				expected_types=expected_types,
				response=response,
			)

		return response

	def list(self, path: str) -> list[Any]:
		"""List a remote directory."""

		response = self.request(LIST, path=path)
		return list(response.get("entries", ()))

	def stat(self, path: str) -> dict[str, Any]:
		"""Fetch remote filesystem metadata."""

		response = self.request(STAT, path=path)
		return dict(response)

	def file_open(self, path: str) -> dict[str, Any]:
		"""Open a remote file and return its metadata."""

		log.debug("Opening remote file: %s", path)
		host_key = self._host_cache_key
		if self._cache is not None:
			mtime = self._remote_mtime(path)
			if mtime is not None:
				cached = self._cache.get_file_meta(host_key, path, mtime)
				if cached is not None:
					self._path_mtime[path] = mtime
					log.debug("Cache hit for file metadata: %s", path)
					return cached

		response = self.request(FILE_OPEN, path=path)
		result = dict(response)
		log.debug("File opened: %s (%d top-level keys)", path, len(result.get("keys", ())))
		mtime_value = result.get("mtime")
		if isinstance(mtime_value, (int, float)):
			self._path_mtime[path] = float(mtime_value)
			if self._cache is not None:
				self._cache.set_file_meta(host_key, path, float(mtime_value), result)
		return result

	def open(self, path: str):
		"""Open a remote file as a pyfive-like proxy."""
		return rFile(self, path)

	def var_open(self, path: str, varname: str) -> dict[str, Any]:
		"""Open a remote variable and return its metadata."""

		log.debug("Fetching metadata for variable %r in %s", varname, path)
		p1 = perf_counter()
		host_key = self._host_cache_key
		mtime = self._path_mtime.get(path)
		if self._cache is not None and mtime is not None:
			cached = self._cache.get_var_meta(host_key, path, varname, mtime)
			if cached is not None:
				log.debug("Cache hit for variable metadata: %r in %s", varname, path)
				return cached

		response = self.request(VAR_OPEN, path=path, varname=varname)
		result = dict(response)
		p2 = perf_counter()
		log.debug("Variable metadata received: %r shape=%s dtype=%s (%.3f s)", varname, result.get("shape"), result.get("dtype"), p2 - p1)
		if self._cache is not None and mtime is not None:
			self._cache.set_var_meta(host_key, path, varname, mtime, result)
		return result

	def get_chunk(self, path: str, varname: str, byte_offset: int, size: int, **fields: Any) -> dict[str, Any]:
		"""Request raw chunk bytes from the remote server."""

		log.debug("Fetching chunk offset=%d size=%d for %r in %s", byte_offset, size, varname, path)
		host_key = self._host_cache_key
		mtime = self._path_mtime.get(path)
		if self._cache is not None and mtime is not None:
			with self._cache.transact():
				cached = self._cache.get_chunk(host_key, path, byte_offset, size, mtime)
				if cached is not None:
					log.debug("Cache hit for chunk offset=%d size=%d", byte_offset, size)
					return cached

		response = self.request(
			GET_CHUNK,
			path=path,
			varname=varname,
			byte_offset=byte_offset,
			size=size,
			**fields,
		)
		result = dict(response)
		if self._cache is not None and mtime is not None:
			with self._cache.transact():
				self._cache.set_chunk(host_key, path, byte_offset, size, mtime, result)
		return result

	def get_chunks(
		self,
		path: str,
		varname: str,
		chunks: list[dict[str, Any]],
		thread_count: int = 4,
	) -> dict[int, dict[str, Any]]:
		"""Request a batch of chunks; returns mapping of byte_offset -> CHUNK_DATA response."""

		log.debug("Fetching %d chunks in batch for %r in %s", len(chunks), varname, path)
		with self._lock:
			write_message(self._stdin, GET_CHUNKS, path=path, varname=varname, chunks=chunks, thread_count=thread_count)
			results: dict[int, dict[str, Any]] = {}
			while True:
				msg = read_message(self._stdout)
				if msg["type"] == CHUNKS_DONE:
					break
				if msg["type"] == ERROR:
					raise ResponseError(msg.get("message", "remote server returned an error"), response=msg)
				if msg["type"] == CHUNK_DATA:
					results[int(msg["byte_offset"])] = dict(msg)
		return results

	def reduce(self, path: str, varname: str, byte_offset: int, size: int, operation: str, **fields: Any) -> dict[str, Any]:
		"""Request a remote reduction result."""

		return self.request(
			REDUCE,
			path=path,
			varname=varname,
			byte_offset=byte_offset,
			size=size,
			operation=operation,
			**fields,
		)

	def file_close(self, path: str) -> dict[str, Any]:
		"""Release a remote file handle."""

		return self.request(FILE_CLOSE, path=path)

	def heartbeat(self) -> dict[str, Any]:
		"""Round-trip a keepalive message."""

		return self.request(HEARTBEAT)

	def set_heartbeat_failure_callback(
		self,
		callback: Callable[[p5remSession, Exception], None] | None,
	) -> None:
		"""Set callback invoked after repeated heartbeat failures."""

		self._heartbeat_failure_callback = callback

	def start_heartbeat(
		self,
		*,
		interval: float | None = None,
		max_failures: int | None = None,
	) -> None:
		"""Start background heartbeat loop if not already running."""

		if interval is not None:
			self._heartbeat_interval = interval
		if max_failures is not None:
			self._heartbeat_max_failures = max(1, int(max_failures))
		if self._heartbeat_interval is None or self._heartbeat_interval <= 0:
			raise ValueError("heartbeat interval must be > 0")
		if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
			return

		self._heartbeat_stop.clear()
		thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
		thread.start()
		self._heartbeat_thread = thread
		log.info("Heartbeat started (interval=%.1fs, max_failures=%d)", self._heartbeat_interval, self._heartbeat_max_failures)

	def stop_heartbeat(self) -> None:
		"""Stop background heartbeat loop."""

		self._heartbeat_stop.set()
		thread = self._heartbeat_thread
		self._heartbeat_thread = None
		if thread is not None and thread.is_alive() and thread is not threading.current_thread():
			thread.join(timeout=1)
		log.info("Heartbeat stopped")

	def _heartbeat_loop(self) -> None:
		failures = 0
		while not self._heartbeat_stop.wait(float(self._heartbeat_interval or 0)):
			try:
				self.heartbeat()
				failures = 0
			except Exception as exc:
				failures += 1
				if failures >= self._heartbeat_max_failures:
					callback = self._heartbeat_failure_callback
					if callback is not None:
						with suppress(Exception):
							callback(self, exc)
					failures = 0

	@property
	def _host_cache_key(self) -> str:
		return self.host if self.host is not None else "local"

	def _remote_mtime(self, path: str) -> float | None:
		try:
			stat_result = self.request(STAT, path=path)
		except Exception:
			return None
		mtime = stat_result.get("mtime")
		if isinstance(mtime, (int, float)):
			return float(mtime)
		return None

	def _expected_types_for(
		self,
		message_type: str,
		expected_type: str | tuple[str, ...] | None,
	) -> tuple[str, ...] | None:
		if expected_type is None:
			inferred = REQUEST_RESPONSE_TYPES.get(message_type)
			return (inferred,) if inferred is not None else None

		if isinstance(expected_type, str):
			return (expected_type,)

		return tuple(expected_type)

	def close(self) -> None:
		"""Close attached streams and terminate the subprocess if needed."""

		self.stop_heartbeat()

		stdin = self._stdin
		stdout = self._stdout
		proc = self._proc

		self._stdin = None
		self._stdout = None
		self._proc = None

		if stdin is not None:
			with suppress(Exception):
				stdin.close()
		if stdout is not None:
			with suppress(Exception):
				stdout.close()

		if proc is not None and proc.poll() is None:
			with suppress(Exception):
				proc.terminate()
			with suppress(Exception):
				proc.wait(timeout=1)


Session = p5remSession


__all__ = [
	"REQUEST_RESPONSE_TYPES",
	"ResponseError",
	"Session",
	"SessionError",
	"UnexpectedResponseError",
	"p5remSession",
]