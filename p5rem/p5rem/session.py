"""Session transport helpers for SSH-backed request handling."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from io import BufferedIOBase
import subprocess
import threading
from typing import Any

from .cache import P5RemCache, get_default_cache
from .protocol import CHUNK_DATA, ERROR, FILE_INFO, FILE_CLOSE, FILE_OPEN, GET_CHUNK, HEARTBEAT, LIST, LIST_RESULT, REDUCE, REDUCTION_RESULT, STAT, STAT_RESULT, VAR_INFO, VAR_OPEN, read_message, write_message

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
	) -> None:
		self.host = host
		self.username = username
		self._proc = process
		self._stdin = stdin if stdin is not None else getattr(process, "stdin", None)
		self._stdout = stdout if stdout is not None else getattr(process, "stdout", None)
		self._lock = threading.RLock()
		self._cache = cache if cache is not None else (get_default_cache() if host is not None else None)
		self._path_mtime: dict[str, float] = {}

		if self._stdin is None or self._stdout is None:
			raise ValueError("session requires readable stdout and writable stdin streams")

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

		host_key = self._host_cache_key
		if self._cache is not None:
			mtime = self._remote_mtime(path)
			if mtime is not None:
				cached = self._cache.get_file_meta(host_key, path, mtime)
				if cached is not None:
					self._path_mtime[path] = mtime
					return cached

		response = self.request(FILE_OPEN, path=path)
		result = dict(response)
		mtime_value = result.get("mtime")
		if isinstance(mtime_value, (int, float)):
			self._path_mtime[path] = float(mtime_value)
			if self._cache is not None:
				self._cache.set_file_meta(host_key, path, float(mtime_value), result)
		return result

	def open(self, path: str):
		"""Open a remote file as a pyfive-like proxy."""

		from .proxy import rFile

		return rFile(self, path)

	def var_open(self, path: str, varname: str) -> dict[str, Any]:
		"""Open a remote variable and return its metadata."""

		host_key = self._host_cache_key
		mtime = self._path_mtime.get(path)
		if self._cache is not None and mtime is not None:
			cached = self._cache.get_var_meta(host_key, path, varname, mtime)
			if cached is not None:
				return cached

		response = self.request(VAR_OPEN, path=path, varname=varname)
		result = dict(response)
		if self._cache is not None and mtime is not None:
			self._cache.set_var_meta(host_key, path, varname, mtime, result)
		return result

	def get_chunk(self, path: str, varname: str, byte_offset: int, size: int, **fields: Any) -> dict[str, Any]:
		"""Request raw chunk bytes from the remote server."""

		host_key = self._host_cache_key
		mtime = self._path_mtime.get(path)
		if self._cache is not None and mtime is not None:
			with self._cache.transact():
				cached = self._cache.get_chunk(host_key, path, byte_offset, size, mtime)
				if cached is not None:
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