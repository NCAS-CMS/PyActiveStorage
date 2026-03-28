"""Session transport helpers for SSH-backed request handling."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from io import BufferedIOBase
import subprocess
import threading
from typing import Any

from .protocol import ERROR, FILE_CLOSE, FILE_OPEN, GET_CHUNK, HEARTBEAT, LIST, REDUCE, STAT, VAR_OPEN, read_message, write_message


class SessionError(RuntimeError):
	"""Base exception for session-layer failures."""


class ResponseError(SessionError):
	"""Raised when the remote server replies with an error message."""

	def __init__(self, message: str, *, response: Mapping[str, Any]):
		super().__init__(message)
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
	) -> None:
		self.host = host
		self.username = username
		self._proc = process
		self._stdin = stdin if stdin is not None else getattr(process, "stdin", None)
		self._stdout = stdout if stdout is not None else getattr(process, "stdout", None)
		self._lock = threading.RLock()

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

	def request(self, message_type: str, /, **fields: Any) -> dict[str, Any]:
		"""Send one request and wait for one response."""

		with self._lock:
			write_message(self._stdin, message_type, **fields)
			response = read_message(self._stdout)

		if response["type"] == ERROR:
			raise ResponseError(response.get("message", "remote server returned an error"), response=response)

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

		return self.request(FILE_OPEN, path=path)

	def var_open(self, path: str, varname: str) -> dict[str, Any]:
		"""Open a remote variable and return its metadata."""

		return self.request(VAR_OPEN, path=path, varname=varname)

	def get_chunk(self, path: str, varname: str, byte_offset: int, size: int, **fields: Any) -> dict[str, Any]:
		"""Request raw chunk bytes from the remote server."""

		return self.request(
			GET_CHUNK,
			path=path,
			varname=varname,
			byte_offset=byte_offset,
			size=size,
			**fields,
		)

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


__all__ = ["ResponseError", "Session", "SessionError", "p5remSession"]