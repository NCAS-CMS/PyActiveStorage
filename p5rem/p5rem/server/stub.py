"""Server-side request dispatch loop and default handlers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import suppress
from io import BufferedIOBase
import os
import sys
import traceback
from typing import Any

from ..protocol import CHUNK_DATA, ERROR, FILE_CLOSE, FILE_INFO, FILE_OPEN, GET_CHUNK, HEARTBEAT, LIST, LIST_RESULT, REDUCE, REDUCTION_RESULT, STAT, STAT_RESULT, VAR_INFO, VAR_OPEN, read_message, write_message


class ServerStub:
	"""Serve p5rem protocol messages over binary streams."""

	def __init__(
		self,
		input_stream: BufferedIOBase | None = None,
		output_stream: BufferedIOBase | None = None,
	) -> None:
		self.input_stream = input_stream if input_stream is not None else sys.stdin.buffer
		self.output_stream = output_stream if output_stream is not None else sys.stdout.buffer
		self._open_files: dict[str, Any] = {}
		self._handlers: dict[str, Callable[..., dict[str, Any]]] = {
			LIST: self.handle_list,
			STAT: self.handle_stat,
			FILE_OPEN: self.handle_file_open,
			VAR_OPEN: self.handle_var_open,
			GET_CHUNK: self.handle_get_chunk,
			REDUCE: self.handle_reduce,
			FILE_CLOSE: self.handle_file_close,
			HEARTBEAT: self.handle_heartbeat,
		}

	def serve_forever(self) -> None:
		"""Process requests until the input stream closes."""

		while self.serve_once():
			pass

	def serve_once(self) -> bool:
		"""Process one request if available."""

		try:
			request = read_message(self.input_stream)
		except EOFError:
			return False

		response = self.dispatch(request)
		write_message(self.output_stream, response)
		return True

	def dispatch(self, request: Mapping[str, Any]) -> dict[str, Any]:
		"""Dispatch one validated request message."""

		request_type = request["type"]
		handler = self._handlers.get(request_type)
		if handler is None:
			return self._error_response(
				f"unsupported request type: {request_type}",
				request_type=request_type,
			)

		fields = {key: value for key, value in request.items() if key != "type"}
		try:
			return handler(**fields)
		except Exception as exc:
			return self._error_response(
				str(exc) or exc.__class__.__name__,
				request_type=request_type,
				error_class=exc.__class__.__name__,
				traceback=traceback.format_exc(),
			)

	def handle_list(self, path: str) -> dict[str, Any]:
		entries = sorted(os.listdir(path))
		return {"type": LIST_RESULT, "path": path, "entries": entries}

	def handle_stat(self, path: str) -> dict[str, Any]:
		stat_result = os.stat(path)
		return {
			"type": STAT_RESULT,
			"path": path,
			"size": stat_result.st_size,
			"mtime": stat_result.st_mtime,
			"mode": stat_result.st_mode,
			"is_dir": os.path.isdir(path),
			"is_file": os.path.isfile(path),
		}

	def handle_file_open(self, path: str) -> dict[str, Any]:
		pyfive = self._load_pyfive()
		file_handle = pyfive.File(path)
		self._open_files[path] = file_handle
		return {
			"type": FILE_INFO,
			"path": path,
			"keys": list(file_handle.keys()),
			"attrs": dict(getattr(file_handle, "attrs", {})),
			"mtime": os.path.getmtime(path),
		}

	def handle_var_open(self, path: str, varname: str) -> dict[str, Any]:
		dataset = self._get_dataset(path, varname)
		file_handle = self._open_files[path]
		chunks = getattr(dataset, "chunks", None)
		if chunks is not None:
			chunks = list(chunks)

		return {
			"type": VAR_INFO,
			"path": path,
			"varname": varname,
			"shape": list(getattr(dataset, "shape", ())),
			"dtype": str(getattr(dataset, "dtype", "unknown")),
			"chunks": chunks,
			"index": self._serialise_value(getattr(getattr(dataset, "id", None), "index", None)),
			"fragmented": not bool(getattr(file_handle, "consolidated_metadata", True)),
		}

	def handle_get_chunk(self, path: str, varname: str, byte_offset: int, size: int, **fields: Any) -> dict[str, Any]:
		dataset = self._get_dataset(path, varname)
		dataset_id = getattr(dataset, "id", None)
		get_raw_chunk = getattr(dataset_id, "_get_raw_chunk", None)

		if get_raw_chunk is None:
			raise NotImplementedError("dataset does not expose raw chunk access")

		if "storeinfo" in fields:
			data = get_raw_chunk(fields["storeinfo"])
		else:
			raise NotImplementedError("GET_CHUNK currently requires a 'storeinfo' field")

		filter_mask = fields.get("filter_mask", 0)
		return {
			"type": CHUNK_DATA,
			"path": path,
			"varname": varname,
			"byte_offset": byte_offset,
			"size": size,
			"filter_mask": filter_mask,
			"data": data,
		}

	def handle_reduce(self, path: str, varname: str, byte_offset: int, size: int, operation: str, **fields: Any) -> dict[str, Any]:
		raise NotImplementedError("REDUCE handling is not implemented yet")

	def handle_file_close(self, path: str) -> dict[str, Any]:
		file_handle = self._open_files.pop(path, None)
		if file_handle is not None:
			close = getattr(file_handle, "close", None)
			if callable(close):
				with suppress(Exception):
					close()
		return {"type": FILE_CLOSE, "path": path, "closed": file_handle is not None}

	def handle_heartbeat(self) -> dict[str, Any]:
		return {"type": HEARTBEAT}

	def _get_dataset(self, path: str, varname: str) -> Any:
		if path not in self._open_files:
			raise FileNotFoundError(f"file is not open: {path}")
		return self._open_files[path][varname]

	def _load_pyfive(self) -> Any:
		try:
			import pyfive
		except ImportError as exc:
			raise RuntimeError("pyfive is required for remote HDF5 access") from exc
		return pyfive

	def _error_response(self, message: str, **fields: Any) -> dict[str, Any]:
		return {"type": ERROR, "message": message, **fields}

	def _serialise_value(self, value: Any) -> Any:
		if isinstance(value, Mapping):
			return {key: self._serialise_value(item) for key, item in value.items()}
		if isinstance(value, tuple):
			return [self._serialise_value(item) for item in value]
		if isinstance(value, list):
			return [self._serialise_value(item) for item in value]
		return value


def main() -> int:
	"""Run the server stub until stdin closes."""

	server = ServerStub()
	server.serve_forever()
	return 0


__all__ = ["ServerStub", "main"]