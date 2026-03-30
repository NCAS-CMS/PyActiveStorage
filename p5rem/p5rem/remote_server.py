"""
Self-contained p5rem remote server.

This file is designed to be uploaded to a remote system and executed standalone.
It has NO imports from the p5rem package — only pyfive, cbor2, and stdlib.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from io import BufferedIOBase
import math
import os
import struct
import sys
import traceback
from typing import Any

import cbor2

# ---------------------------------------------------------------------------
# Protocol constants (inlined from p5rem.protocol)
# ---------------------------------------------------------------------------

LENGTH_PREFIX_SIZE = 4

LIST = "LIST"
STAT = "STAT"
FILE_OPEN = "FILE_OPEN"
VAR_OPEN = "VAR_OPEN"
GET_CHUNK = "GET_CHUNK"
REDUCE = "REDUCE"
FILE_CLOSE = "FILE_CLOSE"
HEARTBEAT = "HEARTBEAT"

LIST_RESULT = "LIST_RESULT"
STAT_RESULT = "STAT_RESULT"
FILE_INFO = "FILE_INFO"
VAR_INFO = "VAR_INFO"
CHUNK_DATA = "CHUNK_DATA"
ERROR = "ERROR"

ALL_TYPES = frozenset({
    LIST, STAT, FILE_OPEN, VAR_OPEN, GET_CHUNK, REDUCE, FILE_CLOSE, HEARTBEAT,
    LIST_RESULT, STAT_RESULT, FILE_INFO, VAR_INFO, CHUNK_DATA, ERROR,
})


def _read_exact(stream: BufferedIOBase, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError(f"unexpected end of stream while reading {size} bytes")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_message(stream: BufferedIOBase) -> dict[str, Any]:
    prefix = _read_exact(stream, LENGTH_PREFIX_SIZE)
    payload_length = struct.unpack(">I", prefix)[0]
    payload = _read_exact(stream, payload_length)
    return cbor2.loads(payload)


def write_message(stream: BufferedIOBase, message: dict[str, Any]) -> None:
    payload = cbor2.dumps(message)
    stream.write(struct.pack(">I", len(payload)) + payload)
    stream.flush()


# ---------------------------------------------------------------------------
# Server stub (inlined from p5rem.server.stub)
# ---------------------------------------------------------------------------

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
        self._handlers = {
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
        while True:
            try:
                request = read_message(self.input_stream)
            except EOFError:
                return
            response = self.dispatch(request)
            write_message(self.output_stream, response)

    def dispatch(self, request: Mapping[str, Any]) -> dict[str, Any]:
        request_type = request.get("type", "")
        handler = self._handlers.get(request_type)
        if handler is None:
            return {"type": ERROR, "message": f"unsupported request type: {request_type}", "request_type": request_type}
        fields = {key: value for key, value in request.items() if key != "type"}
        try:
            return handler(**fields)
        except Exception as exc:
            return {
                "type": ERROR,
                "message": str(exc) or exc.__class__.__name__,
                "request_type": request_type,
                "error_class": exc.__class__.__name__,
                "traceback": traceback.format_exc(),
            }

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
        import pyfive
        file_handle = pyfive.File(path)
        self._open_files[path] = file_handle
        return {
            "type": FILE_INFO,
            "path": path,
            "keys": list(file_handle.keys()),
            "attrs": self._serialise(dict(getattr(file_handle, "attrs", {}))),
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
            "index": self._dataset_index(dataset),
            "attrs": self._serialise(dict(getattr(dataset, "attrs", {}))),
            "fillvalue": self._serialise(getattr(dataset, "fillvalue", None)),
            "filter_pipeline": self._serialise(getattr(getattr(dataset, "id", None), "filter_pipeline", None)),
            "order": self._serialise(getattr(getattr(dataset, "id", None), "_order", "C")),
            "layout": "chunked" if chunks else "contiguous",
            "fragmented": not bool(getattr(file_handle, "consolidated_metadata", True)),
        }

    def handle_get_chunk(self, path: str, varname: str, byte_offset: int, size: int, **fields: Any) -> dict[str, Any]:
        dataset = self._get_dataset(path, varname)
        data, filter_mask, resolved_offset, resolved_size = self._read_raw_data(path, dataset, byte_offset, size, fields)
        return {
            "type": CHUNK_DATA,
            "path": path,
            "varname": varname,
            "byte_offset": resolved_offset,
            "size": resolved_size,
            "filter_mask": filter_mask,
            "data": data,
        }

    def handle_reduce(self, path: str, varname: str, byte_offset: int, size: int, operation: str, **fields: Any) -> dict[str, Any]:
        raise NotImplementedError("REDUCE handling is not implemented yet")

    def handle_file_close(self, path: str) -> dict[str, Any]:
        file_handle = self._open_files.pop(path, None)
        if file_handle is not None:
            with suppress(Exception):
                file_handle.close()
        return {"type": FILE_CLOSE, "path": path, "closed": file_handle is not None}

    def handle_heartbeat(self) -> dict[str, Any]:
        return {"type": HEARTBEAT}

    def _get_dataset(self, path: str, varname: str) -> Any:
        if path not in self._open_files:
            raise FileNotFoundError(f"file is not open: {path}")
        return self._open_files[path][varname]

    def _dataset_nbytes(self, dataset: Any) -> int:
        shape = tuple(getattr(dataset, "shape", ()))
        dtype = getattr(dataset, "dtype", None)
        itemsize = getattr(dtype, "itemsize", None)
        if itemsize is None:
            return 0
        if not shape:
            return int(itemsize)
        return int(itemsize * math.prod(shape))

    def _dataset_index(self, dataset: Any) -> Any:
        dataset_id = getattr(dataset, "id", None)
        if dataset_id is None:
            return None

        chunks = getattr(dataset, "chunks", None)
        if chunks is not None:
            entries = []
            for chunk_offset, storeinfo in dataset_id.index.items():
                entries.append({
                    "chunk_offset": list(chunk_offset),
                    "byte_offset": int(storeinfo.byte_offset),
                    "size": int(storeinfo.size),
                    "filter_mask": int(getattr(storeinfo, "filter_mask", 0)),
                })
            return entries

        data_offset = getattr(dataset_id, "data_offset", None)
        if data_offset is None:
            return None

        try:
            from pyfive.core import UNDEFINED_ADDRESS as _UNDEF
        except ImportError:
            _UNDEF = 0xFFFFFFFFFFFFFFFF

        if int(data_offset) == _UNDEF:
            return []

        return [{
            "chunk_offset": [0],
            "byte_offset": int(data_offset),
            "size": self._dataset_nbytes(dataset),
            "filter_mask": 0,
        }]

    def _read_raw_data(
        self,
        path: str,
        dataset: Any,
        byte_offset: int,
        size: int,
        fields: Mapping[str, Any],
    ) -> tuple[bytes, int, int, int]:
        dataset_id = getattr(dataset, "id", None)
        get_raw_chunk = getattr(dataset_id, "_get_raw_chunk", None)
        if get_raw_chunk is None:
            raise NotImplementedError("dataset does not expose raw chunk access")

        chunks = getattr(dataset, "chunks", None)
        if chunks is not None:
            storeinfo = fields.get("storeinfo")
            if storeinfo is None and "chunk_coord" in fields:
                coord = tuple(fields["chunk_coord"])
                storeinfo = dataset_id.get_chunk_info_by_coord(coord)
            if storeinfo is None:
                for maybe_storeinfo in dataset_id.index.values():
                    if (int(getattr(maybe_storeinfo, "byte_offset", -1)) == int(byte_offset) and
                            size in (0, int(getattr(maybe_storeinfo, "size", -1)))):
                        storeinfo = maybe_storeinfo
                        break
            if storeinfo is None:
                raise ValueError("unable to resolve chunk storeinfo from request")

            data = get_raw_chunk(storeinfo)
            return (
                data,
                int(getattr(storeinfo, "filter_mask", 0)),
                int(getattr(storeinfo, "byte_offset", byte_offset)),
                int(getattr(storeinfo, "size", len(data))),
            )

        data_offset = getattr(dataset_id, "data_offset", None)
        if data_offset is None:
            raise ValueError("contiguous dataset does not expose data_offset")

        absolute_offset = int(byte_offset)
        if absolute_offset < int(data_offset):
            absolute_offset = int(data_offset) + int(byte_offset)

        read_size = int(size)
        if read_size <= 0:
            read_size = max(0, self._dataset_nbytes(dataset) - (absolute_offset - int(data_offset)))

        with open(path, "rb") as handle:
            handle.seek(absolute_offset)
            data = handle.read(read_size)

        return data, 0, absolute_offset, len(data)

    def _serialise(self, value: Any) -> Any:
        try:
            from pyfive.core import Reference as _PyfiveRef
            if isinstance(value, _PyfiveRef):
                return None
        except ImportError:
            pass
        if hasattr(value, "tolist") and callable(value.tolist):
            return self._serialise(value.tolist())
        if hasattr(value, "item") and callable(value.item):
            with suppress(Exception):
                return self._serialise(value.item())
        if isinstance(value, Mapping):
            return {key: self._serialise(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [self._serialise(item) for item in value]
        if not isinstance(value, (bool, int, float, str, bytes, type(None))):
            return None
        return value


if __name__ == "__main__":
    ServerStub().serve_forever()
