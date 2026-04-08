"""
Self-contained p5rem remote server.

This file is designed to be uploaded to a remote system and executed standalone.
It has NO imports from the p5rem package — only pyfive, cbor2, and stdlib.
"""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
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
GET_CHUNKS = "GET_CHUNKS"
REDUCE = "REDUCE"
FILE_CLOSE = "FILE_CLOSE"
HEARTBEAT = "HEARTBEAT"

LIST_RESULT = "LIST_RESULT"
STAT_RESULT = "STAT_RESULT"
FILE_INFO = "FILE_INFO"
VAR_INFO = "VAR_INFO"
CHUNK_DATA = "CHUNK_DATA"
CHUNKS_DONE = "CHUNKS_DONE"
ERROR = "ERROR"

ALL_TYPES = frozenset({
    LIST, STAT, FILE_OPEN, VAR_OPEN, GET_CHUNK, GET_CHUNKS, REDUCE, FILE_CLOSE, HEARTBEAT,
    LIST_RESULT, STAT_RESULT, FILE_INFO, VAR_INFO, CHUNK_DATA, CHUNKS_DONE, ERROR,
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
# Server stub 
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
        self._datasets: dict[str, dict[str, Any]] = {}
        self._dim_id_to_name: dict[str, dict[int, str]] = {}
        self._dim_id_reference_list: dict[str, dict[int, list[list[Any]]]] = {}
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
            request_type = request.get("type", "")
            if request_type == GET_CHUNKS:
                # Streaming handler: writes multiple messages then CHUNKS_DONE.
                fields = {k: v for k, v in request.items() if k != "type"}
                try:
                    self.handle_get_chunks(**fields)
                except Exception as exc:
                    write_message(self.output_stream, {
                        "type": ERROR,
                        "message": str(exc) or exc.__class__.__name__,
                        "request_type": request_type,
                        "error_class": exc.__class__.__name__,
                        "traceback": traceback.format_exc(),
                    })
            else:
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
        entries: list[dict[str, Any]] = []
        with os.scandir(path) as iterator:
            for entry in iterator:
                full_path = os.path.join(path, entry.name)
                is_link = entry.is_symlink()
                # Prefer directory classification when available, otherwise file.
                entry_type = "directory" if entry.is_dir(follow_symlinks=True) else "file"

                size: int | None = None
                mtime: float | None = None
                with suppress(OSError):
                    stat_result = entry.stat(follow_symlinks=False)
                    mtime = float(stat_result.st_mtime)
                    if entry_type == "file":
                        size = int(stat_result.st_size)

                entries.append(
                    {
                        "name": full_path,
                        "type": entry_type,
                        "size": size,
                        "mtime": mtime,
                        "is_link": is_link,
                    }
                )

        entries.sort(key=lambda item: str(item.get("name", "")))
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
        self._datasets.setdefault(path, {})
        self._build_netcdf_maps(path, file_handle)
        return {
            "type": FILE_INFO,
            "path": path,
            "keys": list(file_handle.keys()),
            "attrs": self._serialise(dict(getattr(file_handle, "attrs", {})), file_handle=file_handle),
            "dim_id_to_name": self._dim_id_to_name.get(path, {}),
            "mtime": os.path.getmtime(path),
        }

    def handle_var_open(self, path: str, varname: str) -> dict[str, Any]:
        dataset = self._get_dataset(path, varname)
        file_handle = self._open_files[path]
        chunks = getattr(dataset, "chunks", None)
        if chunks is not None:
            chunks = list(chunks)
        attrs = self._serialise(dict(getattr(dataset, "attrs", {})), file_handle=file_handle)
        if isinstance(attrs, dict):
            attrs = self._resolve_netcdf_reference_attrs(path, varname, attrs)
        return {
            "type": VAR_INFO,
            "path": path,
            "varname": varname,
            "shape": list(getattr(dataset, "shape", ())),
            "dtype": str(getattr(dataset, "dtype", "unknown")),
            "chunks": chunks,
            "index": self._dataset_index(dataset),
            "attrs": attrs,
            "fillvalue": self._serialise(getattr(dataset, "fillvalue", None), file_handle=file_handle),
            "filter_pipeline": self._serialise(getattr(getattr(dataset, "id", None), "filter_pipeline", None), file_handle=file_handle),
            "order": self._serialise(getattr(getattr(dataset, "id", None), "_order", "C"), file_handle=file_handle),
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

    def handle_get_chunks(
        self,
        path: str,
        varname: str,
        chunks: list[dict[str, Any]],
        thread_count: int = 4,
    ) -> None:
        """Parallel-fetch a batch of chunks and stream them back as CHUNK_DATA messages."""
        dataset = self._get_dataset(path, varname)
        dataset_id = getattr(dataset, "id", None)
        index = getattr(dataset_id, "index", {})

        # Resolve StoreInfo for each requested chunk.
        storeinfos = []
        for chunk_desc in chunks:
            byte_offset = int(chunk_desc["byte_offset"])
            chunk_coord_raw = chunk_desc.get("chunk_coord")
            storeinfo = None
            if chunk_coord_raw is not None:
                storeinfo = index.get(tuple(chunk_coord_raw))
            if storeinfo is None:
                # Fall back to linear scan by byte_offset.
                for si in index.values():
                    if int(si.byte_offset) == byte_offset:
                        storeinfo = si
                        break
            if storeinfo is None:
                raise ValueError(f"cannot resolve chunk for offset={byte_offset}")
            storeinfos.append(storeinfo)

        # Read chunks in parallel using os.pread (no seek contention).
        def _read_one(storeinfo: Any) -> bytes:
            with open(path, "rb") as fh:
                return os.pread(fh.fileno(), storeinfo.size, storeinfo.byte_offset)

        with ThreadPoolExecutor(max_workers=max(1, int(thread_count))) as pool:
            raw_chunks = list(pool.map(_read_one, storeinfos))

        for storeinfo, data in zip(storeinfos, raw_chunks):
            write_message(self.output_stream, {
                "type": CHUNK_DATA,
                "path": path,
                "varname": varname,
                "byte_offset": int(storeinfo.byte_offset),
                "size": int(storeinfo.size),
                "filter_mask": int(getattr(storeinfo, "filter_mask", 0)),
                "data": data,
            })

        write_message(self.output_stream, {"type": CHUNKS_DONE})

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
        cached = self._datasets.get(path, {}).get(varname)
        if cached is not None:
            return cached

        if path not in self._open_files:
            raise FileNotFoundError(f"file is not open: {path}")

        dataset = self._open_files[path][varname]
        self._datasets.setdefault(path, {})[varname] = dataset
        return dataset

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

    def _build_netcdf_maps(self, path: str, file_handle: Any) -> None:
        """Precompute deterministic netCDF dimension metadata maps for one file."""

        dim_id_to_name: dict[int, str] = {}
        dim_id_reference_list: dict[int, list[list[Any]]] = {}
        keys = list(file_handle.keys())

        var_dim_coords: dict[str, list[int]] = {}
        for name in keys:
            attrs = dict(getattr(file_handle[name], "attrs", {}))

            raw_dim_id = attrs.get("_Netcdf4Dimid")
            if raw_dim_id is not None:
                try:
                    dim_id_to_name[self._coerce_int(raw_dim_id)] = str(name)
                except (TypeError, ValueError):
                    pass

            coords = attrs.get("_Netcdf4Coordinates")
            if hasattr(coords, "tolist") and callable(coords.tolist):
                coords = coords.tolist()
            if isinstance(coords, (list, tuple)):
                parsed: list[int] = []
                for coord in coords:
                    try:
                        parsed.append(self._coerce_int(coord))
                    except (TypeError, ValueError):
                        parsed.append(-1)
                var_dim_coords[str(name)] = parsed

        for varname, coord_ids in var_dim_coords.items():
            for axis, dim_id in enumerate(coord_ids):
                if dim_id < 0:
                    continue
                dim_id_reference_list.setdefault(dim_id, []).append([varname, int(axis)])

        self._dim_id_to_name[path] = dim_id_to_name
        self._dim_id_reference_list[path] = dim_id_reference_list

    def _coerce_int(self, value: Any) -> int:
        """Convert numpy/list/tuple scalar-like values to int."""

        if hasattr(value, "item") and callable(value.item):
            with suppress(Exception):
                value = value.item()
        if hasattr(value, "tolist") and callable(value.tolist):
            with suppress(Exception):
                value = value.tolist()
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError(f"cannot coerce non-scalar sequence to int: {value!r}")
            value = value[0]
        return int(value)

    def _resolve_netcdf_reference_attrs(self, path: str, varname: str, attrs: dict[str, Any]) -> dict[str, Any]:
        """Resolve netCDF reference-style attrs to concrete names before transport."""

        resolved = dict(attrs)
        dim_id_to_name = self._dim_id_to_name.get(path, {})
        dim_id_reference_list = self._dim_id_reference_list.get(path, {})

        dim_list = resolved.get("DIMENSION_LIST")
        if isinstance(dim_list, list) and any(item == [None] for item in dim_list):
            coords = resolved.get("_Netcdf4Coordinates")
            if hasattr(coords, "tolist") and callable(coords.tolist):
                coords = coords.tolist()
            if isinstance(coords, (list, tuple)):
                rebuilt: list[list[str | None]] = []
                for coord in coords:
                    try:
                        rebuilt.append([dim_id_to_name.get(self._coerce_int(coord))])
                    except (TypeError, ValueError):
                        rebuilt.append([None])
                resolved["DIMENSION_LIST"] = rebuilt

        ref_list = resolved.get("REFERENCE_LIST")
        if isinstance(ref_list, list) and any(isinstance(item, list) and item and item[0] is None for item in ref_list):
            raw_dim_id = resolved.get("_Netcdf4Dimid")
            try:
                dim_id = self._coerce_int(raw_dim_id)
            except (TypeError, ValueError):
                dim_id = None

            if dim_id is not None:
                refs = dim_id_reference_list.get(dim_id, [])
                refs = [entry for entry in refs if entry[0] != varname]
                if refs:
                    resolved["REFERENCE_LIST"] = refs

        return resolved

    def _serialise(self, value: Any, *, file_handle: Any | None = None) -> Any:
        if file_handle is not None and value is not None:
            if hasattr(value, "address_of_reference"):
                with suppress(Exception):
                    obj = file_handle._get_object_by_address(value.address_of_reference)
                    if obj is not None:
                        name = getattr(obj, "name", None)
                        if isinstance(name, str):
                            return name.lstrip("/")

            with suppress(Exception):
                target = file_handle[value]
                name = getattr(target, "name", None)
                if isinstance(name, str):
                    return name.lstrip("/")

        if hasattr(value, "tolist") and callable(value.tolist):
            return self._serialise(value.tolist(), file_handle=file_handle)
        if hasattr(value, "item") and callable(value.item):
            with suppress(Exception):
                return self._serialise(value.item(), file_handle=file_handle)
        if isinstance(value, Mapping):
            return {key: self._serialise(item, file_handle=file_handle) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [self._serialise(item, file_handle=file_handle) for item in value]
        if not isinstance(value, (bool, int, float, str, bytes, type(None))):
            return None
        return value


if __name__ == "__main__":
    ServerStub().serve_forever()
