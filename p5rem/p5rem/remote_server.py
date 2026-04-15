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
import numpy as np

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
REDUCTION_RESULT = "REDUCTION_RESULT"
ERROR = "ERROR"

ALL_TYPES = frozenset({
    LIST, STAT, FILE_OPEN, VAR_OPEN, GET_CHUNK, GET_CHUNKS, REDUCE, FILE_CLOSE, HEARTBEAT,
    LIST_RESULT, STAT_RESULT, FILE_INFO, VAR_INFO, CHUNK_DATA, CHUNKS_DONE, REDUCTION_RESULT, ERROR,
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

# --------------------
# Standard Reductions 
# --------------------

standard_reductions = {
    "sum": lambda x: np.sum(x),
    "mean": lambda x: np.mean(x),
    "max": lambda x: np.max(x),
    "min": lambda x: np.min(x),
    "range": lambda x: (np.min(x), np.max(x)),
    "count": lambda x: int(np.size(x)),
    "argmin": lambda x: np.argmin(x) if np.size(x) else None,
    "argmax": lambda x: np.argmax(x) if np.size(x) else None,
}


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

    def handle_reduce(
        self,
        path: str,
        varname: str,
        operation: str,
        byte_offset: int | None = None,
        size: int | None = None,
        mode: str | None = None,
        selection: Any | None = None,
        thread_count: int = 1,
        **fields: Any,
    ) -> dict[str, Any]:
        dataset = self._get_dataset(path, varname)
        op = str(operation)
        reducer = standard_reductions.get(op)
        if reducer is None:
            supported = ", ".join(sorted(standard_reductions))
            raise ValueError(f"unsupported reduction operation: {op!r}; supported operations: {supported}")

        resolved_mode = str(mode) if mode is not None else ("selection" if selection is not None else "chunk")
        if resolved_mode not in {"chunk", "selection"}:
            raise ValueError(f"invalid reduction mode: {resolved_mode!r}")

        if resolved_mode == "selection":
            workers = max(1, int(thread_count))
            if getattr(dataset, "chunks", None) is not None:
                value = self._parallel_reduce_selection(path, dataset, op, workers, selection)
                if value is None:
                    raise NotImplementedError(
                        "selection reduction could not be planned chunk-wise; refusing array materialization"
                    )
            else:
                value = self._reduce_contiguous_selection(path, dataset, op, selection)
            value = self._serialise(value, file_handle=self._open_files.get(path))
            return {
                "type": REDUCTION_RESULT,
                "path": path,
                "varname": varname,
                "operation": op,
                "mode": "selection",
                "thread_count": workers,
                "value": value,
            }

        if byte_offset is None or size is None:
            raise ValueError("chunk reduction requires byte_offset and size")

        data, _filter_mask, resolved_offset, resolved_size = self._read_raw_data(
            path,
            dataset,
            int(byte_offset),
            int(size),
            fields,
        )
        reduced_input = self._coerce_chunk_reduction_input(dataset, data)
        value = self._serialise(reducer(reduced_input), file_handle=self._open_files.get(path))
        return {
            "type": REDUCTION_RESULT,
            "path": path,
            "varname": varname,
            "operation": op,
            "mode": "chunk",
            "byte_offset": resolved_offset,
            "size": resolved_size,
            "value": value,
        }

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

    def _read_selection_data(self, dataset: Any, selection: Any | None) -> np.ndarray[Any, Any]:
        if selection is None:
            return np.asarray(dataset[()])

        indexer = self._decode_selection_spec(selection)
        return np.asarray(dataset[indexer])

    def _decode_selection_spec(self, spec: Any) -> Any:
        if spec is None:
            return slice(None)

        if isinstance(spec, bool):
            return int(spec)

        if isinstance(spec, (int, np.integer)):
            return int(spec)

        if isinstance(spec, dict):
            if spec.get("type") == "index":
                return int(spec["value"])
            if spec.get("type") in {None, "slice"} and any(key in spec for key in ("start", "stop", "step")):
                return slice(spec.get("start"), spec.get("stop"), spec.get("step"))
            raise ValueError(f"unsupported selection spec dict: {spec!r}")

        if isinstance(spec, (list, tuple)):
            if len(spec) == 3 and all(not isinstance(item, (list, tuple, dict)) for item in spec):
                return slice(spec[0], spec[1], spec[2])
            return tuple(self._decode_selection_spec(item) for item in spec)

        raise TypeError(f"unsupported selection spec type: {type(spec).__name__}")

    def _coerce_chunk_reduction_input(self, dataset: Any, data: bytes) -> np.ndarray[Any, Any]:
        dtype = np.dtype(getattr(dataset, "dtype", np.uint8))
        itemsize = int(getattr(dtype, "itemsize", 1))
        if itemsize > 0 and len(data) % itemsize == 0:
            return np.frombuffer(data, dtype=dtype)
        return np.frombuffer(data, dtype=np.uint8)

    def _parallel_reduce_selection(
        self,
        path: str,
        dataset: Any,
        operation: str,
        thread_count: int,
        selection: Any | None,
    ) -> Any | None:
        """Attempt chunk-parallel reduction for chunked datasets.

        Falls back to serial mode by returning None when unsupported.
        """

        if operation not in {"sum", "mean", "min", "max", "range", "count", "argmin", "argmax"}:
            return None

        dataset_id = getattr(dataset, "id", None)
        chunks = getattr(dataset, "chunks", None)
        if dataset_id is None or chunks is None:
            return None

        shape = tuple(getattr(dataset, "shape", ()))
        if not shape:
            return None

        try:
            from pyfive.h5d import ZarrArrayStub
            from pyfive.indexing import OrthogonalIndexer
        except ImportError:
            return None

        try:
            indexer_args = self._selection_to_indexer_args(selection, len(shape))
            indexer = OrthogonalIndexer(indexer_args, ZarrArrayStub(shape, tuple(chunks)))
            out_shape = tuple(int(x) for x in indexer.shape)
            work_items = list(dataset_id._get_required_chunks(indexer))
        except Exception:
            return None

        if not work_items:
            if operation == "count":
                return 0
            return None

        workers = min(max(1, int(thread_count)), len(work_items))

        decode_chunk = getattr(dataset_id, "_decode_chunk", None)
        dtype = np.dtype(getattr(dataset, "dtype", np.uint8))
        chunk_shape = tuple(int(size) for size in chunks)

        fh = open(path, "rb")
        fd = fh.fileno()

        def _reduce_one(item: tuple[Any, Any, Any, Any]) -> Any:
            _chunk_coords, chunk_selection, out_selection, storeinfo = item
            raw = os.pread(fd, int(storeinfo.size), int(storeinfo.byte_offset))
            if callable(decode_chunk):
                chunk_array = decode_chunk(raw, int(getattr(storeinfo, "filter_mask", 0)), dtype)
            else:
                chunk_array = np.frombuffer(raw, dtype=dtype).reshape(chunk_shape, order=getattr(dataset_id, "_order", "C"))
            slab = np.asarray(chunk_array[chunk_selection])
            if operation == "count":
                return int(slab.size)
            if slab.size == 0:
                return None
            if operation == "sum":
                return np.sum(slab)
            if operation == "mean":
                return (np.sum(slab), int(slab.size))
            if operation == "min":
                return np.min(slab)
            if operation == "max":
                return np.max(slab)
            if operation == "argmin":
                local_idx = int(np.argmin(slab))
                local_coords = np.unravel_index(local_idx, slab.shape, order="C")
                global_idx = self._out_selection_local_to_flat_index(out_selection, local_coords, out_shape)
                if global_idx is None:
                    return None
                return (slab.reshape(-1, order="C")[local_idx], int(global_idx))
            if operation == "argmax":
                local_idx = int(np.argmax(slab))
                local_coords = np.unravel_index(local_idx, slab.shape, order="C")
                global_idx = self._out_selection_local_to_flat_index(out_selection, local_coords, out_shape)
                if global_idx is None:
                    return None
                return (slab.reshape(-1, order="C")[local_idx], int(global_idx))
            # operation == "range"
            return (np.min(slab), np.max(slab))

        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                partials = list(pool.map(_reduce_one, work_items))
        except Exception:
            return None
        finally:
            fh.close()

        if operation == "count":
            return int(sum(int(value) for value in partials))

        filtered = [value for value in partials if value is not None]
        if not filtered:
            return None

        if operation == "sum":
            return np.sum(np.asarray(filtered))
        if operation == "mean":
            total = np.sum(np.asarray([value[0] for value in filtered]))
            count = int(sum(int(value[1]) for value in filtered))
            if count == 0:
                return float("nan")
            return total / count
        if operation == "min":
            return np.min(np.asarray(filtered))
        if operation == "max":
            return np.max(np.asarray(filtered))
        if operation == "argmin":
            best_value, best_index = filtered[0]
            for value, index in filtered[1:]:
                if value < best_value or (value == best_value and index < best_index):
                    best_value, best_index = value, index
            return int(best_index)
        if operation == "argmax":
            best_value, best_index = filtered[0]
            for value, index in filtered[1:]:
                if value > best_value or (value == best_value and index < best_index):
                    best_value, best_index = value, index
            return int(best_index)

        mins = np.asarray([value[0] for value in filtered])
        maxs = np.asarray([value[1] for value in filtered])
        return (np.min(mins), np.max(maxs))

    def _reduce_contiguous_selection(self, path: str, dataset: Any, operation: str, selection: Any | None) -> Any:
        """Reduce contiguous datasets without full-array materialization."""

        shape = tuple(getattr(dataset, "shape", ()))
        if self._is_full_selection(selection, len(shape)):
            return self._stream_reduce_contiguous_full(path, dataset, operation)

        # For non-full selections on contiguous datasets, only the requested
        # subset is materialised.
        data = self._read_selection_data(dataset, selection)
        reducer = standard_reductions[operation]
        return reducer(data)

    def _stream_reduce_contiguous_full(self, path: str, dataset: Any, operation: str) -> Any:
        dataset_id = getattr(dataset, "id", None)
        if dataset_id is None:
            raise ValueError("dataset id is missing")

        data_offset = getattr(dataset_id, "data_offset", None)
        if data_offset is None:
            raise ValueError("contiguous dataset does not expose data_offset")

        total_nbytes = self._dataset_nbytes(dataset)
        if total_nbytes <= 0:
            if operation == "count":
                return 0
            if operation in {"sum", "mean"}:
                return 0
            return None

        dtype = np.dtype(getattr(dataset, "dtype", np.uint8))
        itemsize = max(1, int(getattr(dtype, "itemsize", 1)))
        block_nbytes = max(itemsize, (4 * 1024 * 1024) // itemsize * itemsize)

        count = 0
        sum_value: Any = 0
        min_value: Any | None = None
        max_value: Any | None = None
        argmin_value: Any | None = None
        argmax_value: Any | None = None
        argmin_index = 0
        argmax_index = 0
        seen = 0

        with open(path, "rb") as handle:
            remaining = int(total_nbytes)
            offset = int(data_offset)
            while remaining > 0:
                read_size = min(remaining, block_nbytes)
                handle.seek(offset)
                raw = handle.read(read_size)
                if not raw:
                    break
                arr = np.frombuffer(raw, dtype=dtype)
                if arr.size == 0:
                    break

                if operation in {"count", "mean"}:
                    count += int(arr.size)
                if operation in {"sum", "mean"}:
                    sum_value = sum_value + np.sum(arr)
                if operation in {"min", "range"}:
                    block_min = np.min(arr)
                    if min_value is None or block_min < min_value:
                        min_value = block_min
                if operation in {"max", "range"}:
                    block_max = np.max(arr)
                    if max_value is None or block_max > max_value:
                        max_value = block_max
                if operation == "argmin":
                    local_idx = int(np.argmin(arr))
                    value = arr[local_idx]
                    global_idx = seen + local_idx
                    if argmin_value is None or value < argmin_value or (value == argmin_value and global_idx < argmin_index):
                        argmin_value = value
                        argmin_index = global_idx
                if operation == "argmax":
                    local_idx = int(np.argmax(arr))
                    value = arr[local_idx]
                    global_idx = seen + local_idx
                    if argmax_value is None or value > argmax_value or (value == argmax_value and global_idx < argmax_index):
                        argmax_value = value
                        argmax_index = global_idx

                consumed = len(raw)
                offset += consumed
                remaining -= consumed
                seen += int(arr.size)

        if operation == "count":
            return int(count)
        if operation == "sum":
            return sum_value
        if operation == "mean":
            return float("nan") if count == 0 else (sum_value / count)
        if operation == "min":
            return min_value
        if operation == "max":
            return max_value
        if operation == "range":
            return (min_value, max_value)
        if operation == "argmin":
            return int(argmin_index)
        if operation == "argmax":
            return int(argmax_index)
        raise ValueError(f"unsupported reduction operation: {operation!r}")

    def _is_full_selection(self, selection: Any | None, ndim: int) -> bool:
        if selection is None:
            return True

        decoded = self._decode_selection_spec(selection)
        if isinstance(decoded, tuple):
            parts = list(decoded)
        else:
            parts = [decoded]

        if len(parts) > ndim:
            return False
        parts.extend([slice(None)] * (ndim - len(parts)))
        return all(isinstance(part, slice) and part == slice(None) for part in parts)

    def _out_selection_local_to_flat_index(
        self,
        out_selection: Any,
        local_coords: tuple[int, ...],
        out_shape: tuple[int, ...],
    ) -> int | None:
        if not isinstance(out_selection, tuple):
            out_selection = (out_selection,)

        if len(out_selection) != len(local_coords) or len(out_selection) != len(out_shape):
            return None

        global_coords: list[int] = []
        for dim, (sel, local_coord) in enumerate(zip(out_selection, local_coords)):
            if not isinstance(sel, slice):
                return None
            start, _stop, step = sel.indices(out_shape[dim])
            global_coords.append(start + int(local_coord) * step)

        return int(np.ravel_multi_index(tuple(global_coords), out_shape, order="C"))

    def _selection_to_indexer_args(self, selection: Any | None, ndim: int) -> tuple[Any, ...]:
        """Normalise decoded selection for pyfive OrthogonalIndexer."""

        if selection is None:
            return tuple(slice(None) for _ in range(ndim))

        decoded = self._decode_selection_spec(selection)
        if isinstance(decoded, tuple):
            parts = list(decoded)
        else:
            parts = [decoded]

        if len(parts) > ndim:
            raise ValueError("selection rank exceeds dataset rank")

        if len(parts) < ndim:
            parts.extend([slice(None)] * (ndim - len(parts)))

        return tuple(parts)

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
