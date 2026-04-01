"""Pyfive-like proxy objects backed by a p5rem session."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import logging
from typing import Any

import numpy as np
from pyfive.h5d import DatasetID, StoreInfo
from time import perf_counter

log = logging.getLogger(__name__)


class _AtomicProxyType:
	"""Minimal placeholder type matching pyfive atomic datatype expectations."""

	is_atomic = True

	def __init__(self, dtype: str | np.dtype[Any]) -> None:
		self.dtype = np.dtype(dtype)


@dataclass
class _RemoteDatasetMeta:
	"""Small container matching fields used by pyfive Dataset properties."""

	attributes: dict[str, Any]
	fillvalue: Any
	maxshape: tuple[Any, ...]
	compression: Any = None
	compression_opts: Any = None
	shuffle: bool = False
	fletcher32: bool = False


class _RemoteDatasetID(DatasetID):
	"""DatasetID implementation that fetches data via p5rem session calls."""

	def __init__(self, session: Any, path: str, varname: str, meta: dict[str, Any]) -> None:
		self._session = session
		self._path = path
		self._varname = varname
		self._filename = path
		self._order = str(meta.get("order", "C"))
		self.shape = tuple(meta.get("shape", ()))
		self.rank = len(self.shape)
		self.chunks = tuple(meta["chunks"]) if meta.get("chunks") is not None else None
		self.layout_class = 2 if self.chunks is not None else 1
		self.filter_pipeline = meta.get("filter_pipeline")
		self._ptype = _AtomicProxyType(meta["dtype"])
		self._global_heaps: dict[str, Any] = {}
		self.posix = False
		self.pseudo_chunking_size = 4 * 1024 * 1024
		self._msg_offset = 0
		self.property_offset = 0
		self._unique = (self._filename, self.shape, self._varname)
		self._index_params = None
		self._data = None
		self._meta = _RemoteDatasetMeta(
			attributes=dict(meta.get("attrs", {})),
			fillvalue=meta.get("fillvalue"),
			maxshape=self.shape,
		)

		index_entries = meta.get("index") or []
		if self.layout_class == 2:
			self._index = {
				tuple(entry["chunk_offset"]): StoreInfo(
					tuple(entry["chunk_offset"]),
					int(entry.get("filter_mask", 0)),
					int(entry["byte_offset"]),
					int(entry["size"]),
				)
				for entry in index_entries
			}
			self._DatasetID__index_built = True
		else:
			self._index = {}
			self._DatasetID__index_built = False
			if index_entries:
				self.data_offset = int(index_entries[0]["byte_offset"])
				self._contiguous_nbytes = int(index_entries[0]["size"])
			else:
				# Empty index means no data stored (UNDEFINED_ADDRESS in HDF5).
				# Use pyfive's own sentinel so get_data() returns the fill value
				# without attempting any network fetch.
				from pyfive.core import UNDEFINED_ADDRESS
				self.data_offset = UNDEFINED_ADDRESS
				self._contiguous_nbytes = 0

	def _build_index(self) -> None:
		"""Index is provided by server metadata and should not be rebuilt client-side."""

		raise RuntimeError("remote dataset index is supplied by server metadata")

	def _select_chunks(self, indexer: Any, out: Any, dtype: Any) -> None:
		"""Batch-fetch all required chunks in one GET_CHUNKS request."""

		chunks = self._get_required_chunks(indexer)
		if not chunks:
			return

		chunk_descs = [
			{
				"byte_offset": int(storeinfo.byte_offset),
				"size": int(storeinfo.size),
				"chunk_coord": list(storeinfo.chunk_offset),
			}
			for _coords, _chunk_sel, _out_sel, storeinfo in chunks
		]

		results = self._session.get_chunks(self._path, self._varname, chunk_descs)

		for _coords, chunk_sel, out_sel, storeinfo in chunks:
			chunk_response = results.get(int(storeinfo.byte_offset))
			if chunk_response is None:
				raise RuntimeError(f"missing chunk response for offset={storeinfo.byte_offset}")
			raw = chunk_response["data"]
			filter_mask = int(chunk_response.get("filter_mask", 0))
			out[out_sel] = self._decode_chunk(raw, filter_mask, dtype)[chunk_sel]

	def _get_raw_chunk(self, storeinfo: StoreInfo) -> bytes:
		"""Fetch one raw chunk payload from the remote server."""

		response = self._session.get_chunk(
			self._path,
			self._varname,
			int(storeinfo.byte_offset),
			int(storeinfo.size),
			chunk_coord=list(storeinfo.chunk_offset),
		)
		data = response.get("data")
		if not isinstance(data, bytes):
			raise TypeError("remote chunk response did not include byte payload")
		return data

	def _get_contiguous_data(self, args: Any, fillvalue: Any):
		"""Fetch contiguous bytes remotely and project the requested selection."""

		read_size = int(getattr(self, "_contiguous_nbytes", 0))
		response = self._session.get_chunk(
			self._path,
			self._varname,
			int(getattr(self, "data_offset", 0)),
			read_size,
		)
		raw = response.get("data")
		if not isinstance(raw, bytes):
			raise TypeError("remote contiguous response did not include byte payload")

		total_elems = int(np.prod(self.shape, dtype=int)) if self.shape else 1
		array = np.frombuffer(raw, dtype=self.dtype).copy()
		if array.size < total_elems:
			pad_value = 0 if fillvalue is None else fillvalue
			padded = np.full(total_elems, pad_value, dtype=self.dtype)
			padded[: array.size] = array
			array = padded
		elif array.size > total_elems:
			array = array[:total_elems]

		array = array.reshape(self.shape, order=self._order)
		return array[args]


class rDataset:
	"""Lazy proxy for a remote dataset."""

	def __init__(self, session: Any, path: str, varname: str) -> None:
		self._session = session
		self._path = path
		self._varname = varname
		self._meta: dict[str, Any] | None = None
		self._id: _RemoteDatasetID | None = None
		self._astype: np.dtype[Any] | None = None

	def __repr__(self) -> str:
		return f"HDF5 Dataset Remote Proxy (path={self._path!r}, varname={self._varname!r})"

	def __getitem__(self, args: Any):
		"""Return a NumPy view/value for the requested selection."""

		log.debug("Selection %r from %r in %s", args, self._varname, self._path)
		t1 = perf_counter()
		data = self.id.get_data(args, self.fillvalue)
		t2 = perf_counter()
		log.debug("Received selection data in %.2f seconds", t2 - t1)
		if self._astype is None:
			return data
		return data.astype(self._astype)

	@property
	def name(self) -> str:
		"""Return the dataset name within the file."""

		return self._varname

	@property
	def path(self) -> str:
		"""Return the remote file path that owns the dataset."""

		return self._path

	@property
	def attrs(self) -> dict[str, Any]:
		"""Return dataset attributes if the server provided them."""

		return dict(self.id._meta.attributes)

	@property
	def fillvalue(self) -> Any:
		"""Return dataset fill value."""

		return self.id._meta.fillvalue

	@property
	def shape(self) -> tuple[Any, ...]:
		"""Return the dataset shape."""

		return tuple(self.id.shape)

	@property
	def dtype(self) -> Any:
		"""Return the dataset dtype value from server metadata."""

		return self.id.dtype

	@property
	def chunks(self) -> tuple[Any, ...] | None:
		"""Return chunk metadata, normalised to a tuple when present."""

		chunks = self.id.chunks
		if chunks is None:
			return None
		return tuple(chunks)

	@property
	def index(self) -> Any:
		"""Return the server-serialised dataset index information."""

		return self._ensure_meta().get("index")

	@property
	def fragmented(self) -> bool:
		"""Return the server fragmentation hint for this dataset."""

		return bool(self._ensure_meta().get("fragmented", False))

	def _load_meta(self) -> dict[str, Any]:
		"""Fetch dataset metadata once from the server."""

		log.debug("Loading metadata for %r in %s", self._varname, self._path)
		return dict(self._session.var_open(self._path, self._varname))

	def astype(self, dtype: str | np.dtype[Any]) -> rDataset:
		"""Set conversion dtype for data returned from __getitem__."""

		self._astype = np.dtype(dtype)
		return self

	@property
	def id(self) -> _RemoteDatasetID:
		"""Return the remote-backed DatasetID instance."""

		if self._id is None:
			self._id = _RemoteDatasetID(self._session, self._path, self._varname, self._ensure_meta())
		return self._id

	def _ensure_meta(self) -> dict[str, Any]:
		if self._meta is None:
			self._meta = self._load_meta()
		return self._meta


class rFile:
	"""Pyfive-like file proxy backed by remote metadata calls."""

	def __init__(self, session: Any, path: str) -> None:
		self._session = session
		self._path = path
		log.debug("Opening remote file: %s", path)
		self._meta = dict(session.file_open(path))
		log.debug("Remote file opened: %s (%d top-level keys)", path, len(self._meta.get("keys", ())))
		self._datasets: dict[str, rDataset] = {}
		self._closed = False

	def __repr__(self) -> str:
		return f"rFile(path={self._path!r})"

	def __enter__(self) -> rFile:
		return self

	def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
		self.close()

	def __contains__(self, key: object) -> bool:
		return key in self._keys

	def __getitem__(self, varname: str) -> rDataset:
		self._ensure_open()
		if varname not in self:
			raise KeyError(varname)
		if varname not in self._datasets:
			self._datasets[varname] = rDataset(self._session, self._path, varname)
		return self._datasets[varname]

	def __iter__(self) -> Iterator[str]:
		return iter(self._keys)

	def __len__(self) -> int:
		return len(self._keys)

	@property
	def attrs(self) -> dict[str, Any]:
		"""Return file-level attributes from FILE_OPEN metadata."""

		self._ensure_open()
		return dict(self._meta.get("attrs", {}))

	@property
	def filename(self) -> str:
		"""Return the remote file path for this proxy."""

		return self._path

	@property
	def mtime(self) -> Any:
		"""Return the remote file modification time."""

		self._ensure_open()
		return self._meta.get("mtime")

	@property
	def closed(self) -> bool:
		"""Report whether the proxy has been closed."""

		return self._closed

	def keys(self) -> list[str]:
		"""Return dataset names in the remote file."""

		self._ensure_open()
		return list(self._keys)

	def items(self) -> list[tuple[str, rDataset]]:
		"""Return dataset name and proxy pairs."""

		return [(name, self[name]) for name in self._keys]

	def values(self) -> list[rDataset]:
		"""Return dataset proxies in file order."""

		return [self[name] for name in self._keys]


	def close(self) -> None:
		"""Release the remote file handle once."""

		if self._closed:
			return
		self._session.file_close(self._path)
		self._closed = True
		self._datasets.clear()

	def _ensure_open(self) -> None:
		if self._closed:
			raise ValueError("I/O operation on closed rFile")

	@property
	def _keys(self) -> tuple[str, ...]:
		return tuple(self._meta.get("keys", ()))

__all__ = ["rDataset", "rFile"]