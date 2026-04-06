"""Pyfive-like proxy objects backed by a p5rem session."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import logging
from typing import Any

import numpy as np
import pyfive
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

	def __init__(self, session: Any, path: str, varname: str, file: Any = None) -> None:
		self._session = session
		self._path = path
		self._varname = varname
		self._file = file
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
		"""Return the dataset name within the file, matching pyfive semantics."""

		# pyfive returns names with leading slash, e.g., "/tas"
		return f"/{self._varname}"

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
	def maxshape(self) -> tuple[Any, ...]:
		"""Return the maximum shape of the dataset, matching pyfive API."""

		return tuple(self.id._meta.maxshape)

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

	@property
	def ndim(self) -> int:
		"""Return the number of dimensions in the dataset."""

		return len(self.shape)

	@property
	def size(self) -> int:
		"""Return the total number of elements in the dataset."""

		import functools
		import operator
		if not self.shape:
			return 1
		return functools.reduce(operator.mul, self.shape, 1)

	@property
	def dims(self) -> tuple[str, ...]:
		"""Return dimension names, matching pyfive semantics (currently all generic)."""

		# pyfive uses dimension names from HDF5; for now return generic dimension labels
		# This matches pyfive behavior where dims are the parent Dataset's dimension tuples
		return tuple(f"phony_dim_{i}" for i in range(self.ndim))

	@property
	def parent(self) -> Any:
		"""Return reference to the parent file object."""

		# For remote files, we don't have direct access to the file object
		# Return None to indicate no parent; cfdm may handle this gracefully
		return None

	@property
	def file(self) -> Any:
		"""Return the parent rFile, matching pyfive.Dataset semantics."""

		return self._file

	@property
	def value(self) -> np.ndarray:
		"""Return the full dataset as a NumPy array."""

		# This tries to read all data at once - potentially large!
		# For efficiency, cfdm should prefer using [] slicing
		return self[()]

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
		host = getattr(self._session, "host", None)
		if isinstance(host, str) and host:
			short_host = host.split(".", 1)[0]
			return f"rFile(host={short_host!r}, path={self._path!r})"
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
			self._datasets[varname] = rDataset(self._session, self._path, varname, file=self)
		return self._datasets[varname]

	def __iter__(self) -> Iterator[str]:
		return iter(self._keys)

	def __len__(self) -> int:
		return len(self._keys)

	@property
	def name(self) -> str:
		"""Return the file-level group name, matching pyfive root semantics."""

		return "/"

	@property
	def file(self) -> "rFile":
		"""Return self, matching pyfive.File semantics."""

		return self

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

	def keys(self) -> Any:
		"""Return dataset names in the remote file with KeysView, matching pyfive."""

		self._ensure_open()
		# Return dict_keys which is a KeysView subclass
		# This matches pyfive's behavior of returning a live view of available datasets
		return dict({k: None for k in self._keys}).keys()

	def items(self) -> list[tuple[str, rDataset]]:
		"""Return dataset name and proxy pairs."""

		return [(name, self[name]) for name in self._keys]

	def values(self) -> list[rDataset]:
		"""Return dataset proxies in file order."""

		return [self[name] for name in self._keys]

	def get(self, key: str, default: Any = None) -> Any:
		"""Get a dataset by name, returning default if not found (dict-like)."""

		self._ensure_open()
		try:
			return self[key]
		except KeyError:
			return default

	def visit(self, func: Any) -> None:
		"""Call func on each dataset name, matching pyfive.File.visit() behavior."""

		self._ensure_open()
		for name in self._keys:
			func(name)

	def visititems(self, func: Any) -> None:
		"""Call func(name, dataset) on each dataset, matching pyfive semantics."""

		self._ensure_open()
		for name in self._keys:
			func(name, self[name])


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


# Let external libraries (e.g. cfdm) recognize rFile via isinstance(..., pyfive.File)
# without inheriting pyfive.File implementation internals.
pyfive.File.register(rFile)
pyfive.Dataset.register(rDataset)

__all__ = ["rDataset", "rFile"]