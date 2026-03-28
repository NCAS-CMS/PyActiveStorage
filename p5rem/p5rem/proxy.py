"""Pyfive-like proxy objects backed by a p5rem session."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


class p5remDataset:
	"""Lazy proxy for a remote dataset."""

	def __init__(self, session: Any, path: str, varname: str) -> None:
		self._session = session
		self._path = path
		self._varname = varname
		self._meta: dict[str, Any] | None = None

	def __repr__(self) -> str:
		return f"p5remDataset(path={self._path!r}, varname={self._varname!r})"

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

		return dict(self._ensure_meta().get("attrs", {}))

	@property
	def shape(self) -> tuple[Any, ...]:
		"""Return the dataset shape."""

		return tuple(self._ensure_meta().get("shape", ()))

	@property
	def dtype(self) -> Any:
		"""Return the dataset dtype value from server metadata."""

		return self._ensure_meta().get("dtype")

	@property
	def chunks(self) -> tuple[Any, ...] | None:
		"""Return chunk metadata, normalised to a tuple when present."""

		chunks = self._ensure_meta().get("chunks")
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

	def refresh(self) -> dict[str, Any]:
		"""Reload dataset metadata from the server."""

		self._meta = dict(self._session.var_open(self._path, self._varname))
		return dict(self._meta)

	def get_chunk(self, byte_offset: int, size: int, **fields: Any) -> dict[str, Any]:
		"""Delegate chunk retrieval to the session."""

		return self._session.get_chunk(
			self._path,
			self._varname,
			byte_offset,
			size,
			**fields,
		)

	def reduce(self, byte_offset: int, size: int, operation: str, **fields: Any) -> dict[str, Any]:
		"""Delegate remote reduction requests to the session."""

		return self._session.reduce(
			self._path,
			self._varname,
			byte_offset,
			size,
			operation,
			**fields,
		)

	def _ensure_meta(self) -> dict[str, Any]:
		if self._meta is None:
			self.refresh()
		return self._meta


class p5remProxy:
	"""Pyfive-like file proxy backed by remote metadata calls."""

	def __init__(self, session: Any, path: str) -> None:
		self._session = session
		self._path = path
		self._meta = dict(session.file_open(path))
		self._datasets: dict[str, p5remDataset] = {}
		self._closed = False

	def __repr__(self) -> str:
		return f"p5remProxy(path={self._path!r})"

	def __enter__(self) -> p5remProxy:
		return self

	def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
		self.close()

	def __contains__(self, key: object) -> bool:
		return key in self._keys

	def __getitem__(self, varname: str) -> p5remDataset:
		self._ensure_open()
		if varname not in self:
			raise KeyError(varname)
		if varname not in self._datasets:
			self._datasets[varname] = p5remDataset(self._session, self._path, varname)
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

	def items(self) -> list[tuple[str, p5remDataset]]:
		"""Return dataset name and proxy pairs."""

		return [(name, self[name]) for name in self._keys]

	def values(self) -> list[p5remDataset]:
		"""Return dataset proxies in file order."""

		return [self[name] for name in self._keys]

	def refresh(self) -> dict[str, Any]:
		"""Reload file metadata from the server."""

		self._ensure_open()
		self._meta = dict(self._session.file_open(self._path))
		self._datasets.clear()
		return dict(self._meta)

	def close(self) -> None:
		"""Release the remote file handle once."""

		if self._closed:
			return
		self._session.file_close(self._path)
		self._closed = True
		self._datasets.clear()

	def _ensure_open(self) -> None:
		if self._closed:
			raise ValueError("I/O operation on closed p5remProxy")

	@property
	def _keys(self) -> tuple[str, ...]:
		return tuple(self._meta.get("keys", ()))


__all__ = ["p5remDataset", "p5remProxy"]