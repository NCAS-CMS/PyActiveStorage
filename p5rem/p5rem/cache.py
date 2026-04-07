"""Persistent metadata and chunk cache helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from diskcache import Cache
import logging

log = logging.getLogger(__name__)


class P5RemCache:
	"""Persistent cache wrapper for metadata and chunk payloads."""

	def __init__(self, directory: str | None = None, *, size_limit: int = 10 * 2**30) -> None:
		cache_dir = Path(directory).expanduser() if directory is not None else Path.home() / ".cache" / "p5rem"
		self._cache = Cache(str(cache_dir), size_limit=size_limit)
		log.info(f"Initialized p54rem cache at {cache_dir} with size limit {size_limit} bytes")
		self._hits = 0
		self._misses = 0

	def close(self) -> None:
		self._cache.close()

	def _file_key(self, host: str, path: str, mtime: float) -> tuple[str, str, str, float]:
		return ("file", host, path, float(mtime))

	def _var_key(self, host: str, path: str, varname: str, mtime: float) -> tuple[str, str, str, str, float]:
		return ("var", host, path, varname, float(mtime))

	def _chunk_key(
		self,
		host: str,
		path: str,
		byte_offset: int,
		size: int,
		mtime: float,
	) -> tuple[str, str, str, int, int, float]:
		return ("chunk", host, path, int(byte_offset), int(size), float(mtime))

	def _cache_get(self, key: Any) -> Any:
		value = self._cache.get(key, default=None)
		if value is None:
			self._misses += 1
		else:
			self._hits += 1
		return value

	def get_file_meta(self, host: str, path: str, mtime: float) -> dict[str, Any] | None:
		value = self._cache_get(self._file_key(host, path, mtime))
		if value is None:
			return None
		return dict(value)

	def set_file_meta(self, host: str, path: str, mtime: float, value: dict[str, Any]) -> None:
		self._cache[self._file_key(host, path, mtime)] = dict(value)

	def get_var_meta(self, host: str, path: str, varname: str, mtime: float) -> dict[str, Any] | None:
		value = self._cache_get(self._var_key(host, path, varname, mtime))
		if value is None:
			return None
		return dict(value)

	def set_var_meta(self, host: str, path: str, varname: str, mtime: float, value: dict[str, Any]) -> None:
		self._cache[self._var_key(host, path, varname, mtime)] = dict(value)

	def get_chunk(
		self,
		host: str,
		path: str,
		byte_offset: int,
		size: int,
		mtime: float,
	) -> dict[str, Any] | None:
		value = self._cache_get(self._chunk_key(host, path, byte_offset, size, mtime))
		if value is None:
			return None
		return dict(value)

	def set_chunk(
		self,
		host: str,
		path: str,
		byte_offset: int,
		size: int,
		mtime: float,
		value: dict[str, Any],
	) -> None:
		self._cache[self._chunk_key(host, path, byte_offset, size, mtime)] = dict(value)

	def clear(self, host: str | None = None) -> int:
		if host is None:
			removed = len(self._cache)
			self._cache.clear()
			return removed

		removed = 0
		for key in list(self._cache.iterkeys()):
			if isinstance(key, tuple) and len(key) > 1 and key[1] == host:
				del self._cache[key]
				removed += 1
		return removed

	def info(self) -> dict[str, Any]:
		return {
			"entries": int(len(self._cache)),
			"size_bytes": int(self._cache.volume()),
			"hits": self._hits,
			"misses": self._misses,
		}

	def transact(self):
		return self._cache.transact()


_default_cache: P5RemCache | None = None


def get_default_cache() -> P5RemCache:
	"""Return a process-wide shared cache instance."""

	global _default_cache
	if _default_cache is None:
		_default_cache = P5RemCache()
	return _default_cache


def clear(host: str | None = None) -> int:
	"""Clear cache entries globally or for one host."""

	return get_default_cache().clear(host=host)


def info() -> dict[str, Any]:
	"""Return shared cache metrics and size stats."""

	return get_default_cache().info()


__all__ = [
	"P5RemCache",
	"clear",
	"get_default_cache",
	"info",
]