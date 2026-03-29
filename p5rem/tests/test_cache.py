"""Tests for persistent metadata/chunk caching."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from p5rem.cache import P5RemCache
from p5rem.protocol import CHUNK_DATA, FILE_INFO, FILE_OPEN, GET_CHUNK, STAT, STAT_RESULT, VAR_INFO, VAR_OPEN
from p5rem.session import p5remSession


def test_cache_clear_scoped_by_host(tmp_path: Path) -> None:
	cache = P5RemCache(str(tmp_path / "cache"), size_limit=10 * 2**20)
	try:
		cache.set_file_meta("host-a", "/a.nc", 1.0, {"type": FILE_INFO, "keys": ["x"]})
		cache.set_var_meta("host-a", "/a.nc", "x", 1.0, {"type": VAR_INFO, "dtype": "float32"})
		cache.set_chunk(
			"host-a",
			"/a.nc",
			100,
			10,
			1.0,
			{"type": CHUNK_DATA, "data": b"1234567890", "size": 10},
		)
		cache.set_file_meta("host-b", "/b.nc", 1.0, {"type": FILE_INFO, "keys": ["y"]})

		removed = cache.clear(host="host-a")
		assert removed == 3
		assert cache.get_file_meta("host-a", "/a.nc", 1.0) is None
		assert cache.get_file_meta("host-b", "/b.nc", 1.0) is not None
	finally:
		cache.close()


class _CountingSession(p5remSession):
	"""Session double for verifying cache short-circuiting."""

	def __init__(self, cache: P5RemCache) -> None:
		super().__init__(
			host="cache-host",
			stdin=BytesIO(),
			stdout=BytesIO(),
			cache=cache,
		)
		self.calls: dict[str, int] = {
			STAT: 0,
			FILE_OPEN: 0,
			VAR_OPEN: 0,
			GET_CHUNK: 0,
		}

	def request(self, message_type: str, /, *, expected_type=None, **fields):
		self.calls[message_type] = self.calls.get(message_type, 0) + 1
		path = fields.get("path", "/data/example.nc")

		if message_type == STAT:
			return {"type": STAT_RESULT, "path": path, "mtime": 123.0}
		if message_type == FILE_OPEN:
			return {
				"type": FILE_INFO,
				"path": path,
				"keys": ["temperature"],
				"attrs": {},
				"mtime": 123.0,
			}
		if message_type == VAR_OPEN:
			return {
				"type": VAR_INFO,
				"path": path,
				"varname": fields["varname"],
				"shape": [1],
				"dtype": "float32",
				"chunks": [1],
				"index": [
					{
						"chunk_offset": [0],
						"byte_offset": 64,
						"size": 4,
						"filter_mask": 0,
					}
				],
				"attrs": {},
				"fillvalue": None,
				"filter_pipeline": None,
				"order": "C",
				"layout": "chunked",
				"fragmented": False,
			}
		if message_type == GET_CHUNK:
			return {
				"type": CHUNK_DATA,
				"path": path,
				"varname": fields["varname"],
				"byte_offset": int(fields["byte_offset"]),
				"size": int(fields["size"]),
				"filter_mask": 0,
				"data": b"ABCD",
			}
		raise AssertionError(f"unexpected request type: {message_type}")


def test_session_caches_file_var_and_chunk(tmp_path: Path) -> None:
	cache = P5RemCache(str(tmp_path / "cache"), size_limit=10 * 2**20)
	try:
		session = _CountingSession(cache)
		path = "/data/example.nc"

		first_file = session.file_open(path)
		second_file = session.file_open(path)
		assert first_file == second_file

		first_var = session.var_open(path, "temperature")
		second_var = session.var_open(path, "temperature")
		assert first_var == second_var

		first_chunk = session.get_chunk(path, "temperature", 64, 4)
		second_chunk = session.get_chunk(path, "temperature", 64, 4)
		assert first_chunk == second_chunk

		# file_open performs STAT on each open for mtime validation.
		assert session.calls[STAT] == 2
		assert session.calls[FILE_OPEN] == 1
		assert session.calls[VAR_OPEN] == 1
		assert session.calls[GET_CHUNK] == 1
	finally:
		cache.close()