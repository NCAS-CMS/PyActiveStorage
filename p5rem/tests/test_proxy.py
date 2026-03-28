"""Tests for the pyfive-like proxy layer."""

from __future__ import annotations

import pytest

from p5rem import Session, p5remDataset, p5remProxy


class MockSession:
	def __init__(self) -> None:
		self.file_open_calls: list[str] = []
		self.var_open_calls: list[tuple[str, str]] = []
		self.file_close_calls: list[str] = []
		self.chunk_calls: list[tuple[str, str, int, int, dict[str, object]]] = []
		self.reduce_calls: list[tuple[str, str, int, int, str, dict[str, object]]] = []

	def file_open(self, path: str) -> dict[str, object]:
		self.file_open_calls.append(path)
		return {
			"keys": ["temperature", "pressure"],
			"attrs": {"title": "example"},
			"mtime": 12345,
		}

	def var_open(self, path: str, varname: str) -> dict[str, object]:
		self.var_open_calls.append((path, varname))
		return {
			"shape": [180, 360],
			"dtype": "float32",
			"chunks": [45, 90],
			"index": {"kind": "btree"},
			"fragmented": varname == "pressure",
		}

	def get_chunk(self, path: str, varname: str, byte_offset: int, size: int, **fields: object) -> dict[str, object]:
		self.chunk_calls.append((path, varname, byte_offset, size, fields))
		return {"type": "CHUNK_DATA", "data": b"abc", "size": size}

	def reduce(
		self,
		path: str,
		varname: str,
		byte_offset: int,
		size: int,
		operation: str,
		**fields: object,
	) -> dict[str, object]:
		self.reduce_calls.append((path, varname, byte_offset, size, operation, fields))
		return {"type": "REDUCTION_RESULT", "value": 42.0}

	def file_close(self, path: str) -> dict[str, object]:
		self.file_close_calls.append(path)
		return {"type": "FILE_CLOSE", "closed": True}


def test_proxy_loads_file_metadata_immediately() -> None:
		session = MockSession()

		proxy = p5remProxy(session, "/data/example.nc")

		assert proxy.keys() == ["temperature", "pressure"]
		assert proxy.attrs == {"title": "example"}
		assert proxy.mtime == 12345
		assert session.file_open_calls == ["/data/example.nc"]


def test_session_open_returns_proxy() -> None:
		session = MockSession()

		proxy = Session.open(session, "/data/example.nc")

		assert isinstance(proxy, p5remProxy)
		assert proxy.filename == "/data/example.nc"


def test_dataset_metadata_is_loaded_lazily_and_cached() -> None:
		session = MockSession()
		proxy = p5remProxy(session, "/data/example.nc")

		dataset = proxy["temperature"]

		assert isinstance(dataset, p5remDataset)
		assert session.var_open_calls == []
		assert dataset.shape == (180, 360)
		assert dataset.dtype == "float32"
		assert dataset.chunks == (45, 90)
		assert dataset.index == {"kind": "btree"}
		assert dataset.fragmented is False
		assert session.var_open_calls == [("/data/example.nc", "temperature")]
		assert dataset.shape == (180, 360)
		assert session.var_open_calls == [("/data/example.nc", "temperature")]


def test_dataset_operations_delegate_to_session() -> None:
		session = MockSession()
		dataset = p5remProxy(session, "/data/example.nc")["pressure"]

		chunk_response = dataset.get_chunk(1024, 4096, storeinfo={"offset": 1024})
		reduction_response = dataset.reduce(1024, 4096, "mean", axis=0)

		assert chunk_response["data"] == b"abc"
		assert reduction_response["value"] == 42.0
		assert session.chunk_calls == [
			("/data/example.nc", "pressure", 1024, 4096, {"storeinfo": {"offset": 1024}})
		]
		assert session.reduce_calls == [
			("/data/example.nc", "pressure", 1024, 4096, "mean", {"axis": 0})
		]


def test_proxy_context_manager_closes_file_once() -> None:
		session = MockSession()

		with p5remProxy(session, "/data/example.nc") as proxy:
			assert proxy["temperature"].name == "temperature"

		assert session.file_close_calls == ["/data/example.nc"]
		assert proxy.closed is True


def test_closed_proxy_rejects_further_access() -> None:
		session = MockSession()
		proxy = p5remProxy(session, "/data/example.nc")
		proxy.close()

		with pytest.raises(ValueError, match="closed p5remProxy"):
			proxy.keys()