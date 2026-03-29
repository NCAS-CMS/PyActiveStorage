"""Tests for the pyfive-like proxy layer."""

from __future__ import annotations

import numpy as np
import pytest

from p5rem import Session, rDataset, rFile


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
		if varname == "temperature":
			return {
				"shape": [2, 3],
				"dtype": "float32",
				"chunks": [2, 3],
				"index": [
					{
						"chunk_offset": [0, 0],
						"byte_offset": 100,
						"size": 24,
						"filter_mask": 0,
					}
				],
				"attrs": {"units": "K"},
				"fillvalue": None,
				"filter_pipeline": None,
				"order": "C",
				"layout": "chunked",
				"fragmented": False,
			}
		return {
			"shape": [4],
			"dtype": "float32",
			"chunks": None,
			"index": [
				{
					"chunk_offset": [0],
					"byte_offset": 200,
					"size": 16,
					"filter_mask": 0,
				}
			],
			"attrs": {},
			"fillvalue": None,
			"filter_pipeline": None,
			"order": "C",
			"layout": "contiguous",
			"fragmented": varname == "pressure",
		}

	def get_chunk(self, path: str, varname: str, byte_offset: int, size: int, **fields: object) -> dict[str, object]:
		self.chunk_calls.append((path, varname, byte_offset, size, fields))
		if varname == "temperature":
			data = np.arange(6, dtype=np.float32).tobytes()
			return {"type": "CHUNK_DATA", "data": data, "size": len(data)}
		if varname == "pressure":
			data = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32).tobytes()
			return {"type": "CHUNK_DATA", "data": data, "size": len(data)}
		return {"type": "CHUNK_DATA", "data": b"", "size": 0}

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

		proxy = rFile(session, "/data/example.nc")

		assert proxy.keys() == ["temperature", "pressure"]
		assert proxy.attrs == {"title": "example"}
		assert proxy.mtime == 12345
		assert session.file_open_calls == ["/data/example.nc"]


def test_session_open_returns_proxy() -> None:
		session = MockSession()

		proxy = Session.open(session, "/data/example.nc")

		assert isinstance(proxy, rFile)
		assert proxy.filename == "/data/example.nc"


def test_dataset_metadata_is_loaded_lazily_and_cached() -> None:
		session = MockSession()
		proxy = rFile(session, "/data/example.nc")

		dataset = proxy["temperature"]

		assert isinstance(dataset, rDataset)
		assert session.var_open_calls == []
		assert dataset.shape == (2, 3)
		assert dataset.dtype == np.dtype("float32")
		assert dataset.chunks == (2, 3)
		assert isinstance(dataset.index, list)
		assert dataset.attrs == {"units": "K"}
		assert dataset.fragmented is False
		assert session.var_open_calls == [("/data/example.nc", "temperature")]
		assert dataset.shape == (2, 3)
		assert session.var_open_calls == [("/data/example.nc", "temperature")]


def test_dataset_operations_delegate_to_session() -> None:
		session = MockSession()
		dataset = rFile(session, "/data/example.nc")["pressure"]

		chunk_response = dataset.get_chunk(1024, 4096, storeinfo={"offset": 1024})
		reduction_response = dataset.reduce(1024, 4096, "mean", axis=0)

		assert chunk_response["data"] == np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32).tobytes()
		assert reduction_response["value"] == 42.0
		assert session.chunk_calls == [
			("/data/example.nc", "pressure", 1024, 4096, {"storeinfo": {"offset": 1024}})
		]
		assert session.reduce_calls == [
			("/data/example.nc", "pressure", 1024, 4096, "mean", {"axis": 0})
		]


def test_dataset_getitem_reads_chunked_data() -> None:
		session = MockSession()
		dataset = rFile(session, "/data/example.nc")["temperature"]

		result = dataset[:, 1:]

		expected = np.arange(6, dtype=np.float32).reshape(2, 3)[:, 1:]
		assert np.array_equal(result, expected)
		assert session.chunk_calls == [
			("/data/example.nc", "temperature", 100, 24, {"chunk_coord": [0, 0]})
		]


def test_dataset_getitem_reads_contiguous_data() -> None:
		session = MockSession()
		dataset = rFile(session, "/data/example.nc")["pressure"]

		result = dataset[1:3]

		expected = np.array([20.0, 30.0], dtype=np.float32)
		assert np.array_equal(result, expected)
		assert session.chunk_calls == [
			("/data/example.nc", "pressure", 200, 16, {})
		]


def test_proxy_context_manager_closes_file_once() -> None:
		session = MockSession()

		with rFile(session, "/data/example.nc") as proxy:
			assert proxy["temperature"].name == "temperature"

		assert session.file_close_calls == ["/data/example.nc"]
		assert proxy.closed is True


def test_closed_proxy_rejects_further_access() -> None:
		session = MockSession()
		proxy = rFile(session, "/data/example.nc")
		proxy.close()

		with pytest.raises(ValueError, match="closed rFile"):
			proxy.keys()