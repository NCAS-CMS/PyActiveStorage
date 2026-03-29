"""End-to-end loopback tests for session and server transport."""

from __future__ import annotations

import socket
import threading
from contextlib import suppress
from pathlib import Path
from typing import Any

import pyfive
import pytest
import numpy as np

from p5rem.session import ResponseError, UnexpectedResponseError, p5remSession
from p5rem.server.stub import ServerStub


class LoopbackServer(ServerStub):
	def handle_file_open(self, path: str) -> dict[str, Any]:
		self._open_files[path] = {"temperature": object()}
		return {
			"type": "FILE_INFO",
			"path": path,
			"keys": ["temperature"],
			"attrs": {"title": "loopback"},
			"mtime": 12345,
		}

	def handle_var_open(self, path: str, varname: str) -> dict[str, Any]:
		self._get_dataset(path, varname)
		return {
			"type": "VAR_INFO",
			"path": path,
			"varname": varname,
			"shape": [2, 3],
			"dtype": "float32",
			"chunks": [1, 3],
			"index": [
				{
					"chunk_offset": [0, 0],
					"byte_offset": 10,
					"size": 4,
					"filter_mask": 0,
				}
			],
			"fragmented": False,
			"attrs": {"units": "K"},
			"fillvalue": None,
			"filter_pipeline": None,
			"order": "C",
		}

	def handle_get_chunk(self, path: str, varname: str, byte_offset: int, size: int, **fields: Any) -> dict[str, Any]:
		self._get_dataset(path, varname)
		return {
			"type": "CHUNK_DATA",
			"path": path,
			"varname": varname,
			"byte_offset": byte_offset,
			"size": size,
			"filter_mask": 0,
			"data": b"x" * size,
		}

	def handle_reduce(self, path: str, varname: str, byte_offset: int, size: int, operation: str, **fields: Any) -> dict[str, Any]:
		raise NotImplementedError("REDUCE handling is not implemented yet")


class WrongHeartbeatServer(ServerStub):
	def handle_heartbeat(self) -> dict[str, Any]:
		return {"type": "LIST_RESULT", "path": "/", "entries": []}


def _start_loopback_server(server_cls: type[ServerStub]) -> tuple[p5remSession, threading.Thread, socket.socket, socket.socket, Any, Any, Any, Any]:
	client_sock, server_sock = socket.socketpair()
	client_reader = client_sock.makefile("rb")
	client_writer = client_sock.makefile("wb")
	server_reader = server_sock.makefile("rb")
	server_writer = server_sock.makefile("wb")
	server = server_cls(server_reader, server_writer)
	thread = threading.Thread(target=server.serve_forever, daemon=True)
	thread.start()
	session = p5remSession(stdin=client_writer, stdout=client_reader)
	return session, thread, client_sock, server_sock, client_reader, client_writer, server_reader, server_writer


def _stop_loopback_server(
	session: p5remSession,
	thread: threading.Thread,
	client_sock: socket.socket,
	server_sock: socket.socket,
	client_reader: Any,
	client_writer: Any,
	server_reader: Any,
	server_writer: Any,
) -> None:
	with suppress(Exception):
		client_sock.shutdown(socket.SHUT_RDWR)
	with suppress(Exception):
		server_sock.shutdown(socket.SHUT_RDWR)
	with suppress(Exception):
		session.close()
	with suppress(Exception):
		client_reader.close()
	with suppress(Exception):
		client_writer.close()
	with suppress(Exception):
		server_reader.close()
	with suppress(Exception):
		server_writer.close()
	with suppress(Exception):
		client_sock.close()
	with suppress(Exception):
		server_sock.close()
	thread.join(timeout=1)


@pytest.fixture
def loopback_session(tmp_path):
	connection = _start_loopback_server(LoopbackServer)
	session = connection[0]
	try:
		yield session, tmp_path
	finally:
		_stop_loopback_server(*connection)


def test_loopback_list_and_stat(loopback_session) -> None:
	session, tmp_path = loopback_session
	data_file = tmp_path / "sample.nc"
	data_file.write_text("placeholder", encoding="utf-8")

	entries = session.list(str(tmp_path))
	stat_result = session.stat(str(data_file))

	assert entries == ["sample.nc"]
	assert stat_result["type"] == "STAT_RESULT"
	assert stat_result["path"] == str(data_file)
	assert stat_result["is_file"] is True
	assert stat_result["size"] == len("placeholder")


def test_loopback_proxy_round_trip(loopback_session) -> None:
	session, tmp_path = loopback_session
	path = str(tmp_path / "mock.nc")

	with session.open(path) as proxy:
		assert proxy.keys() == ["temperature"]
		assert proxy.attrs == {"title": "loopback"}
		assert proxy.mtime == 12345

		dataset = proxy["temperature"]
		assert dataset.shape == (2, 3)
		assert dataset.dtype == np.dtype("float32")
		assert dataset.chunks == (1, 3)
		assert dataset.attrs == {"units": "K"}

		chunk = dataset.get_chunk(10, 4)

		assert chunk["data"] == b"xxxx"
		with pytest.raises(ResponseError, match="not implemented"):
			dataset.reduce(10, 4, "mean")

	assert proxy.closed is True


def test_loopback_error_response_raises_response_error() -> None:
	connection = _start_loopback_server(LoopbackServer)
	session = connection[0]
	try:
		with pytest.raises(ResponseError, match="file is not open"):
			session.var_open("/not-open.nc", "temperature")
	finally:
		_stop_loopback_server(*connection)


def test_unexpected_response_type_raises() -> None:
	connection = _start_loopback_server(WrongHeartbeatServer)
	session = connection[0]
	try:
		with pytest.raises(UnexpectedResponseError, match="expected response type"):
			session.heartbeat()
	finally:
		_stop_loopback_server(*connection)


def test_real_server_var_open_includes_rich_metadata() -> None:
	connection = _start_loopback_server(ServerStub)
	session = connection[0]
	data_path = Path(__file__).parent / "data" / "test1.nc"
	try:
		file_info = session.file_open(str(data_path))
		assert "tas" in file_info["keys"]

		chunked = session.var_open(str(data_path), "tas")
		assert chunked["layout"] == "chunked"
		assert chunked["chunks"] == [6, 32, 32]
		assert isinstance(chunked["attrs"], dict)
		assert "units" in chunked["attrs"]
		assert isinstance(chunked["index"], list)
		assert len(chunked["index"]) > 0
		assert {"chunk_offset", "byte_offset", "size", "filter_mask"}.issubset(chunked["index"][0])

		contiguous = session.var_open(str(data_path), "bounds")
		assert contiguous["layout"] == "contiguous"
		assert contiguous["chunks"] is None
		assert isinstance(contiguous["index"], list)
		assert len(contiguous["index"]) == 1
		assert contiguous["index"][0]["size"] > 0

		session.file_close(str(data_path))
	finally:
		_stop_loopback_server(*connection)


def test_real_server_get_chunk_for_chunked_and_contiguous() -> None:
	connection = _start_loopback_server(ServerStub)
	session = connection[0]
	data_path = Path(__file__).parent / "data" / "test1.nc"
	try:
		session.file_open(str(data_path))

		chunked = session.var_open(str(data_path), "tas")
		entry = chunked["index"][0]
		chunk_response = session.get_chunk(
			str(data_path),
			"tas",
			entry["byte_offset"],
			entry["size"],
		)
		pyfive_file = pyfive.File(str(data_path))
		pyfive_dataset = pyfive_file["tas"]
		storeinfo = pyfive_dataset.id.index[tuple(entry["chunk_offset"])]
		expected_chunk_bytes = pyfive_dataset.id._get_raw_chunk(storeinfo)

		assert chunk_response["type"] == "CHUNK_DATA"
		assert isinstance(chunk_response["data"], bytes)
		assert len(chunk_response["data"]) == entry["size"]
		assert chunk_response["data"] == expected_chunk_bytes

		with open(data_path, "rb") as handle:
			handle.seek(entry["byte_offset"])
			on_disk_chunk_bytes = handle.read(entry["size"])
		assert chunk_response["data"] == on_disk_chunk_bytes

		contiguous = session.var_open(str(data_path), "bounds")
		contiguous_entry = contiguous["index"][0]
		contig_response = session.get_chunk(
			str(data_path),
			"bounds",
			contiguous_entry["byte_offset"],
			contiguous_entry["size"],
		)
		assert contig_response["type"] == "CHUNK_DATA"
		assert isinstance(contig_response["data"], bytes)
		assert len(contig_response["data"]) == contiguous_entry["size"]

		with open(data_path, "rb") as handle:
			handle.seek(contiguous_entry["byte_offset"])
			expected_contiguous = handle.read(contiguous_entry["size"])
		assert contig_response["data"] == expected_contiguous

		session.file_close(str(data_path))
	finally:
		_stop_loopback_server(*connection)


def test_real_server_reduce_is_not_implemented() -> None:
	connection = _start_loopback_server(ServerStub)
	session = connection[0]
	data_path = Path(__file__).parent / "data" / "test1.nc"
	try:
		session.file_open(str(data_path))
		with pytest.raises(ResponseError, match="not implemented"):
			session.reduce(str(data_path), "tas", 0, 0, "mean")
	finally:
		_stop_loopback_server(*connection)