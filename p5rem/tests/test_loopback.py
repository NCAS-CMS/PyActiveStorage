"""End-to-end loopback tests for session and server transport."""

from __future__ import annotations

import socket
import threading
from contextlib import suppress
from typing import Any

import pytest

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
			"index": {"kind": "mock"},
			"fragmented": False,
			"attrs": {"units": "K"},
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
		self._get_dataset(path, varname)
		return {
			"type": "REDUCTION_RESULT",
			"path": path,
			"varname": varname,
			"operation": operation,
			"value": 7.5,
		}


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
		assert dataset.dtype == "float32"
		assert dataset.chunks == (1, 3)
		assert dataset.attrs == {"units": "K"}

		chunk = dataset.get_chunk(10, 4)
		reduction = dataset.reduce(10, 4, "mean")

		assert chunk["data"] == b"xxxx"
		assert reduction["value"] == 7.5

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