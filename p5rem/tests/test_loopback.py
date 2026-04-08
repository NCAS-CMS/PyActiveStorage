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
from p5rem.remote_server import ServerStub
from tests.roundtrip_assertions import assert_roundtrip_file_matches


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
	connection = _start_loopback_server(ServerStub)
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

	assert len(entries) == 1
	entry = entries[0]
	assert entry["name"] == str(data_file)
	assert entry["type"] == "file"
	assert entry["size"] == len("placeholder")
	assert isinstance(entry["mtime"], float)
	assert entry["is_link"] is False
	assert stat_result["type"] == "STAT_RESULT"
	assert stat_result["path"] == str(data_file)
	assert stat_result["is_file"] is True
	assert stat_result["size"] == len("placeholder")


def test_loopback_proxy_round_trip() -> None:
	connection = _start_loopback_server(ServerStub)
	session = connection[0]
	data_path = str(Path(__file__).parent / "data" / "test1.nc")
	try:
		with session.open(data_path) as proxy:
			assert "tas" in proxy.keys()
			assert isinstance(proxy.attrs, dict)
			assert isinstance(proxy.mtime, float)

			# Chunked, shuffle+deflate compressed variable
			ref_file = pyfive.File(data_path)
			tas = proxy["tas"]
			assert tas.shape == (12, 64, 128)
			assert tas.dtype == ref_file["tas"].dtype
			assert tas.chunks == (6, 32, 32)
			assert "units" in tas.attrs

			# Read first time step; decompression happens client-side
			data = tas[0, :, :]
			assert data.shape == (64, 128)
			assert np.allclose(data, ref_file["tas"][0, :, :])

			# Contiguous variable with UNDEFINED_ADDRESS → proxy returns fillvalue
			bounds_data = proxy["bounds"][()]
			ref_bounds = ref_file["bounds"][()]
			assert bounds_data.shape == ref_bounds.shape
			assert np.allclose(bounds_data, ref_bounds)

			with pytest.raises(ResponseError, match="not implemented"):
				session.reduce(data_path, "tas", 0, 100, "mean")

		assert proxy.closed is True
	finally:
		_stop_loopback_server(*connection)


def test_contiguous_file_data_and_coordinates_round_trip() -> None:
	connection = _start_loopback_server(ServerStub)
	session = connection[0]
	data_path = str(Path(__file__).parent / "data" / "contiguous_eg.nc")
	try:
		with session.open(data_path) as proxy:
			ref_file = pyfive.File(data_path)

			assert set(proxy.keys()) == set(ref_file.keys())

			# Validate all contiguous variables round-trip correctly.
			for name in proxy.keys():
				remote = proxy[name]
				local = ref_file[name]

				assert remote.chunks is None
				assert remote.shape == local.shape
				assert remote.dtype == local.dtype
				assert np.allclose(remote[()], local[()])

			# Validate coordinate linkage attributes on the data variable.
			q_attrs = proxy["q"].attrs
			assert q_attrs.get("coordinates") == ref_file["q"].attrs.get("coordinates")
			assert proxy["lat"].attrs.get("bounds") == ref_file["lat"].attrs.get("bounds")
			assert proxy["lon"].attrs.get("bounds") == ref_file["lon"].attrs.get("bounds")

			# Exercise data access through coordinates and bounds variables.
			assert np.allclose(proxy["lat"][()], ref_file["lat"][()])
			assert np.allclose(proxy["lon"][()], ref_file["lon"][()])
			assert np.allclose(proxy["lat_bnds"][()], ref_file["lat_bnds"][()])
			assert np.allclose(proxy["lon_bnds"][()], ref_file["lon_bnds"][()])
			assert np.allclose(proxy["time"][()], ref_file["time"][()])

		assert proxy.closed is True
	finally:
		_stop_loopback_server(*connection)


def test_enum_file_round_trip() -> None:
	connection = _start_loopback_server(ServerStub)
	session = connection[0]
	data_path = str(Path(__file__).parent / "data" / "enum_variable.nc")
	try:
		with session.open(data_path) as proxy:
			ref_file = pyfive.File(data_path)

			assert "enum_var" in proxy.keys()
			assert "axis" in proxy.keys()

			enum_remote = proxy["enum_var"]
			enum_ref = ref_file["enum_var"]
			assert enum_remote.chunks is None
			assert enum_remote.shape == enum_ref.shape
			assert enum_remote.dtype == enum_ref.dtype
			assert np.array_equal(enum_remote[()], enum_ref[()])
			assert np.array_equal(enum_remote[1:4], enum_ref[1:4])

			axis_remote = proxy["axis"]
			axis_ref = ref_file["axis"]
			assert axis_remote.shape == axis_ref.shape
			assert axis_remote.dtype == axis_ref.dtype
			assert np.allclose(axis_remote[()], axis_ref[()])

		assert proxy.closed is True
	finally:
		_stop_loopback_server(*connection)


@pytest.mark.parametrize("filename", ["test1.nc", "contiguous_eg.nc", "enum_variable.nc"])
def test_loopback_shared_roundtrip_assertions(filename: str) -> None:
	connection = _start_loopback_server(ServerStub)
	session = connection[0]
	data_path = Path(__file__).parent / "data" / filename
	try:
		assert_roundtrip_file_matches(session, data_path, str(data_path))
	finally:
		_stop_loopback_server(*connection)


def test_loopback_error_response_raises_response_error() -> None:
	connection = _start_loopback_server(ServerStub)
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
		# bounds has UNDEFINED_ADDRESS (no stored data) so index is empty
		assert contiguous["index"] == []

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

		# bounds has UNDEFINED_ADDRESS (no stored data); its index should be empty
		contiguous = session.var_open(str(data_path), "bounds")
		assert contiguous["index"] == []

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