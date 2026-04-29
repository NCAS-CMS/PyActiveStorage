"""End-to-end loopback tests for session and server transport."""

from __future__ import annotations

import io
import socket
import threading
from contextlib import suppress
from pathlib import Path
from typing import Any

import pyfive
import pytest
import numpy as np

from activestorage.p5rem.session import ResponseError, UnexpectedResponseError, p5remSession
from activestorage.p5rem.remote_server import ServerStub
from tests.roundtrip_assertions import assert_roundtrip_file_matches


class WrongHeartbeatServer(ServerStub):
	def handle_heartbeat(self) -> dict[str, Any]:
		return {"type": "LIST_RESULT", "path": "/", "entries": []}


class DropFileStateBeforeGetChunkServer(ServerStub):
	def __init__(self, *args: Any, **kwargs: Any) -> None:
		super().__init__(*args, **kwargs)
		self._dropped = False

	def _drop_state_once(self) -> None:
		if self._dropped:
			return
		self._dropped = True
		self._open_files.clear()
		self._datasets.clear()
		self._dim_id_to_name.clear()
		self._dim_id_reference_list.clear()

	def handle_get_chunk(self, path: str, varname: str, byte_offset: int, size: int, **fields: Any) -> dict[str, Any]:
		self._drop_state_once()
		return super().handle_get_chunk(path, varname, byte_offset, size, **fields)


class DropFileStateBeforeGetChunksServer(ServerStub):
	def __init__(self, *args: Any, **kwargs: Any) -> None:
		super().__init__(*args, **kwargs)
		self._dropped = False

	def _drop_state_once(self) -> None:
		if self._dropped:
			return
		self._dropped = True
		self._open_files.clear()
		self._datasets.clear()
		self._dim_id_to_name.clear()
		self._dim_id_reference_list.clear()

	def handle_get_chunks(self, path: str, varname: str, chunks: list[dict[str, Any]], thread_count: int = 4) -> None:
		self._drop_state_once()
		return super().handle_get_chunks(path, varname, chunks, thread_count)


class DropFileStateBeforeReduceServer(ServerStub):
	def __init__(self, *args: Any, **kwargs: Any) -> None:
		super().__init__(*args, **kwargs)
		self._dropped = False

	def _drop_state_once(self) -> None:
		if self._dropped:
			return
		self._dropped = True
		self._open_files.clear()
		self._datasets.clear()
		self._dim_id_to_name.clear()
		self._dim_id_reference_list.clear()

	def handle_reduce(self, path: str, varname: str, operation: str, **fields: Any) -> dict[str, Any]:
		self._drop_state_once()
		return super().handle_reduce(path, varname, operation, **fields)


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


def test_server_detect_file_format_from_extension() -> None:
	server = ServerStub(io.BytesIO(), io.BytesIO())
	assert server._detect_file_format("demo.nc") == "hdf5"
	assert server._detect_file_format("demo.h5") == "hdf5"
	assert server._detect_file_format("demo.pp") == "pp"
	assert server._detect_file_format("demo.pp.gz") == "pp"


def test_server_detect_file_format_from_magic(tmp_path: Path) -> None:
	server = ServerStub(io.BytesIO(), io.BytesIO())

	hdf_path = tmp_path / "mystery_hdf"
	hdf_path.write_bytes(b"\x89HDF\r\n\x1a\nEXTRA")
	assert server._detect_file_format(str(hdf_path)) == "hdf5"

	pp_path = tmp_path / "mystery_pp"
	pp_path.write_bytes(b"\x00\x01\x02\x03EXTRA")
	assert server._detect_file_format(str(pp_path)) == "pp"

	unknown_path = tmp_path / "mystery_unknown"
	unknown_path.write_bytes(b"ABCDEFGH")
	with pytest.raises(ValueError):
		server._detect_file_format(str(unknown_path))


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

			reduce_response = session.reduce_selection(
				data_path,
				"tas",
				"mean",
				selection=[{"type": "index", "value": 0}, None, None],
			)
			expected_mean = float(np.mean(ref_file["tas"][0, :, :]))
			assert reduce_response["type"] == "REDUCTION_RESULT"
			assert reduce_response["mode"] == "selection"
			assert np.isclose(reduce_response["value"], expected_mean)

		assert proxy.closed is True
	finally:
		_stop_loopback_server(*connection)


def test_loopback_get_chunk_recovers_when_server_loses_file_state() -> None:
	connection = _start_loopback_server(DropFileStateBeforeGetChunkServer)
	session = connection[0]
	data_path = str(Path(__file__).parent / "data" / "contiguous_eg.nc")
	try:
		with session.open(data_path) as proxy:
			q = proxy["q"]
			data = q[0, :]
			assert data.size > 0
	finally:
		_stop_loopback_server(*connection)


def test_loopback_get_chunks_recovers_when_server_loses_file_state() -> None:
	connection = _start_loopback_server(DropFileStateBeforeGetChunksServer)
	session = connection[0]
	data_path = str(Path(__file__).parent / "data" / "test1.nc")
	try:
		with session.open(data_path) as proxy:
			tas = proxy["tas"]
			data = tas[0, :, :]
			assert data.shape == (64, 128)
	finally:
		_stop_loopback_server(*connection)


def test_loopback_reduce_recovers_when_server_loses_file_state() -> None:
	connection = _start_loopback_server(DropFileStateBeforeReduceServer)
	session = connection[0]
	data_path = str(Path(__file__).parent / "data" / "test1.nc")
	try:
		session.file_open(data_path)
		response = session.reduce_selection(
			data_path,
			"tas",
			"mean",
			selection=[{"type": "index", "value": 0}, None, None],
		)
		assert response["type"] == "REDUCTION_RESULT"
		assert response["mode"] == "selection"
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


def test_handle_file_open_resets_cached_datasets_for_reopen(monkeypatch: pytest.MonkeyPatch) -> None:
	class FakeDataset:
		def __init__(self, label: str, dim_id: int) -> None:
			self.label = label
			self.attrs = {"_Netcdf4Dimid": dim_id}

	class FakeFile:
		def __init__(self, label: str, dataset: FakeDataset) -> None:
			self.label = label
			self.attrs: dict[str, Any] = {}
			self.consolidated_metadata = True
			self._dataset = dataset
			self.closed = False

		def keys(self):
			return [self._dataset.label]

		def __getitem__(self, key: str) -> FakeDataset:
			assert key == self._dataset.label
			return self._dataset

		def close(self) -> None:
			self.closed = True

	data_path = Path(__file__).parent / "data" / "test1.nc"
	first_file = FakeFile("old_var", FakeDataset("old_var", 1))
	second_file = FakeFile("new_var", FakeDataset("new_var", 2))
	opened_files = iter([first_file, second_file])

	monkeypatch.setattr(pyfive, "File", lambda path: next(opened_files))

	server = ServerStub(io.BytesIO(), io.BytesIO())
	path = str(data_path)

	server.handle_file_open(path)
	first_dataset = server._get_dataset(path, "old_var")
	assert first_dataset is first_file._dataset
	assert server._dim_id_to_name[path] == {1: "old_var"}

	close_response = server.handle_file_close(path)
	assert close_response == {"type": "FILE_CLOSE", "path": path, "closed": True}
	assert first_file.closed is True
	assert path not in server._datasets
	assert path not in server._dim_id_to_name
	assert path not in server._dim_id_reference_list

	server.handle_file_open(path)
	second_dataset = server._get_dataset(path, "new_var")
	assert second_dataset is second_file._dataset
	assert second_dataset is not first_dataset
	assert server._dim_id_to_name[path] == {2: "new_var"}


def test_get_dataset_rejects_stale_cache_when_file_not_open() -> None:
	server = ServerStub(io.BytesIO(), io.BytesIO())
	path = "/tmp/not-open.nc"
	stale_dataset = object()
	server._datasets[path] = {"tas": stale_dataset}

	with pytest.raises(FileNotFoundError, match="file is not open"):
		server._get_dataset(path, "tas")

	assert path not in server._datasets


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


def test_real_server_reduce_supports_selection_and_chunk_modes() -> None:
	connection = _start_loopback_server(ServerStub)
	session = connection[0]
	data_path = Path(__file__).parent / "data" / "contiguous_eg.nc"
	try:
		session.file_open(str(data_path))

		selection_response = session.reduce_selection(
			str(data_path),
			"q",
			"mean",
			selection=[{"type": "index", "value": 0}, None],
		)
		assert selection_response["type"] == "REDUCTION_RESULT"
		assert selection_response["mode"] == "selection"
		assert selection_response["thread_count"] == 1

		q_meta = session.var_open(str(data_path), "q")
		chunk_entry = q_meta["index"][0]
		chunk_response = session.reduce_chunk(
			str(data_path),
			"q",
			chunk_entry["byte_offset"],
			chunk_entry["size"],
			"count",
		)
		assert chunk_response["type"] == "REDUCTION_RESULT"
		assert chunk_response["mode"] == "chunk"

		ref_file = pyfive.File(str(data_path))
		expected_count = int(ref_file["q"][()].size)
		assert int(chunk_response["value"]) == expected_count

		with pytest.raises(ResponseError, match="unsupported reduction operation"):
			session.reduce_selection(str(data_path), "q", "does_not_exist")
	finally:
		_stop_loopback_server(*connection)


def test_real_server_reduce_selection_parallel_full_dataset() -> None:
	connection = _start_loopback_server(ServerStub)
	session = connection[0]
	data_path = Path(__file__).parent / "data" / "test1.nc"
	try:
		session.file_open(str(data_path))
		response = session.reduce_selection(
			str(data_path),
			"tas",
			"mean",
			selection=None,
			thread_count=3,
		)

		ref_file = pyfive.File(str(data_path))
		expected_mean = float(np.mean(ref_file["tas"][()]))

		assert response["type"] == "REDUCTION_RESULT"
		assert response["mode"] == "selection"
		assert response["thread_count"] == 3
		assert np.isclose(response["value"], expected_mean)
	finally:
		_stop_loopback_server(*connection)


def test_real_server_reduce_selection_parallel_partial_selection() -> None:
	connection = _start_loopback_server(ServerStub)
	session = connection[0]
	data_path = Path(__file__).parent / "data" / "test1.nc"
	try:
		session.file_open(str(data_path))
		response = session.reduce_selection(
			str(data_path),
			"tas",
			"sum",
			selection=[
				{"type": "slice", "start": 0, "stop": 7, "step": 1},
				{"type": "slice", "start": 5, "stop": 55, "step": 1},
				{"type": "slice", "start": 10, "stop": 110, "step": 1},
			],
			thread_count=4,
		)

		ref_file = pyfive.File(str(data_path))
		expected_sum = float(np.sum(ref_file["tas"][0:7, 5:55, 10:110]))

		assert response["type"] == "REDUCTION_RESULT"
		assert response["mode"] == "selection"
		assert response["thread_count"] == 4
		assert np.isclose(response["value"], expected_sum)
	finally:
		_stop_loopback_server(*connection)


def test_real_server_reduce_selection_chunk_planned_with_single_thread() -> None:
	connection = _start_loopback_server(ServerStub)
	session = connection[0]
	data_path = Path(__file__).parent / "data" / "test1.nc"
	try:
		session.file_open(str(data_path))
		response = session.reduce_selection(
			str(data_path),
			"tas",
			"sum",
			selection=[
				{"type": "slice", "start": 1, "stop": 8, "step": 1},
				{"type": "slice", "start": 4, "stop": 48, "step": 1},
				{"type": "slice", "start": 7, "stop": 100, "step": 1},
			],
			thread_count=1,
		)

		ref_file = pyfive.File(str(data_path))
		expected_sum = float(np.sum(ref_file["tas"][1:8, 4:48, 7:100]))

		assert response["type"] == "REDUCTION_RESULT"
		assert response["mode"] == "selection"
		assert response["thread_count"] == 1
		assert np.isclose(response["value"], expected_sum)
	finally:
		_stop_loopback_server(*connection)


# ---------------------------------------------------------------------------
# Operation coverage: min / max / range / argmin / argmax on chunked datasets
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("operation,ref_fn", [
	("min",    lambda a: float(np.min(a))),
	("max",    lambda a: float(np.max(a))),
	("mean",   lambda a: float(np.mean(a))),
	("count",  lambda a: int(a.size)),
	("range",  lambda a: [float(np.min(a)), float(np.max(a))]),
	("argmin", lambda a: int(np.argmin(a))),
	("argmax", lambda a: int(np.argmax(a))),
])
def test_reduce_selection_chunked_all_operations_full_dataset(operation, ref_fn) -> None:
	connection = _start_loopback_server(ServerStub)
	session = connection[0]
	data_path = Path(__file__).parent / "data" / "test1.nc"
	try:
		session.file_open(str(data_path))
		response = session.reduce_selection(str(data_path), "tas", operation, selection=None, thread_count=4)
		ref = ref_fn(pyfive.File(str(data_path))["tas"][()])
		assert response["type"] == "REDUCTION_RESULT"
		assert response["mode"] == "selection"
		if isinstance(ref, list):
			assert len(response["value"]) == 2
			assert np.isclose(response["value"][0], ref[0])
			assert np.isclose(response["value"][1], ref[1])
		elif isinstance(ref, int):
			assert int(response["value"]) == ref
		else:
			assert np.isclose(response["value"], ref)
	finally:
		_stop_loopback_server(*connection)


@pytest.mark.parametrize("operation,ref_fn", [
	("min",    lambda a: float(np.min(a))),
	("max",    lambda a: float(np.max(a))),
	("mean",   lambda a: float(np.mean(a))),
	("sum",    lambda a: float(np.sum(a))),
	("count",  lambda a: int(a.size)),
	("argmin", lambda a: int(np.argmin(a))),
	("argmax", lambda a: int(np.argmax(a))),
])
def test_reduce_selection_chunked_all_operations_partial_selection(operation, ref_fn) -> None:
	connection = _start_loopback_server(ServerStub)
	session = connection[0]
	data_path = Path(__file__).parent / "data" / "test1.nc"
	sel = [
		{"type": "slice", "start": 2, "stop": 10, "step": 1},
		{"type": "slice", "start": 10, "stop": 50, "step": 1},
		{"type": "slice", "start": 20, "stop": 90, "step": 1},
	]
	try:
		session.file_open(str(data_path))
		response = session.reduce_selection(str(data_path), "tas", operation, selection=sel, thread_count=4)
		ref_array = pyfive.File(str(data_path))["tas"][2:10, 10:50, 20:90]
		ref = ref_fn(ref_array)
		assert response["type"] == "REDUCTION_RESULT"
		assert response["mode"] == "selection"
		if isinstance(ref, int):
			assert int(response["value"]) == ref
		else:
			assert np.isclose(response["value"], ref)
	finally:
		_stop_loopback_server(*connection)


# ---------------------------------------------------------------------------
# Operation coverage: contiguous dataset streaming path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("operation,ref_fn", [
	("min",    lambda a: float(np.min(a))),
	("max",    lambda a: float(np.max(a))),
	("mean",   lambda a: float(np.mean(a))),
	("sum",    lambda a: float(np.sum(a))),
	("count",  lambda a: int(a.size)),
	("argmin", lambda a: int(np.argmin(a))),
	("argmax", lambda a: int(np.argmax(a))),
])
def test_reduce_selection_contiguous_all_operations_full_dataset(operation, ref_fn) -> None:
	connection = _start_loopback_server(ServerStub)
	session = connection[0]
	data_path = Path(__file__).parent / "data" / "contiguous_eg.nc"
	try:
		session.file_open(str(data_path))
		response = session.reduce_selection(str(data_path), "q", operation, selection=None)
		ref = ref_fn(pyfive.File(str(data_path))["q"][()])
		assert response["type"] == "REDUCTION_RESULT"
		assert response["mode"] == "selection"
		if isinstance(ref, int):
			assert int(response["value"]) == ref
		else:
			assert np.isclose(response["value"], ref)
	finally:
		_stop_loopback_server(*connection)


# ---------------------------------------------------------------------------
# reduce_chunk: chunked variable (raw compressed chunk, count of raw elements)
# and contiguous variable (raw bytes, count of stored elements)
# ---------------------------------------------------------------------------

def test_reduce_chunk_on_chunked_variable() -> None:
	connection = _start_loopback_server(ServerStub)
	session = connection[0]
	data_path = Path(__file__).parent / "data" / "test1.nc"
	try:
		session.file_open(str(data_path))
		meta = session.var_open(str(data_path), "tas")
		# Use the first chunk index entry.
		entry = meta["index"][0]
		for operation in ("min", "max", "sum", "count", "mean"):
			response = session.reduce_chunk(
				str(data_path), "tas",
				entry["byte_offset"], entry["size"],
				operation,
			)
			assert response["type"] == "REDUCTION_RESULT"
			assert response["mode"] == "chunk"
			assert response["byte_offset"] == entry["byte_offset"]
			assert response["size"] == entry["size"]
			assert response["value"] is not None
	finally:
		_stop_loopback_server(*connection)


def test_reduce_chunk_on_contiguous_variable() -> None:
	connection = _start_loopback_server(ServerStub)
	session = connection[0]
	data_path = Path(__file__).parent / "data" / "contiguous_eg.nc"
	try:
		session.file_open(str(data_path))
		meta = session.var_open(str(data_path), "q")
		entry = meta["index"][0]
		ref_array = pyfive.File(str(data_path))["q"][()]

		for operation, ref_fn in [
			("min",   lambda a: float(np.min(a))),
			("max",   lambda a: float(np.max(a))),
			("sum",   lambda a: float(np.sum(a))),
			("mean",  lambda a: float(np.mean(a))),
			("count", lambda a: int(a.size)),
		]:
			response = session.reduce_chunk(
				str(data_path), "q",
				entry["byte_offset"], entry["size"],
				operation,
			)
			assert response["type"] == "REDUCTION_RESULT"
			assert response["mode"] == "chunk"
			ref = ref_fn(ref_array)
			if isinstance(ref, int):
				assert int(response["value"]) == ref
			else:
				assert np.isclose(response["value"], ref)
	finally:
		_stop_loopback_server(*connection)


# ---------------------------------------------------------------------------
# Error path: unknown operation on chunked variable
# ---------------------------------------------------------------------------

def test_reduce_selection_unknown_operation_chunked_raises() -> None:
	connection = _start_loopback_server(ServerStub)
	session = connection[0]
	data_path = Path(__file__).parent / "data" / "test1.nc"
	try:
		session.file_open(str(data_path))
		with pytest.raises(ResponseError, match="unsupported reduction operation"):
			session.reduce_selection(str(data_path), "tas", "variance")
	finally:
		_stop_loopback_server(*connection)


# ---------------------------------------------------------------------------
# Edge: thread_count larger than number of intersecting chunks clamps cleanly
# ---------------------------------------------------------------------------

def test_reduce_selection_thread_count_exceeds_chunk_count() -> None:
	connection = _start_loopback_server(ServerStub)
	session = connection[0]
	data_path = Path(__file__).parent / "data" / "test1.nc"
	try:
		session.file_open(str(data_path))
		# A 1-timestep selection hits at most a handful of chunks; use 1000 workers.
		response = session.reduce_selection(
			str(data_path), "tas", "sum",
			selection=[
				{"type": "index", "value": 0},
				{"type": "slice", "start": 0, "stop": 64, "step": 1},
				{"type": "slice", "start": 0, "stop": 128, "step": 1},
			],
			thread_count=1000,
		)
		ref = float(np.sum(pyfive.File(str(data_path))["tas"][0, :, :]))
		assert response["type"] == "REDUCTION_RESULT"
		assert np.isclose(response["value"], ref)
	finally:
		_stop_loopback_server(*connection)