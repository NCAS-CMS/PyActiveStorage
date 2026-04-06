"""Tests for cfdm interoperability with local and p5rem-remote files."""

from __future__ import annotations
from p5rem.remote_server import ServerStub
from p5rem.session import p5remSession
import socket
import threading
from contextlib import suppress
from pathlib import Path
from typing import Any

import pytest

cfdm = pytest.importorskip("cfdm")
fsspec = pytest.importorskip("fsspec")
pyfive = pytest.importorskip("pyfive")


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


def test_cfdm_read_local_file_handle_repeatedly() -> None:
    """cfdm.read should work with a local file-handle repeatedly."""

    data_path = Path(__file__).parent / "data" / "test1.nc"
    local_fs = fsspec.filesystem("local")
    with local_fs.open(str(data_path), "rb") as handle:
        handle.seek(0)
        assert len(cfdm.read(handle)) == 1

        handle.seek(0)
        assert len(cfdm.read(handle)) == 1

        handle.seek(0)
        assert len(cfdm.read([handle])) == 1


def test_cfdm_read_open_pyfive_file() -> None:
    """cfdm.read should work when given an actual open pyfive.File object."""

    data_path = Path(__file__).parent / "data" / "test1.nc"
    with pyfive.File(str(data_path)) as file_obj:
        print(f"\n[DEBUG] pyfive.File test")
        print(f"  type: {type(file_obj)}")
        print(f"  keys: {file_obj.keys()}")
        fields = cfdm.read(file_obj)
        print(f"  cfdm.read returned {len(fields)} fields")
    assert len(fields) == 1


def test_cfdm_read_remote_rfile_over_loopback() -> None:
    """cfdm.read should work with a p5rem remote rFile proxy."""

    connection = _start_loopback_server(ServerStub)
    session = connection[0]
    data_path = str(Path(__file__).parent / "data" / "test1.nc")
    try:
        with session.open(data_path) as remote_file:
            fields = cfdm.read(remote_file)
            assert len(fields) == 1
    finally:
        _stop_loopback_server(*connection)