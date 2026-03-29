"""Tests for bootstrap upload-and-exec helpers."""

from __future__ import annotations

from pathlib import Path
import shlex
import subprocess
from typing import Any

import pytest

import p5rem.bootstrap as bootstrap_module
from p5rem.bootstrap import bootstrap_reconnecting_session, bootstrap_server
from p5rem.session import SessionError


class _FakeSFTP:
	def __init__(self, root: Path) -> None:
		self.root = root

	def _local_path(self, remote_path: str) -> Path:
		cleaned = remote_path.strip("/")
		return self.root / cleaned

	def stat(self, path: str) -> Any:
		return self._local_path(path).stat()

	def mkdir(self, path: str) -> None:
		self._local_path(path).mkdir(parents=False, exist_ok=False)

	def put(self, local_path: str, remote_path: str) -> None:
		target = self._local_path(remote_path)
		target.parent.mkdir(parents=True, exist_ok=True)
		target.write_bytes(Path(local_path).read_bytes())

	def chmod(self, remote_path: str, mode: int) -> None:
		self._local_path(remote_path).chmod(mode)

	def close(self) -> None:
		return None


class _FakeChannel:
	def __init__(self, remote_root: Path) -> None:
		self._remote_root = remote_root
		self._proc: subprocess.Popen[bytes] | None = None
		self._timeout: float | None = None

	def exec_command(self, command: str) -> None:
		self._proc = subprocess.Popen(
			command,
			shell=True,
			cwd=str(self._remote_root),
			stdin=subprocess.PIPE,
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
		)

	def makefile(self, mode: str):
		assert self._proc is not None
		if "w" in mode:
			return self._proc.stdin
		return self._proc.stdout

	def makefile_stderr(self, mode: str):
		assert self._proc is not None
		return self._proc.stderr

	def exit_status_ready(self) -> bool:
		assert self._proc is not None
		return self._proc.poll() is not None

	def recv_exit_status(self) -> int:
		assert self._proc is not None
		return self._proc.wait(timeout=self._timeout)

	def settimeout(self, timeout: float) -> None:
		self._timeout = timeout

	def close(self) -> None:
		if self._proc is not None:
			if self._proc.poll() is None:
				self._proc.terminate()


class _FakeTransport:
	def __init__(self, remote_root: Path) -> None:
		self._remote_root = remote_root

	def open_session(self) -> _FakeChannel:
		return _FakeChannel(self._remote_root)


class _FakeSSHClient:
	def __init__(self, remote_root: Path) -> None:
		self.remote_root = remote_root
		self._transport = _FakeTransport(remote_root)
		self.connect_args: dict[str, Any] | None = None

	def set_missing_host_key_policy(self, policy: Any) -> None:
		_ = policy

	def connect(self, **kwargs: Any) -> None:
		self.connect_args = dict(kwargs)

	def open_sftp(self) -> _FakeSFTP:
		return _FakeSFTP(self.remote_root)

	def get_transport(self) -> _FakeTransport:
		return self._transport

	def close(self) -> None:
		return None


def test_bootstrap_uploads_script_and_executes_it(tmp_path: Path) -> None:
	remote_root = tmp_path / "remote"
	remote_root.mkdir()
	local_script = tmp_path / "server.py"
	local_script.write_text(
		"print('BOOT_OK', flush=True)\n",
		encoding="utf-8",
	)

	client = _FakeSSHClient(remote_root)
	proc = bootstrap_server(
		host="fake-host",
		username="fake-user",
		password="fake-pass",
		local_script_path=str(local_script),
		remote_dir=".p5rem",
		remote_filename="boot.py",
		remote_python="python3",
		ssh_client_factory=lambda: client,
	)
	try:
		uploaded = remote_root / ".p5rem" / "boot.py"
		assert uploaded.exists()
		assert uploaded.read_text(encoding="utf-8") == local_script.read_text(encoding="utf-8")
		assert proc.remote_path == ".p5rem/boot.py"
		assert proc.command == "python3 -u .p5rem/boot.py"
		assert client.connect_args is not None
		assert client.connect_args["hostname"] == "fake-host"
		assert client.connect_args["username"] == "fake-user"

		exit_code = proc.wait(timeout=5)
		assert exit_code == 0
		assert proc.stdout.read().strip() == b"BOOT_OK"
	finally:
		proc.close()


def test_bootstrap_command_quotes_arguments(tmp_path: Path) -> None:
	remote_root = tmp_path / "remote"
	remote_root.mkdir()
	local_script = tmp_path / "server.py"
	local_script.write_text("print('ok')\n", encoding="utf-8")
	client = _FakeSSHClient(remote_root)

	proc = bootstrap_server(
		host="fake-host",
		username="fake-user",
		password="fake-pass",
		local_script_path=str(local_script),
		remote_dir=".p5rem",
		remote_filename="boot.py",
		remote_python="python3",
		args=("--name", "with spaces"),
		ssh_client_factory=lambda: client,
	)
	try:
		parts = shlex.split(proc.command)
		assert parts[-2:] == ["--name", "with spaces"]
	finally:
		proc.close()


def test_bootstrap_adds_no_capture_output_for_conda_run(tmp_path: Path) -> None:
	remote_root = tmp_path / "remote"
	remote_root.mkdir()
	local_script = tmp_path / "server.py"
	local_script.write_text("print('ok')\n", encoding="utf-8")
	client = _FakeSSHClient(remote_root)

	proc = bootstrap_server(
		host="fake-host",
		username="fake-user",
		password="fake-pass",
		local_script_path=str(local_script),
		remote_dir=".p5rem",
		remote_filename="boot.py",
		remote_python="conda run -n jas26 python",
		login_shell=True,
		ssh_client_factory=lambda: client,
	)
	try:
		inner_command = shlex.split(proc.command)[2]
		parts = shlex.split(inner_command)
		assert parts[:4] == ["conda", "run", "--no-capture-output", "-n"]
		assert parts[4:6] == ["jas26", "python"]
	finally:
		proc.close()


def test_bootstrap_resolves_shortname_from_ssh_config(tmp_path: Path) -> None:
	remote_root = tmp_path / "remote"
	remote_root.mkdir()
	local_script = tmp_path / "server.py"
	local_script.write_text("print('ok')\n", encoding="utf-8")
	ssh_config = tmp_path / "ssh_config"
	ssh_config.write_text(
		"Host shortname\n"
		"  HostName resolved.example.org\n"
		"  User config-user\n"
		"  Port 2202\n"
		"  IdentityFile /tmp/config-key\n",
		encoding="utf-8",
	)

	client = _FakeSSHClient(remote_root)
	proc = bootstrap_server(
		host="shortname",
		local_script_path=str(local_script),
		remote_dir=".p5rem",
		remote_filename="boot.py",
		remote_python="python3",
		ssh_config_path=str(ssh_config),
		ssh_client_factory=lambda: client,
	)
	try:
		assert client.connect_args is not None
		assert client.connect_args["hostname"] == "resolved.example.org"
		assert client.connect_args["username"] == "config-user"
		assert client.connect_args["port"] == 2202
		assert client.connect_args["key_filename"] == "/tmp/config-key"
	finally:
		proc.close()


def test_bootstrap_explicit_connection_values_override_ssh_config(tmp_path: Path) -> None:
	remote_root = tmp_path / "remote"
	remote_root.mkdir()
	local_script = tmp_path / "server.py"
	local_script.write_text("print('ok')\n", encoding="utf-8")
	ssh_config = tmp_path / "ssh_config"
	ssh_config.write_text(
		"Host shortname\n"
		"  HostName resolved.example.org\n"
		"  User config-user\n"
		"  Port 2202\n"
		"  IdentityFile /tmp/config-key\n",
		encoding="utf-8",
	)

	client = _FakeSSHClient(remote_root)
	proc = bootstrap_server(
		host="shortname",
		username="explicit-user",
		port=2022,
		key_filename="/tmp/explicit-key",
		local_script_path=str(local_script),
		remote_dir=".p5rem",
		remote_filename="boot.py",
		remote_python="python3",
		ssh_config_path=str(ssh_config),
		ssh_client_factory=lambda: client,
	)
	try:
		assert client.connect_args is not None
		assert client.connect_args["hostname"] == "resolved.example.org"
		assert client.connect_args["username"] == "explicit-user"
		assert client.connect_args["port"] == 2022
		assert client.connect_args["key_filename"] == "/tmp/explicit-key"
	finally:
		proc.close()


def test_reconnecting_bootstrap_session_retries_after_session_error(monkeypatch) -> None:
	class DummySession:
		def __init__(self, fails_once: bool) -> None:
			self._fails_once = fails_once
			self._failed = False
			self._callback = None
			self.closed = False

		def set_heartbeat_failure_callback(self, callback):
			self._callback = callback

		def start_heartbeat(self, **kwargs):
			_ = kwargs

		def list(self, path: str):
			if self._fails_once and not self._failed:
				self._failed = True
				raise SessionError("dropout")
			return [path, "ok"]

		def close(self) -> None:
			self.closed = True

	created: list[DummySession] = []

	def fake_bootstrap_session(**kwargs):
		_ = kwargs
		session = DummySession(fails_once=(len(created) == 0))
		created.append(session)
		return session

	monkeypatch.setattr(bootstrap_module, "bootstrap_session", fake_bootstrap_session)

	manager = bootstrap_reconnecting_session(
		host="fake-host",
		username="fake-user",
		local_script_path="/tmp/noop.py",
	)
	try:
		assert manager.list("/tmp") == ["/tmp", "ok"]
		assert len(created) == 2
		assert created[0].closed is True
	finally:
		manager.close()


def test_reconnecting_bootstrap_session_heartbeat_callback_triggers_reconnect(monkeypatch) -> None:
	class DummySession:
		def __init__(self) -> None:
			self.callback = None
			self.closed = False

		def set_heartbeat_failure_callback(self, callback):
			self.callback = callback

		def start_heartbeat(self, **kwargs):
			_ = kwargs

		def list(self, path: str):
			return [path]

		def close(self) -> None:
			self.closed = True

	created: list[DummySession] = []

	def fake_bootstrap_session(**kwargs):
		_ = kwargs
		session = DummySession()
		created.append(session)
		return session

	monkeypatch.setattr(bootstrap_module, "bootstrap_session", fake_bootstrap_session)

	manager = bootstrap_reconnecting_session(
		host="fake-host",
		username="fake-user",
		local_script_path="/tmp/noop.py",
	)
	try:
		first = created[0]
		assert first.callback is not None
		first.callback(first, RuntimeError("heartbeat failure"))
		assert len(created) == 2
		assert first.closed is True
	finally:
		manager.close()
