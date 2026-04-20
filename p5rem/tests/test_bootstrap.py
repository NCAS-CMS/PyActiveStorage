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


def test_bootstrap_adds_no_capture_output_for_conda_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	remote_root = tmp_path / "remote"
	remote_root.mkdir()
	local_script = tmp_path / "server.py"
	local_script.write_text("print('ok')\n", encoding="utf-8")
	client = _FakeSSHClient(remote_root)
	monkeypatch.setattr(bootstrap_module, "_probe_remote_python", lambda *args, **kwargs: None)

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


def test_bootstrap_reports_remote_python_preflight_failure(tmp_path: Path) -> None:
	remote_root = tmp_path / "remote"
	remote_root.mkdir()
	local_script = tmp_path / "server.py"
	local_script.write_text("print('ok')\n", encoding="utf-8")
	client = _FakeSSHClient(remote_root)

	with pytest.raises(bootstrap_module.BootstrapError, match="remote python preflight failed") as excinfo:
		bootstrap_server(
			host="fake-host",
			username="fake-user",
			password="fake-pass",
			local_script_path=str(local_script),
			remote_dir=".p5rem",
			remote_filename="boot.py",
			remote_python="conda run -n missingenv python",
			login_shell=True,
			ssh_client_factory=lambda: client,
		)

	message = str(excinfo.value)
	assert "remote conda environment appears unavailable" in message
	assert "conda run -n missingenv python" in message


def test_bootstrap_supports_shell_fragment_remote_python(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	remote_root = tmp_path / "remote"
	remote_root.mkdir()
	local_script = tmp_path / "server.py"
	local_script.write_text("print('ok')\n", encoding="utf-8")
	client = _FakeSSHClient(remote_root)
	monkeypatch.setattr(bootstrap_module, "_probe_remote_python", lambda *args, **kwargs: None)

	proc = bootstrap_server(
		host="fake-host",
		username="fake-user",
		password="fake-pass",
		local_script_path=str(local_script),
		remote_dir=".p5rem",
		remote_filename="boot.py",
		remote_python="module load py/3.12 && python",
		login_shell=True,
		ssh_client_factory=lambda: client,
	)
	try:
		inner_command = shlex.split(proc.command)[2]
		assert "module load py/3.12 && python -u" in inner_command
	finally:
		proc.close()


def test_bootstrap_supports_remote_setup_prelude(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	remote_root = tmp_path / "remote"
	remote_root.mkdir()
	local_script = tmp_path / "server.py"
	local_script.write_text("print('ok')\n", encoding="utf-8")
	client = _FakeSSHClient(remote_root)
	monkeypatch.setattr(bootstrap_module, "_probe_remote_python", lambda *args, **kwargs: None)

	proc = bootstrap_server(
		host="fake-host",
		username="fake-user",
		password="fake-pass",
		local_script_path=str(local_script),
		remote_dir=".p5rem",
		remote_filename="boot.py",
		remote_python="python3",
		remote_setup="module load py/3.12",
		login_shell=True,
		ssh_client_factory=lambda: client,
	)
	try:
		inner_command = shlex.split(proc.command)[2]
		assert inner_command.startswith("module load py/3.12 && python3 -u")
	finally:
		proc.close()


def test_build_python_probe_command_supports_remote_setup() -> None:
	command = bootstrap_module._build_python_probe_command(
		"python3",
		remote_setup="module load py/3.12",
		login_shell=True,
	)
	inner = shlex.split(command)[2]
	assert inner.startswith("module load py/3.12 && python3 -c")


def test_bootstrap_prompts_hidden_password_for_keyboard_interactive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	class _AuthFailThenSucceedSSHClient(_FakeSSHClient):
		def __init__(self, remote_root: Path, *, should_fail: bool) -> None:
			super().__init__(remote_root)
			self.should_fail = should_fail

		def connect(self, **kwargs: Any) -> None:
			self.connect_args = dict(kwargs)
			if self.should_fail:
				raise bootstrap_module.paramiko.AuthenticationException("challenge required")

	remote_root = tmp_path / "remote"
	remote_root.mkdir()
	local_script = tmp_path / "server.py"
	local_script.write_text("print('ok')\n", encoding="utf-8")

	first_client = _AuthFailThenSucceedSSHClient(remote_root, should_fail=True)
	second_client = _AuthFailThenSucceedSSHClient(remote_root, should_fail=False)
	clients = iter([first_client, second_client])

	class _DummyStdin:
		@staticmethod
		def isatty() -> bool:
			return True

	monkeypatch.setattr(bootstrap_module.sys, "stdin", _DummyStdin())
	monkeypatch.setattr(bootstrap_module.getpass, "getpass", lambda prompt: "secret-pass")

	proc = bootstrap_server(
		host="fake-host",
		username="fake-user",
		password=None,
		local_script_path=str(local_script),
		remote_dir=".p5rem",
		remote_filename="boot.py",
		remote_python="python3",
		ssh_client_factory=lambda: next(clients),
	)
	try:
		assert first_client.connect_args is not None
		assert first_client.connect_args["password"] == "secret-pass"
		assert second_client.connect_args is not None
		assert second_client.connect_args["password"] == "secret-pass"
		assert second_client.connect_args["allow_agent"] is False
		assert second_client.connect_args["look_for_keys"] is False
	finally:
		proc.close()


def test_bootstrap_masks_paramiko_input_prompt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	class _PromptSSHClient(_FakeSSHClient):
		def __init__(self, remote_root: Path) -> None:
			super().__init__(remote_root)
			self.password_value: str | None = None

		def connect(self, **kwargs: Any) -> None:
			self.connect_args = dict(kwargs)
			self.password_value = kwargs.get("password")

	remote_root = tmp_path / "remote"
	remote_root.mkdir()
	local_script = tmp_path / "server.py"
	local_script.write_text("print('ok')\n", encoding="utf-8")

	client = _PromptSSHClient(remote_root)

	class _DummyStdin:
		@staticmethod
		def isatty() -> bool:
			return True

	monkeypatch.setattr(bootstrap_module.sys, "stdin", _DummyStdin())
	monkeypatch.setattr(bootstrap_module.getpass, "getpass", lambda prompt: "hidden-secret")

	proc = bootstrap_server(
		host="fake-host",
		username="fake-user",
		password=None,
		local_script_path=str(local_script),
		remote_dir=".p5rem",
		remote_filename="boot.py",
		remote_python="python3",
		ssh_client_factory=lambda: client,
	)
	try:
		assert client.password_value == "hidden-secret"
		assert client.connect_args is not None
		assert client.connect_args["allow_agent"] is False
		assert client.connect_args["look_for_keys"] is False
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


def test_bootstrap_session_can_disable_default_cache(monkeypatch) -> None:
	proc = object()
	recorded: dict[str, Any] = {}

	def fake_bootstrap_server(**kwargs):
		_ = kwargs
		return type("Proc", (), {"stdin": object(), "stdout": object()})()

	def fake_session(**kwargs):
		recorded.update(kwargs)
		return object()

	monkeypatch.setattr(bootstrap_module, "bootstrap_server", fake_bootstrap_server)
	monkeypatch.setattr(bootstrap_module, "p5remSession", fake_session)

	result = bootstrap_module.bootstrap_session(
		host="fake-host",
		remote_python="python3",
		use_cache=False,
	)

	assert result is not None
	assert recorded["host"] is None
	assert recorded["cache"] is None


def test_bootstrap_session_reports_missing_remote_runtime(monkeypatch) -> None:
	class FakeProc:
		def poll(self) -> int:
			return 127

		def read_stderr_snippet(self) -> str:
			return "bash: mamba: command not found"

		stdin = object()
		stdout = object()

	class FailingSession:
		def __init__(self, **kwargs):
			_ = kwargs

		def heartbeat(self):
			raise RuntimeError("startup failure")

		def close(self):
			return None

	monkeypatch.setattr(bootstrap_module, "bootstrap_server", lambda **kwargs: FakeProc())
	monkeypatch.setattr(bootstrap_module, "p5remSession", FailingSession)

	with pytest.raises(bootstrap_module.BootstrapError) as excinfo:
		bootstrap_module.bootstrap_session(
			host="fake-host",
			remote_python="mamba run -n p5rem-remote python",
			use_cache=False,
		)

	message = str(excinfo.value)
	assert "remote_python command appears unavailable" in message
	assert "mamba run -n p5rem-remote python" in message


def test_bootstrap_session_reports_missing_remote_runtime_without_exit_code(monkeypatch) -> None:
	class FakeProc:
		def poll(self):
			return None

		def read_stderr_snippet(self) -> str:
			return "bash: line 1: conda: command not found"

		stdin = object()
		stdout = object()

	class FailingSession:
		def __init__(self, **kwargs):
			_ = kwargs

		def heartbeat(self):
			raise RuntimeError("startup failure")

		def close(self):
			return None

	monkeypatch.setattr(bootstrap_module, "bootstrap_server", lambda **kwargs: FakeProc())
	monkeypatch.setattr(bootstrap_module, "p5remSession", FailingSession)

	with pytest.raises(bootstrap_module.BootstrapError) as excinfo:
		bootstrap_module.bootstrap_session(
			host="fake-host",
			remote_python="conda run -n p5server python",
			use_cache=False,
		)

	message = str(excinfo.value)
	assert "remote_python command appears unavailable" in message
	assert "conda run -n p5server python" in message


def test_bootstrap_session_verbose_errors_include_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
	class FakeProc:
		def poll(self):
			return 127

		def read_stderr_snippet(self) -> str:
			return "bash: line 1: conda: command not found"

		stdin = object()
		stdout = object()

	class FailingSession:
		def __init__(self, **kwargs):
			_ = kwargs

		def heartbeat(self):
			raise RuntimeError("startup failure")

		def close(self):
			return None

	monkeypatch.setattr(bootstrap_module, "bootstrap_server", lambda **kwargs: FakeProc())
	monkeypatch.setattr(bootstrap_module, "p5remSession", FailingSession)
	monkeypatch.setenv("P5REM_BOOTSTRAP_VERBOSE_ERRORS", "1")

	with pytest.raises(bootstrap_module.BootstrapError) as excinfo:
		bootstrap_module.bootstrap_session(
			host="fake-host",
			remote_python="conda run -n p5server python",
			use_cache=False,
		)

	message = str(excinfo.value)
	assert "stderr=bash: line 1: conda: command not found" in message


def test_probe_remote_python_reports_setup_stage_failure(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setattr(
		bootstrap_module,
		"_run_probe_command",
		lambda *args, **kwargs: (41, "__P5REM_PREFLIGHT_STAGE__:setup"),
	)

	with pytest.raises(bootstrap_module.BootstrapError) as excinfo:
		bootstrap_module._probe_remote_python(
			client=object(),
			remote_python="python3",
			remote_setup="module load python/3.11",
			login_shell=True,
			timeout=10.0,
		)

	message = str(excinfo.value)
	assert "remote setup preflight failed" in message
	assert "module load python/3.11" in message


def test_probe_remote_python_reports_dependency_stage_failure(monkeypatch: pytest.MonkeyPatch) -> None:
	stderr = "ModuleNotFoundError: No module named 'cbor2'\n__P5REM_PREFLIGHT_STAGE__:dependencies"
	monkeypatch.setattr(
		bootstrap_module,
		"_run_probe_command",
		lambda *args, **kwargs: (43, stderr),
	)

	with pytest.raises(bootstrap_module.BootstrapError) as excinfo:
		bootstrap_module._probe_remote_python(
			client=object(),
			remote_python="python3",
			remote_setup=None,
			login_shell=False,
			timeout=10.0,
		)

	message = str(excinfo.value)
	assert "remote dependency preflight failed" in message
	assert "pyfive and cbor2" in message


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
