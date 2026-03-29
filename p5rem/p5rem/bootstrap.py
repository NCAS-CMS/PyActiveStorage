"""Bootstrap helpers for uploading and launching the remote server over SSH."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from pathlib import PurePosixPath
import shlex
import threading
from typing import Any, Callable

import paramiko

from .session import SessionError, p5remSession


class BootstrapError(RuntimeError):
	"""Raised when bootstrap operations fail."""


@dataclass
class BootstrappedProcess:
	"""Minimal process-like wrapper around a Paramiko channel."""

	client: Any
	channel: Any
	stdin: Any
	stdout: Any
	stderr: Any
	remote_path: str
	command: str

	def poll(self) -> int | None:
		if self.channel.exit_status_ready():
			return int(self.channel.recv_exit_status())
		return None

	def terminate(self) -> None:
		self.channel.close()

	def wait(self, timeout: float | None = None) -> int:
		if timeout is None:
			return int(self.channel.recv_exit_status())
		if not self.channel.exit_status_ready():
			self.channel.settimeout(timeout)
		try:
			return int(self.channel.recv_exit_status())
		except Exception as exc:
			raise BootstrapError("remote process did not exit before timeout") from exc

	def close(self) -> None:
		for stream in (self.stdin, self.stdout, self.stderr):
			try:
				stream.close()
			except Exception:
				pass
		try:
			self.channel.close()
		except Exception:
			pass
		try:
			self.client.close()
		except Exception:
			pass


def _ensure_remote_dir(sftp: Any, remote_dir: str) -> None:
	parts = [part for part in PurePosixPath(remote_dir).parts if part and part != "/"]
	current = "/" if remote_dir.startswith("/") else "."
	for part in parts:
		current = str(PurePosixPath(current) / part)
		try:
			sftp.stat(current)
		except Exception:
			sftp.mkdir(current)


def _build_command(remote_python: str, remote_path: str, args: tuple[str, ...]) -> str:
	argv = [remote_python, "-u", remote_path, *args]
	return " ".join(shlex.quote(item) for item in argv)


def bootstrap_server(
	*,
	host: str,
	username: str,
	local_script_path: str,
	remote_dir: str = ".p5rem",
	remote_filename: str = "p5rem_server.py",
	remote_python: str = "python3",
	port: int = 22,
	timeout: float = 10.0,
	password: str | None = None,
	key_filename: str | None = None,
	args: tuple[str, ...] = (),
	ssh_client_factory: Callable[[], Any] | None = None,
) -> BootstrappedProcess:
	"""Upload a server script over SFTP and execute it via SSH."""

	client_factory = ssh_client_factory or paramiko.SSHClient
	client = client_factory()
	client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
	client.connect(
		hostname=host,
		port=port,
		username=username,
		password=password,
		key_filename=key_filename,
		timeout=timeout,
	)

	try:
		sftp = client.open_sftp()
		try:
			_ensure_remote_dir(sftp, remote_dir)
			remote_path = str(PurePosixPath(remote_dir) / remote_filename)
			sftp.put(local_script_path, remote_path)
			try:
				sftp.chmod(remote_path, 0o700)
			except Exception:
				# chmod may be unavailable on some remote filesystems.
				pass
		finally:
			sftp.close()

		transport = client.get_transport()
		if transport is None:
			raise BootstrapError("ssh transport not available")
		channel = transport.open_session()
		command = _build_command(remote_python, remote_path, args)
		channel.exec_command(command)

		stdin = channel.makefile("wb")
		stdout = channel.makefile("rb")
		stderr = channel.makefile_stderr("rb")
		return BootstrappedProcess(
			client=client,
			channel=channel,
			stdin=stdin,
			stdout=stdout,
			stderr=stderr,
			remote_path=remote_path,
			command=command,
		)
	except Exception:
		client.close()
		raise


def bootstrap_session(
	*,
	host: str,
	username: str,
	local_script_path: str,
	remote_dir: str = ".p5rem",
	remote_filename: str = "p5rem_server.py",
	remote_python: str = "python3",
	port: int = 22,
	timeout: float = 10.0,
	password: str | None = None,
	key_filename: str | None = None,
	args: tuple[str, ...] = (),
	ssh_client_factory: Callable[[], Any] | None = None,
) -> p5remSession:
	"""Bootstrap and return a ``p5remSession`` bound to remote stdio streams."""

	proc = bootstrap_server(
		host=host,
		username=username,
		local_script_path=local_script_path,
		remote_dir=remote_dir,
		remote_filename=remote_filename,
		remote_python=remote_python,
		port=port,
		timeout=timeout,
		password=password,
		key_filename=key_filename,
		args=args,
		ssh_client_factory=ssh_client_factory,
	)
	return p5remSession(
		host=host,
		username=username,
		process=proc,
		stdin=proc.stdin,
		stdout=proc.stdout,
	)


class ReconnectingBootstrappedSession:
	"""Session facade that can reconnect by re-running bootstrap."""

	def __init__(
		self,
		*,
		heartbeat_interval: float | None = 30.0,
		heartbeat_max_failures: int = 3,
		**bootstrap_kwargs: Any,
	) -> None:
		self._bootstrap_kwargs = dict(bootstrap_kwargs)
		self._heartbeat_interval = heartbeat_interval
		self._heartbeat_max_failures = heartbeat_max_failures
		self._lock = threading.RLock()
		self._session: p5remSession | None = None
		self._connect()

	def _connect(self) -> p5remSession:
		session = bootstrap_session(**self._bootstrap_kwargs)
		session.set_heartbeat_failure_callback(lambda _session, _exc: self.reconnect())
		if self._heartbeat_interval is not None and self._heartbeat_interval > 0:
			session.start_heartbeat(
				interval=self._heartbeat_interval,
				max_failures=self._heartbeat_max_failures,
			)
		self._session = session
		return session

	@property
	def session(self) -> p5remSession:
		with self._lock:
			if self._session is None:
				return self._connect()
			return self._session

	def reconnect(self) -> p5remSession:
		with self._lock:
			old = self._session
			self._session = None
			if old is not None:
				with suppress(Exception):
					old.close()
			return self._connect()

	def close(self) -> None:
		with self._lock:
			session = self._session
			self._session = None
		if session is not None:
			session.close()

	def _call(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
		for attempt in (0, 1):
			session = self.session
			method = getattr(session, method_name)
			try:
				return method(*args, **kwargs)
			except (SessionError, EOFError, BrokenPipeError, OSError):
				if attempt == 0:
					self.reconnect()
					continue
				raise
		raise RuntimeError("unreachable")

	def request(self, message_type: str, /, **fields: Any) -> dict[str, Any]:
		return self._call("request", message_type, **fields)

	def list(self, path: str) -> list[Any]:
		return self._call("list", path)

	def stat(self, path: str) -> dict[str, Any]:
		return self._call("stat", path)

	def file_open(self, path: str) -> dict[str, Any]:
		return self._call("file_open", path)

	def open(self, path: str):
		return self._call("open", path)

	def var_open(self, path: str, varname: str) -> dict[str, Any]:
		return self._call("var_open", path, varname)

	def get_chunk(self, path: str, varname: str, byte_offset: int, size: int, **fields: Any) -> dict[str, Any]:
		return self._call("get_chunk", path, varname, byte_offset, size, **fields)

	def reduce(self, path: str, varname: str, byte_offset: int, size: int, operation: str, **fields: Any) -> dict[str, Any]:
		return self._call("reduce", path, varname, byte_offset, size, operation, **fields)

	def file_close(self, path: str) -> dict[str, Any]:
		return self._call("file_close", path)

	def heartbeat(self) -> dict[str, Any]:
		return self._call("heartbeat")


def bootstrap_reconnecting_session(
	*,
	heartbeat_interval: float | None = 30.0,
	heartbeat_max_failures: int = 3,
	**bootstrap_kwargs: Any,
) -> ReconnectingBootstrappedSession:
	"""Create a reconnecting bootstrap-backed session facade."""

	return ReconnectingBootstrappedSession(
		heartbeat_interval=heartbeat_interval,
		heartbeat_max_failures=heartbeat_max_failures,
		**bootstrap_kwargs,
	)


__all__ = [
	"BootstrappedProcess",
	"BootstrapError",
	"ReconnectingBootstrappedSession",
	"bootstrap_reconnecting_session",
	"bootstrap_server",
	"bootstrap_session",
]