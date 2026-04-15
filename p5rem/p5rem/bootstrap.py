"""Bootstrap helpers for uploading and launching the remote server over SSH."""

from __future__ import annotations

from contextlib import suppress
import logging
from pathlib import Path
from pathlib import PurePosixPath
import shlex
import threading
from typing import Any, Callable
import paramiko
from .session import SessionError, p5remSession
from time import perf_counter

log = logging.getLogger(__name__)
_DEFAULT_REMOTE_SERVER = str(Path(__file__).parent / "remote_server.py")


class BootstrapError(RuntimeError):
	"""Raised when bootstrap operations fail."""

class BootstrappedProcess:
	"""Minimal process-like wrapper around a Paramiko channel."""

	def __init__(
		self,
		*,
		client: Any,
		channel: Any,
		stdin: Any,
		stdout: Any,
		stderr: Any,
		remote_path: str,
		command: str,
	) -> None:
		self.client = client
		self.channel = channel
		self.stdin = stdin
		self.stdout = stdout
		self.stderr = stderr
		self.remote_path = remote_path
		self.command = command

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

	def read_stderr_snippet(self, *, max_bytes: int = 4096) -> str:
		"""Best-effort stderr tail for diagnostics without blocking on live channels."""

		chunks: list[bytes] = []
		remaining = max(1, int(max_bytes))

		recv_ready = getattr(self.channel, "recv_stderr_ready", None)
		recv = getattr(self.channel, "recv_stderr", None)
		if callable(recv_ready) and callable(recv):
			while remaining > 0 and recv_ready():
				data = recv(min(remaining, 1024))
				if not data:
					break
				if isinstance(data, str):
					data = data.encode("utf-8", errors="replace")
				chunks.append(data)
				remaining -= len(data)

		if not chunks and self.channel.exit_status_ready():
			with suppress(Exception):
				data = self.stderr.read(max_bytes)
				if isinstance(data, str):
					data = data.encode("utf-8", errors="replace")
				if data:
					chunks.append(data)

		if not chunks:
			return ""

		return b"".join(chunks).decode("utf-8", errors="replace").strip()


def _ensure_remote_dir(sftp: Any, remote_dir: str) -> None:
	parts = [part for part in PurePosixPath(remote_dir).parts if part and part != "/"]
	current = "/" if remote_dir.startswith("/") else "."
	for part in parts:
		current = str(PurePosixPath(current) / part)
		try:
			sftp.stat(current)
		except Exception:
			sftp.mkdir(current)


def _normalise_remote_python(remote_python: str) -> list[str]:
	"""Return a shell argv for the remote Python launcher.

	When ``conda run`` is used, force live stdio passthrough so the remote
	server can speak the binary protocol over SSH.
	"""

	argv = shlex.split(remote_python)
	if argv[:2] == ["conda", "run"]:
		has_live_stdio = "--no-capture-output" in argv or "--live-stream" in argv
		if not has_live_stdio:
			argv.insert(2, "--no-capture-output")
	return argv


def _build_command(
	remote_python: str,
	remote_path: str,
	args: tuple[str, ...],
	*,
	login_shell: bool,
) -> str:
	argv = [*_normalise_remote_python(remote_python), "-u", remote_path, *args]
	command = shlex.join(argv)
	if login_shell:
		return f"bash -lc {shlex.quote(command)}"
	return command


def _resolve_ssh_connect_params(
	*,
	host: str,
	username: str | None,
	port: int | None,
	key_filename: str | None,
	ssh_config_path: str | None,
) -> dict[str, Any]:
	resolved_host = host
	resolved_user = username
	resolved_port = port
	resolved_key = key_filename

	config_path = Path(ssh_config_path).expanduser() if ssh_config_path else Path.home() / ".ssh" / "config"
	if config_path.exists():
		ssh_config = paramiko.SSHConfig()
		with config_path.open("r", encoding="utf-8") as handle:
			ssh_config.parse(handle)
		options = ssh_config.lookup(host)
		resolved_host = str(options.get("hostname", resolved_host))
		if resolved_user is None and "user" in options:
			resolved_user = str(options["user"])
		if resolved_port is None and "port" in options:
			resolved_port = int(options["port"])
		if resolved_key is None and "identityfile" in options and options["identityfile"]:
			resolved_key = str(options["identityfile"][0])

	if resolved_port is None:
		resolved_port = 22

	return {
		"hostname": resolved_host,
		"username": resolved_user,
		"port": int(resolved_port),
		"key_filename": resolved_key,
	}




def bootstrap_server(
	*,
	host: str,
	username: str | None = None,
	local_script_path: str | None = None,
	remote_dir: str = ".p5rem",
	remote_filename: str = "p5rem_server.py",
	remote_python: str = "python3",
	port: int | None = None,
	timeout: float = 10.0,
	password: str | None = None,
	key_filename: str | None = None,
	ssh_config_path: str | None = None,
	login_shell: bool = False,
	args: tuple[str, ...] = (),
	ssh_client_factory: Callable[[], Any] | None = None,
) -> BootstrappedProcess:
	"""Upload a server script over SFTP and execute it via SSH."""

	if local_script_path is None:
		local_script_path = _DEFAULT_REMOTE_SERVER

	client_factory = ssh_client_factory or paramiko.SSHClient
	client = client_factory()
	client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
	connect_params = _resolve_ssh_connect_params(
		host=host,
		username=username,
		port=port,
		key_filename=key_filename,
		ssh_config_path=ssh_config_path,
	)
	connect_kwargs: dict[str, Any] = {
		"hostname": connect_params["hostname"],
		"port": connect_params["port"],
		"username": connect_params["username"],
		"password": password,
		"key_filename": connect_params["key_filename"],
		"timeout": timeout,
		# Support agent-based auth and discovered keys even when key_filename is unset.
		"allow_agent": True,
		"look_for_keys": True,
	}
	t1 = perf_counter()
	log.info(
		"Connecting to %s:%s as %s",
		connect_params["hostname"],
		connect_params["port"],
		connect_params["username"],
	)
	try:
		client.connect(**connect_kwargs)
	except paramiko.AuthenticationException:
		# Password auth failed; the server may require keyboard-interactive
		# (challenge-response / OTP).  The transport is still open — retry
		# using auth_interactive, responding to every prompt with the password.
		if password is None:
			raise
		transport = client.get_transport()
		if transport is None:
			raise
		log.info("Password auth failed; retrying with keyboard-interactive")

		def _ki_handler(title: str, instructions: str, prompt_list: list) -> list[str]:
			return [password for _ in prompt_list]

		transport.auth_interactive(connect_params["username"], _ki_handler)

	t2 = perf_counter()
	log.info("SSH connection established (%.2f seconds)", t2 - t1)

	try:
		sftp = client.open_sftp()
		try:
			_ensure_remote_dir(sftp, remote_dir)
			if hasattr(sftp, "normalize") and callable(getattr(sftp, "normalize")):
				remote_dir_path = str(sftp.normalize(remote_dir))
			else:
				remote_dir_path = remote_dir
			remote_path = str(PurePosixPath(remote_dir_path) / remote_filename)
			sftp.put(local_script_path, remote_path)
			log.info("Uploaded server script to %s (%.2f seconds)", remote_path, perf_counter() - t2)
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
		command = _build_command(remote_python, remote_path, args, login_shell=login_shell)
		log.info("Starting remote server: %s", command)
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
	username: str | None = None,
	local_script_path: str | None = None,
	remote_dir: str = ".p5rem",
	remote_filename: str = "p5rem_server.py",
	remote_python: str = "python3",
	port: int | None = None,
	timeout: float = 10.0,
	password: str | None = None,
	key_filename: str | None = None,
	ssh_config_path: str | None = None,
	login_shell: bool = False,
	use_cache: bool = True,
	args: tuple[str, ...] = (),
	ssh_client_factory: Callable[[], Any] | None = None,
) -> p5remSession:
	"""Bootstrap and return a ``p5remSession`` bound to remote stdio streams."""

	log.info("Bootstrapping session to %s", host)
	p1 = perf_counter()
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
		ssh_config_path=ssh_config_path,
		login_shell=login_shell,
		args=args,
		ssh_client_factory=ssh_client_factory,
	)
	session = p5remSession(
		host=host if use_cache else None,
		username=username,
		process=proc,
		stdin=proc.stdin,
		stdout=proc.stdout,
		cache=None,
	)

	# Fail fast with actionable diagnostics if the remote server did not start.
	if hasattr(session, "heartbeat"):
		try:
			session.heartbeat()
		except Exception as exc:
			exit_code = None
			stderr_snippet = ""
			if hasattr(proc, "poll") and callable(proc.poll):
				with suppress(Exception):
					exit_code = proc.poll()
			if hasattr(proc, "read_stderr_snippet") and callable(proc.read_stderr_snippet):
				with suppress(Exception):
					stderr_snippet = proc.read_stderr_snippet()
			with suppress(Exception):
				session.close()
			message = ["remote p5rem server failed startup"]
			if exit_code is not None:
				message.append(f"exit_code={exit_code}")
			if stderr_snippet:
				message.append(f"stderr={stderr_snippet}")
			message.append(
				"hint: verify remote_python points to an environment with pyfive and cbor2 (for conda run, keep --no-capture-output/live-stream)."
			)
			raise BootstrapError("; ".join(message)) from exc

	p2 = perf_counter()
	log.info("Session bootstrapped (host=%s, cache=%s, time=%.2f seconds)", host, "enabled" if use_cache else "disabled", p2 - p1)
	return session


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
		log.info("ReconnectingBootstrappedSession: connecting")
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
		log.info("ReconnectingBootstrappedSession: reconnecting")
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

	def reduce_chunk(
		self,
		path: str,
		varname: str,
		byte_offset: int,
		size: int,
		operation: str,
		**fields: Any,
	) -> dict[str, Any]:
		return self._call("reduce_chunk", path, varname, byte_offset, size, operation, **fields)

	def reduce_selection(
		self,
		path: str,
		varname: str,
		operation: str,
		selection: Any | None = None,
		thread_count: int = 1,
		**fields: Any,
	) -> dict[str, Any]:
		return self._call(
			"reduce_selection",
			path,
			varname,
			operation,
			selection=selection,
			thread_count=thread_count,
			**fields,
		)

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


def discover_remote_conda_envs(
	*,
	host: str,
	username: str | None = None,
	port: int | None = None,
	password: str | None = None,
	key_filename: str | None = None,
	ssh_config_path: str | None = None,
	timeout: float = 10.0,
	login_shell: bool = True,
) -> dict[str, str]:
	"""
	Discover available conda environments on a remote system.

	Executes `conda env list` via SSH and parses the output to extract environment
	names and their corresponding paths. Does NOT require the p5rem server.

	Args:
		host: SSH hostname or config alias.
		username: SSH username (resolved from config if None).
		port: SSH port (resolved from config if None).
		password: SSH password (if key auth fails).
		key_filename: Path to identity file (resolved from config if None).
		ssh_config_path: Path to SSH config file; defaults to ~/.ssh/config.
		timeout: SSH operation timeout in seconds.
		login_shell: If True, wraps `conda env list` with `bash -lc` to ensure
			conda is initialized in the login shell context.

	Returns:
		Dict mapping environment name to its path. Examples:
		```python
		{
			"base": "/path/to/miniforge3",
			"work26": "/path/to/miniforge3/envs/work26",
			"jas26": "/path/to/miniforge3/envs/jas26",
		}
		```

	Raises:
		BootstrapError: If SSH connection or command execution fails.
	"""
	# Resolve SSH connection parameters using SSH config
	connect_params = _resolve_ssh_connect_params(
		host=host,
		username=username,
		port=port,
		key_filename=key_filename,
		ssh_config_path=ssh_config_path,
	)

	client = paramiko.SSHClient()
	client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

	try:
		# Connect with optional password fallback and agent/key discovery
		client.connect(
			connect_params["hostname"],
			port=connect_params["port"],
			username=connect_params["username"],
			password=password,
			key_filename=connect_params.get("key_filename"),
			timeout=timeout,
			allow_agent=True,
			look_for_keys=True,
		)

		# Build command to list conda environments
		cmd = "conda env list"
		if login_shell:
			cmd = f"bash -lc {shlex.quote(cmd)}"

		# Execute remotely
		_stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
		output = stdout.read().decode("utf-8", errors="ignore")
		error = stderr.read().decode("utf-8", errors="ignore")

		if stdout.channel.recv_exit_status() != 0:
			raise BootstrapError(
				f"remote 'conda env list' failed: {error or output}"
			)

		# Parse output: each line is "name  /path/to/env" (or "*" prefix for active env)
		envs: dict[str, str] = {}
		for line in output.split("\n"):
			line = line.strip()
			if not line or line.startswith("#"):
				continue
			# Remove leading "*" if present (marks active env)
			if line.startswith("*"):
				line = line[1:].strip()
			# Split on whitespace; first part is name, rest is path
			parts = line.split()
			if len(parts) >= 2:
				name = parts[0]
				path = parts[-1]  # Last part is the path
				if not path.startswith("-"):  # Skip lines with flags
					envs[name] = path

		return envs

	except paramiko.AuthenticationException as exc:
		raise BootstrapError(f"SSH authentication failed for {host}: {exc}") from exc
	except paramiko.SSHException as exc:
		raise BootstrapError(f"SSH error connecting to {host}: {exc}") from exc
	except Exception as exc:
		raise BootstrapError(f"error discovering remote conda envs: {exc}") from exc
	finally:
		client.close()


__all__ = [
	"BootstrappedProcess",
	"BootstrapError",
	"ReconnectingBootstrappedSession",
	"bootstrap_reconnecting_session",
	"bootstrap_server",
	"bootstrap_session",
	"discover_remote_conda_envs",
]