from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import pytest

from p5rem import bootstrap_session, discover_remote_conda_envs


@dataclass(frozen=True)
class SSHIntegrationConfig:
	host: str
	username: str | None
	key_filename: str | None
	ssh_config_path: str | None
	port: int | None
	remote_dir: str
	remote_python: str
	login_shell: bool


def get_ssh_integration_config() -> SSHIntegrationConfig | None:
	host_alias = os.environ.get("P5REM_SSH_HOST_ALIAS")
	if host_alias:
		return SSHIntegrationConfig(
			host=host_alias,
			username=os.environ.get("P5REM_SSH_USER"),
			key_filename=os.environ.get("P5REM_SSH_KEY"),
			ssh_config_path=os.environ.get("P5REM_SSH_CONFIG"),
			port=None,
			remote_dir=os.environ.get("P5REM_SSH_REMOTE_DIR", "p5test"),
			remote_python=os.environ.get("P5REM_SSH_PYTHON", "python3"),
			login_shell=os.environ.get("P5REM_SSH_LOGIN_SHELL", "0") == "1",
		)

	host = os.environ.get("P5REM_SSH_HOST")
	username = os.environ.get("P5REM_SSH_USER")
	if host and username:
		return SSHIntegrationConfig(
			host=host,
			username=username,
			key_filename=os.environ.get("P5REM_SSH_KEY"),
			ssh_config_path=os.environ.get("P5REM_SSH_CONFIG"),
			port=int(os.environ.get("P5REM_SSH_PORT", "22")),
			remote_dir=os.environ.get("P5REM_SSH_REMOTE_DIR", "p5test"),
			remote_python=os.environ.get("P5REM_SSH_PYTHON", "python3"),
			login_shell=os.environ.get("P5REM_SSH_LOGIN_SHELL", "0") == "1",
		)

	return None


def require_ssh_integration_config() -> SSHIntegrationConfig:
	config = get_ssh_integration_config()
	if config is None:
		pytest.skip(
			"set either (P5REM_SSH_HOST,P5REM_SSH_USER[,P5REM_SSH_KEY]) "
			"or P5REM_SSH_HOST_ALIAS to run SSH integration tests"
		)
	return config


def discover_integration_envs(config: SSHIntegrationConfig) -> dict[str, str]:
	return discover_remote_conda_envs(
		host=config.host,
		username=config.username,
		port=config.port,
		key_filename=config.key_filename,
		ssh_config_path=config.ssh_config_path,
		login_shell=config.login_shell,
		timeout=20.0,
	)


def bootstrap_integration_session(config: SSHIntegrationConfig):
	return bootstrap_session(
		host=config.host,
		username=config.username,
		port=config.port,
		key_filename=config.key_filename,
		ssh_config_path=config.ssh_config_path,
		remote_python=config.remote_python,
		login_shell=config.login_shell,
		timeout=20.0,
		use_cache=False,
	)


def list_remote_netcdf_files(session, remote_dir: str) -> list[str]:
	entries = session.list(remote_dir)
	paths = [
		entry.get("name")
		for entry in entries
		if isinstance(entry, dict)
		and entry.get("type") == "file"
		and isinstance(entry.get("name"), str)
		and entry["name"].endswith(".nc")
	]
	return sorted(paths)


def local_test_data_dir() -> Path:
	return Path(__file__).parent / "data"