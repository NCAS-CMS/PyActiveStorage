"""Optional real-SSH bootstrap integration tests.

These tests are skipped unless one of the following env setups is provided:

1. Explicit connection vars:
	- ``P5REM_SSH_HOST``
	- ``P5REM_SSH_USER``
	- ``P5REM_SSH_KEY``

2. SSH config alias vars:
	- ``P5REM_SSH_HOST_ALIAS``
	- optional ``P5REM_SSH_CONFIG`` (defaults to ~/.ssh/config)

Optional vars used by both modes:
- ``P5REM_SSH_REMOTE_DIR``
- ``P5REM_SSH_PYTHON``
- ``P5REM_SSH_LOGIN_SHELL`` (set to ``1`` to run command via ``bash -lc``)
- ``P5REM_SSH_PORT`` (explicit mode only)
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from activestorage.bootstrap import bootstrap_server
from tests.integration_helpers import require_ssh_integration_config


pytestmark = pytest.mark.integration


def test_bootstrap_server_real_ssh_upload_and_exec(tmp_path: Path) -> None:
	config = require_ssh_integration_config()
	remote_dir = os.environ.get("P5REM_SSH_REMOTE_DIR", ".p5rem-test")

	local_script = tmp_path / "remote_boot.py"
	local_script.write_text("print('BOOT_OK', flush=True)\n", encoding="utf-8")

	proc = bootstrap_server(
		host=config.host,
		username=config.username,
		key_filename=config.key_filename,
		ssh_config_path=config.ssh_config_path,
		local_script_path=str(local_script),
		remote_dir=remote_dir,
		remote_filename="remote_boot.py",
		remote_python=config.remote_python,
		login_shell=config.login_shell,
		port=config.port,
	)
	try:
		exit_code = proc.wait(timeout=20)
		stdout = proc.stdout.read()
		stderr = proc.stderr.read()
		assert exit_code == 0, (
			f"remote command failed with exit={exit_code}; "
			f"command={proc.command!r}; stdout={stdout!r}; stderr={stderr!r}"
		)
		assert stdout.strip() == b"BOOT_OK"
	finally:
		proc.close()
