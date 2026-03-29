"""Optional real-SSH bootstrap integration tests.

These tests are skipped unless explicit SSH environment variables are set.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from p5rem.bootstrap import bootstrap_server


pytestmark = pytest.mark.integration


def _required_env(name: str) -> str:
	value = os.environ.get(name)
	if not value:
		pytest.skip(f"{name} is not set; skipping SSH integration test")
	return value


def test_bootstrap_server_real_ssh_upload_and_exec(tmp_path: Path) -> None:
	host = _required_env("P5REM_SSH_HOST")
	username = _required_env("P5REM_SSH_USER")
	key_filename = _required_env("P5REM_SSH_KEY")

	remote_dir = os.environ.get("P5REM_SSH_REMOTE_DIR", ".p5rem-test")
	remote_python = os.environ.get("P5REM_SSH_PYTHON", "python3")
	port = int(os.environ.get("P5REM_SSH_PORT", "22"))

	local_script = tmp_path / "remote_boot.py"
	local_script.write_text("print('BOOT_OK', flush=True)\n", encoding="utf-8")

	proc = bootstrap_server(
		host=host,
		username=username,
		key_filename=key_filename,
		local_script_path=str(local_script),
		remote_dir=remote_dir,
		remote_filename="remote_boot.py",
		remote_python=remote_python,
		port=port,
	)
	try:
		exit_code = proc.wait(timeout=20)
		assert exit_code == 0
		assert proc.stdout.read().strip() == b"BOOT_OK"
	finally:
		proc.close()
