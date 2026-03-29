"""Optional real-SSH round-trip integration tests using shared file assertions."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.integration_helpers import (
	bootstrap_integration_session,
	list_remote_netcdf_files,
	local_test_data_dir,
	require_ssh_integration_config,
)
from tests.roundtrip_assertions import assert_roundtrip_file_matches


pytestmark = pytest.mark.integration


def test_remote_roundtrip_matches_local_testdata() -> None:
	config = require_ssh_integration_config()
	local_dir = local_test_data_dir()
	session = bootstrap_integration_session(config)
	try:
		assert session.stat(config.remote_dir)["is_dir"] is True
		remote_files = list_remote_netcdf_files(session, config.remote_dir)
		assert remote_files, f"no .nc files found in remote directory {config.remote_dir!r}"

		for filename in remote_files:
			local_path = local_dir / filename
			assert local_path.exists(), f"missing local reference file {local_path}"
			assert_roundtrip_file_matches(session, local_path, f"{config.remote_dir}/{filename}")
	finally:
		session.close()