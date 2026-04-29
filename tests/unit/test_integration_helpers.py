from __future__ import annotations

from tests.integration_helpers import list_remote_netcdf_files


class _FakeSession:
	def __init__(self, entries):
		self._entries = entries

	def list(self, remote_dir: str):
		_ = remote_dir
		return list(self._entries)


def test_list_remote_netcdf_files_returns_sorted_basenames() -> None:
	session = _FakeSession([
		{"type": "file", "name": "/remote/path/zeta.nc"},
		{"type": "file", "name": "/remote/path/alpha.nc"},
		{"type": "file", "name": "beta.nc"},
	])

	assert list_remote_netcdf_files(session, "/remote/path") == ["alpha.nc", "beta.nc", "zeta.nc"]


def test_list_remote_netcdf_files_filters_non_netcdf_and_invalid_entries() -> None:
	session = _FakeSession([
		{"type": "directory", "name": "/remote/path/subdir"},
		{"type": "file", "name": "/remote/path/notes.txt"},
		{"type": "file", "name": None},
		{"type": "file"},
		"not-a-dict",
		{"type": "file", "name": "/remote/path/ok.nc"},
	])

	assert list_remote_netcdf_files(session, "/remote/path") == ["ok.nc"]