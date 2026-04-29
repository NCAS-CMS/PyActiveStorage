#!/usr/bin/env python3
"""
Acid test: End-to-end validation of p5rem remote file operations.

This script validates that remote files accessed via p5rem produce identical
metadata and data as local pyfive reads. It connects to a remote HPC system
(via SSH config alias), bootstraps the p5rem server, and compares file
contents with local test data.

Usage:
    cd /Users/bnl28/Repositories/p5rem
    source tests/testenv.sh  # Load SSH config, host, python env
    python tests/acid_test.py [--remote-dir p5test]

Environment variables (from testenv.sh or manual):
    P5REM_SSH_HOST_ALIAS     SSH config alias (e.g., "xfer1")
    P5REM_SSH_PYTHON         Remote Python command (e.g., "conda run -n jas26 python")
    P5REM_SSH_LOGIN_SHELL    Enable login shell wrapping (1 or 0)
"""

import os
import sys
from pathlib import Path

import pyfive

# Add package to path so we can import p5rem
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.integration_helpers import (
	bootstrap_integration_session,
	discover_integration_envs,
	get_ssh_integration_config,
	list_remote_netcdf_files,
	local_test_data_dir,
)
from tests.roundtrip_assertions import compare_roundtrip_file


def load_env_config():
	"""Load SSH configuration from environment variables."""
	config = get_ssh_integration_config()
	if config is None:
		print("ERROR: P5REM_SSH_HOST_ALIAS not set")
		print("       Run: source tests/testenv.sh")
		sys.exit(1)
	return config


def discover_envs(config):
	"""Discover available conda environments on remote."""
	print("\n[1] Discovering remote conda environments...")
	try:
		envs = discover_integration_envs(config)
		print(f"    ✓ Found {len(envs)} environment(s):")
		for name in sorted(envs.keys()):
			print(f"      - {name}")
		return envs
	except Exception as e:
		import traceback
		print(f"    ✗ FAILED: {e}")
		traceback.print_exc()
		sys.exit(1)


def bootstrap_remote_session(config):
	"""Bootstrap p5rem session to remote system."""
	print("\n[2] Bootstrapping p5rem session to remote...")
	try:
		session = bootstrap_integration_session(config)
		print("    ✓ Session bootstrapped successfully")
		return session
	except Exception as e:
		print(f"    ✗ FAILED: {e}")
		sys.exit(1)


def list_remote_files(session, remote_dir):
	"""List files in remote directory."""
	print(f"\n[3] Listing files in remote {remote_dir}...")
	try:
		# First, test that the session is working with a simple request
		print(f"    Testing session with STAT on {remote_dir}...")
		stat_result = session.stat(remote_dir)
		print(f"    ✓ STAT succeeded: {stat_result}")
		
		files = session.list(remote_dir)
		print(f"    ✓ Found {len(files)} item(s):")
		nc_files = list_remote_netcdf_files(session, remote_dir)
		for entry in files:
			name = entry.get("name") if isinstance(entry, dict) else str(entry)
			marker = "  (NetCDF)" if isinstance(name, str) and name.endswith('.nc') else ""
			print(f"      - {name}{marker}")
		return nc_files
	except Exception as e:
		import traceback
		print(f"    ✗ FAILED: {e}")
		traceback.print_exc()
		session.close()
		sys.exit(1)


def get_local_file_path(filename):
	"""Resolve local test data file path."""
	test_data_dir = local_test_data_dir()
	return test_data_dir / filename


def compare_file(local_path, remote_path, session, filename):
	"""Compare one local file against one remote file using shared round-trip assertions."""

	print(f"\n[4.{filename}] Comparing {filename}...")
	try:
		local_file = pyfive.File(str(local_path))
		print(f"    Local file: {len(list(local_file.keys()))} item(s)")
		failures, skipped = compare_roundtrip_file(session, local_path, remote_path)
		if skipped:
			for item in skipped:
				print(f"    - skipped {item}")
		if failures:
			for failure in failures:
				print(f"    ✗ {failure}")
			return False
		print("    ✓ File matches")
		return True
	except Exception as e:
		print(f"    ✗ Comparison FAILED: {e}")
		return False


def main():
	"""Run acid test."""
	print("=" * 70)
	print("p5rem ACID TEST: Local vs Remote File Comparison")
	print("=" * 70)

	# Parse command line
	remote_dir = "p5test"
	if "--remote-dir" in sys.argv:
		idx = sys.argv.index("--remote-dir")
		if idx + 1 < len(sys.argv):
			remote_dir = sys.argv[idx + 1]

	# Load config
	config = load_env_config()
	print(f"\nConfiguration:")
	print(f"  Host:           {config.host}")
	print(f"  Remote Python:  {config.remote_python}")
	print(f"  Login Shell:    {config.login_shell}")
	print(f"  Remote dir:     {remote_dir}")

	# Discover environments
	envs = discover_envs(config)

	# Bootstrap session
	session = bootstrap_remote_session(config)

	try:
		# List remote files
		nc_files = list_remote_files(session, remote_dir)

		if not nc_files:
			print("\n✗ No .nc files found in remote directory!")
			sys.exit(1)

		# Get local test data directory
		local_data_dir = get_local_file_path("")

		# Compare each file
		results = {}
		for filename in nc_files:
			local_path = get_local_file_path(filename)
			remote_path = f"{remote_dir}/{filename}"

			if not local_path.exists():
				print(f"\n✗ Local file not found: {local_path}")
				results[filename] = False
				continue

			results[filename] = compare_file(local_path, remote_path, session, filename)

		# Summary
		print("\n" + "=" * 70)
		print("SUMMARY")
		print("=" * 70)
		passed = sum(1 for v in results.values() if v)
		total = len(results)
		print(f"\n{passed}/{total} files validated successfully")
		for filename, ok in results.items():
			status = "✓ PASS" if ok else "✗ FAIL"
			print(f"  {status}: {filename}")

		if passed == total:
			print("\n✓ ACID TEST PASSED: All files match!")
			sys.exit(0)
		else:
			print(f"\n✗ ACID TEST FAILED: {total - passed} file(s) mismatch")
			sys.exit(1)

	finally:
		session.close()


if __name__ == "__main__":
	main()
