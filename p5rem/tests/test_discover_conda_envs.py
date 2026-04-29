"""Tests for discover_remote_conda_envs() utility."""

import pytest

from tests.integration_helpers import discover_integration_envs, require_ssh_integration_config


@pytest.mark.integration
def test_discover_remote_conda_envs_real_ssh():
	config = require_ssh_integration_config()
	envs = discover_integration_envs(config)

	# Verify result is a dict with at least "base" env
	assert isinstance(envs, dict), f"Expected dict, got {type(envs)}"
	assert "base" in envs, f"Expected 'base' env, got keys: {list(envs.keys())}"
	assert isinstance(envs["base"], str), "env path should be a string"
	assert len(envs["base"]) > 0, "env path should not be empty"
	
	# Print available envs for debugging
	print(f"\nDiscovered {len(envs)} conda environment(s):")
	for name, path in sorted(envs.items()):
		print(f"  {name:20} {path}")
