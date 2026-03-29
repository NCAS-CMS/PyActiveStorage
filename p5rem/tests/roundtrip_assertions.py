from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pyfive


def normalise_metadata_value(value: Any) -> Any:
	"""Convert local pyfive metadata to the wire-safe representation used remotely."""

	try:
		from pyfive.core import Reference as pyfive_reference
		if isinstance(value, pyfive_reference):
			return None
	except ImportError:
		pass

	if hasattr(value, "tolist") and callable(value.tolist):
		return normalise_metadata_value(value.tolist())
	if hasattr(value, "item") and callable(value.item):
		try:
			return normalise_metadata_value(value.item())
		except Exception:
			pass
	if isinstance(value, Mapping):
		return {key: normalise_metadata_value(item) for key, item in value.items()}
	if isinstance(value, (tuple, list)):
		return [normalise_metadata_value(item) for item in value]
	if isinstance(value, (bool, int, float, str, bytes, type(None))):
		return value
	return None


def has_data_interface(value: Any) -> bool:
	"""Return True when a file entry behaves like a readable dataset."""

	return callable(getattr(value, "__getitem__", None)) and hasattr(value, "dtype")


def compare_roundtrip_file(session: Any, local_path: str | Path, remote_path: str) -> tuple[list[str], list[str]]:
	"""Compare one local pyfive file against one remotely opened file.

	Returns ``(failures, skipped_entries)``.
	"""

	local_file = pyfive.File(str(local_path))
	failures: list[str] = []
	skipped: list[str] = []

	with session.open(remote_path) as remote_file:
		local_keys = sorted(local_file.keys())
		remote_keys = sorted(remote_file.keys())
		if local_keys != remote_keys:
			failures.append(f"keys mismatch: local={local_keys!r} remote={remote_keys!r}")
			return failures, skipped

		for varname in local_keys:
			local_var = local_file[varname]
			remote_var = remote_file[varname]

			if not has_data_interface(local_var):
				skipped.append(f"{varname} ({type(local_var).__name__})")
				continue

			local_shape = tuple(local_var.shape)
			remote_shape = tuple(remote_var.shape)
			if local_shape != remote_shape:
				failures.append(f"{varname}: shape mismatch {local_shape!r} != {remote_shape!r}")
				continue

			local_dtype = str(local_var.dtype)
			remote_dtype = str(remote_var.dtype)
			if local_dtype != remote_dtype:
				failures.append(f"{varname}: dtype mismatch {local_dtype!r} != {remote_dtype!r}")
				continue

			local_chunks = None if getattr(local_var, "chunks", None) is None else tuple(local_var.chunks)
			remote_chunks = None if getattr(remote_var, "chunks", None) is None else tuple(remote_var.chunks)
			if local_chunks != remote_chunks:
				failures.append(f"{varname}: chunks mismatch {local_chunks!r} != {remote_chunks!r}")
				continue

			local_attrs = normalise_metadata_value(dict(getattr(local_var, "attrs", {})))
			remote_attrs = normalise_metadata_value(dict(getattr(remote_var, "attrs", {})))
			if local_attrs != remote_attrs:
				failures.append(f"{varname}: attrs mismatch {local_attrs!r} != {remote_attrs!r}")
				continue

			local_data = local_var[()]
			remote_data = remote_var[()]
			if not np.array_equal(local_data, remote_data):
				failures.append(f"{varname}: data mismatch")

	return failures, skipped


def assert_roundtrip_file_matches(session: Any, local_path: str | Path, remote_path: str) -> None:
	"""Assert that a remote file matches the local reference file."""

	failures, _ = compare_roundtrip_file(session, local_path, remote_path)
	assert failures == [], "\n".join(failures)