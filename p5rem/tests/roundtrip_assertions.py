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


def _as_list(value: Any) -> list[Any] | None:
	if hasattr(value, "tolist") and callable(value.tolist):
		value = value.tolist()
	if isinstance(value, tuple):
		value = list(value)
	if isinstance(value, list):
		return value
	return None


def _resolve_ref_like(item: Any, file_obj: Any) -> str | None:
	if isinstance(item, str):
		return item.lstrip("/")
	if item is None:
		return None
	try:
		resolved = file_obj[item]
		resolved_name = getattr(resolved, "name", None)
		if isinstance(resolved_name, str):
			return resolved_name.lstrip("/")
	except Exception:
		pass
	return None


def canonicalise_dimension_list(raw_attrs: dict[str, Any], file_obj: Any) -> list[list[str | None]] | None:
	"""Normalise DIMENSION_LIST entries to ``[[dim_name_or_none], ...]``."""

	raw_dim_list = raw_attrs.get("DIMENSION_LIST")
	entries = _as_list(raw_dim_list)
	if entries is None:
		return None

	canonical: list[list[str | None]] = []
	for entry in entries:
		inner = _as_list(entry)
		item = inner[0] if inner else None
		canonical.append([_resolve_ref_like(item, file_obj)])

	return canonical


def canonicalise_reference_list(raw_attrs: dict[str, Any], file_obj: Any) -> list[list[Any]] | None:
	"""Normalise REFERENCE_LIST entries to ``[[var_name_or_none, dim_index], ...]``."""

	raw_ref_list = raw_attrs.get("REFERENCE_LIST")
	entries = _as_list(raw_ref_list)
	if entries is None:
		return None

	canonical: list[list[Any]] = []
	for entry in entries:
		inner = _as_list(entry)
		if inner is None or len(inner) < 2:
			canonical.append([None, None])
			continue

		var_name = _resolve_ref_like(inner[0], file_obj)
		dim_index = inner[1]
		if hasattr(dim_index, "item") and callable(dim_index.item):
			try:
				dim_index = dim_index.item()
			except Exception:
				pass

		canonical.append([var_name, dim_index])

	return canonical


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

			local_raw_attrs = dict(getattr(local_var, "attrs", {}))
			remote_raw_attrs = dict(getattr(remote_var, "attrs", {}))

			local_attrs = normalise_metadata_value(local_raw_attrs)
			remote_attrs = normalise_metadata_value(remote_raw_attrs)

			local_dim_list = canonicalise_dimension_list(local_raw_attrs, local_file)
			remote_dim_list = canonicalise_dimension_list(remote_raw_attrs, remote_file)
			if local_dim_list is not None or remote_dim_list is not None:
				local_attrs["DIMENSION_LIST"] = local_dim_list
				remote_attrs["DIMENSION_LIST"] = remote_dim_list

			local_ref_list = canonicalise_reference_list(local_raw_attrs, local_file)
			remote_ref_list = canonicalise_reference_list(remote_raw_attrs, remote_file)
			if local_ref_list is not None or remote_ref_list is not None:
				local_attrs["REFERENCE_LIST"] = local_ref_list
				remote_attrs["REFERENCE_LIST"] = remote_ref_list
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