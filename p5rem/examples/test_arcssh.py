#!/usr/bin/env python3
"""Minimal non-GUI example: open a remote file and materialise a slice locally."""

from __future__ import annotations
from p5rem import bootstrap_session
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger().handlers[0].setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("p5rem").setLevel(logging.DEBUG)

REMOTE_HOST = "arc-ssh"
REMOTE_PYTHON = "conda run -n jas26 python"
REMOTE_FILE = "p5test/test1.nc"
REMOTE_FILE = "canari/public/bnl/da193a_25_3hr__198807-198807.nc"
VARIABLE = "tas"
VARIABLE = "m01s00i507_10"
SELECTION = (slice(None), slice(None), slice(None))

def main() -> None:
	with bootstrap_session(
		host=REMOTE_HOST,
		remote_python=REMOTE_PYTHON,
		login_shell=True,
		use_cache=False,
	) as session:
		with session.open(REMOTE_FILE) as remote_file:
			for v in remote_file:
				var = remote_file[v]
				print(f"variable: {var.name} shape={var.shape} dtype={var.dtype}")
			data = remote_file[VARIABLE][SELECTION]
			print(f"file: {REMOTE_FILE}")
			print(f"variable: {VARIABLE}")
			print(f"selection: {SELECTION}")
			print(f"shape: {data.shape}")
			print(f"dtype: {data.dtype}")
			print(f"min={data.min()} max={data.max()}")


if __name__ == "__main__":
	main()