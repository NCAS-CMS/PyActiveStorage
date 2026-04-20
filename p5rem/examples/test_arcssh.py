#!/usr/bin/env python3
"""
Minimal non-GUI example: open a remote file and materialise a slice locally.
This script is intended to be run on a local machine and accesss a remote
file via SSH as a test. The particular remote server used for this
test prompted the user for a password on the connection, and was used
to validate the use of a password alongside the use of an ssh key.
"""

from __future__ import annotations
from p5rem import bootstrap_session
from p5rem.bootstrap import BootstrapError
import logging
import sys
logging.basicConfig(level=logging.DEBUG)
logging.getLogger().handlers[0].setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("p5rem").setLevel(logging.DEBUG)

REMOTE_HOST = "arc-ssh"
REMOTE_PYTHON = "conda run -n p5server python"
REMOTE_FILE = "data/tas_day_eng.nc"
VARIABLE = "tas"
SELECTION = (slice(None), slice(None), slice(None))

def main() -> None:
	try:
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
	except BootstrapError as exc:
		print(f"Bootstrap failed: {exc}", file=sys.stderr)
		raise SystemExit(2)


if __name__ == "__main__":
	main()