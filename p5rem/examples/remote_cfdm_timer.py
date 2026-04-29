#!/usr/bin/env python3
"""Minimal non-GUI example: open a remote file and materialise a slice locally."""

from __future__ import annotations
from p5rem import bootstrap_session
import logging
import cf
import time

logging.basicConfig(level=logging.DEBUG)
logging.getLogger().handlers[0].setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("p5rem").setLevel(logging.DEBUG)

REMOTE_HOST = "xfer1"
REMOTE_PYTHON = "conda run -n jas26 python"
#REMOTE_FILE = "p5test/test1.nc"
REMOTE_FILE = "canari/public/bnl/da193a_25_3hr__198807-198807.nc"
#VARIABLE = "tas"
VARIABLE = "m01s00i507_10"
SELECTION = (slice(None), slice(None), slice(None))

def main() -> None:
	with bootstrap_session(
		host=REMOTE_HOST,
		remote_python=REMOTE_PYTHON,
		login_shell=True,
		use_cache=True,
	) as session:
		with session.open(REMOTE_FILE) as remote_file:
			p1 = time.perf_counter()
			fields = cf.read(remote_file)
			p2 = time.perf_counter()
			print(f"cf.read time: {p2-p1:.2f}s for len{fields} fields")
			for f in fields:
				print(f)
			p3 = time.perf_counter()
			print(f"Iterating fields time: {p3-p2:.2f}s")
			

if __name__ == "__main__":
	main()