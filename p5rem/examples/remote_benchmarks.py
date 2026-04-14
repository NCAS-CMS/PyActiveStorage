"""
Minimal non-GUI example: open a remote file and materialise a slice locally
via three different interfaces: https, s3, and ssh 
"""
from __future__ import annotations
from pathlib import Path
import time
from urllib.parse import urlparse

import json
import fsspec
from p5rem import bootstrap_session
from xconv2.remote_fs import ShimmyFS 
import pyfive
 
import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger().handlers[0].setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
logging.getLogger().setLevel(logging.INFO)
logging.getLogger("pyfive").setLevel(logging.INFO)

pairs = [("da193a_25_3hr__198807-198807.nc","m01s00i507_10"),
    ]

REMOTE_HOST = "xfer1"
REMOTE_PYTHON = "conda run -n jas26 python"
REMOTE_FILE = "canari/public/bnl/da193a_25_3hr__198807-198807.nc"
LOCAL_FILE = Path.home() / "data" / "da193a_25_3hr__198807-198807.nc"
VARIABLE = "m01s00i507_10"
SLICE = (slice(None), slice(None), slice(None))
BLOCKSIZE = 2*1024*1024
HTTP_URL = "https://gws-access.jasmin.ac.uk/public/canari/bnl/da193a_25_3hr__198807-198807.nc"
S3_URL = "s3://uor-aces-o.s3-ext.jc.rl.ac.uk/bnl/da193a_25_3hr__198807-198807.nc"

#S3_URL = "s3://s3.swift.home/canari/da193a_25_3hr__198807-198807.nc"

def handle_endpoint(url: str, creds: dict[str, str]) -> tuple[str, dict[str, str]]:
    parsed = urlparse(url)
    path_parts = [p for p in parsed.path.split("/") if p]
    bucket = path_parts[0]
    key = "/".join(path_parts[1:])
    normalized_url = f"s3://{bucket}/{key}"
    storage_options = dict(creds)
    # Use http for plain hostnames (no dots suggesting a real TLS endpoint) or explicit .home/.local
    tld = parsed.netloc.rsplit(".", 1)[-1].lower()
    scheme = "http" if tld in {"home", "local", "internal", "lan"} else "https"
    storage_options["client_kwargs"] = {"endpoint_url": f"{scheme}://{parsed.netloc}"}
    return normalized_url, storage_options

def get_credentials(url: str) -> dict[str, str]:
    home = Path.home()
    parsed = urlparse(url if "://" in url else f"https://{url}")
    # Strip port if present for matching
    host = parsed.hostname or ""
    with open(home / ".mc" / "config.json") as f:
        config = json.load(f)
        for alias, creds in config.get("aliases", {}).items():
            cred_url = creds.get("url", "")
            cred_host = urlparse(cred_url).hostname or ""
            if cred_host == host:
                return {"key": creds["accessKey"], "secret": creds["secretKey"]}
        raise ValueError(f"Credentials for {host} not found in config file")
 

def http_read(cat_ranges_on) -> None:
    """ Use pyfive to open the file via HTTP and read a slice."""
    t1 = time.perf_counter()
    fs, root_path = fsspec.core.url_to_fs(HTTP_URL) 
    fs = ShimmyFS(fs, block_size=BLOCKSIZE)
    with fs.open(root_path) as remote_file:
        p5 = pyfive.File(remote_file)
        var = p5[VARIABLE]
        t2 = time.perf_counter()
        var.id.set_parallelism(cat_range_allowed=cat_ranges_on)
        data = var[SLICE]
        data_max = data.max()
        t3 = time.perf_counter()
        print(f"HTTP read: shape={data.shape} dtype={data.dtype} max={data_max} time={t2-t1:.2f}s {t3-t2:.2f}s (CR={cat_ranges_on})")


def s3_read(cat_ranges_on) -> None:
    """ Use pyfive to open the file via S3 and read a slice."""
    creds = get_credentials(S3_URL)
    normalized_url, storage_options = handle_endpoint(S3_URL, creds)
    t1 = time.perf_counter()
    base_fs, root_path = fsspec.core.url_to_fs(normalized_url, **storage_options)
    fs = ShimmyFS(base_fs, block_size=BLOCKSIZE)
    with fs.open(root_path) as remote_file:
        p5 = pyfive.File(remote_file)
        var = p5[VARIABLE]
        var.id.set_parallelism(cat_range_allowed=cat_ranges_on)
        t2 = time.perf_counter()
        data = var[SLICE]
        data_max = data.max()
        t3 = time.perf_counter()
        print(f"S3 read: shape={data.shape} dtype={data.dtype}  max={data_max} time={t2-t1:.2f}s {t3-t2:.2f}s (CR={cat_ranges_on})")   


def ssh_read(cat_ranges_on) -> None:
    """ Use p5rem to open the file via SSH and read a slice."""
    with bootstrap_session(
        host=REMOTE_HOST,
        remote_python=REMOTE_PYTHON,
        login_shell=True,
        use_cache=False,
    ) as session:
        t1 = time.perf_counter()
        with session.open(REMOTE_FILE) as remote_file:
            var = remote_file[VARIABLE]
            t2 = time.perf_counter()
            data = var[SLICE]
            data_max = data.max()
            t3 = time.perf_counter()
            print(f"SSH read: shape={data.shape} dtype={data.dtype} max={data_max} time={t2-t1:.2f}s {t3-t2:.2f}s (CR=UNAVAILBLE))")  

def posix_read(cat_ranges_on) -> None:
    """ Use pyfive to open the file via local POSIX path and read a slice."""
    t1 = time.perf_counter()
    with pyfive.File(LOCAL_FILE) as p5:
        var = p5[VARIABLE]
        var.id.set_parallelism(thread_count=None, cat_range_allowed=cat_ranges_on)
        t2 = time.perf_counter()
        data = var[SLICE]
        data_max = data.max()
        t3 = time.perf_counter()
        print(f"POSIX read: shape={data.shape} dtype={data.dtype} max={data_max} time={t2-t1:.2f}s {t3-t2:.2f} (CR={cat_ranges_on})")

def iterate(repeat=5, cat_ranges_on=False) -> None:
    for _ in range(repeat):
        http_read(cat_ranges_on)
        s3_read(cat_ranges_on)
        ssh_read(cat_ranges_on)
        posix_read(cat_ranges_on)


if __name__ == "__main__":
    iterate(cat_ranges_on=False)
    iterate(cat_ranges_on=True)