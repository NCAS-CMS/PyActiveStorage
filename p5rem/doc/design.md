---
updated: ["2026-03-28T17:02"]
---

# p5rem — Design Summary

## Overview

`p5rem` is a lightweight Python package for efficient remote HDF5 chunk serving and reduction over SSH. It allows desktop GUI tools (e.g. xconv2) to read and plot data from HDF5/NetCDF files residing on a remote HPC system, without requiring custom server infrastructure, open ports, or HPC admin involvement.

## Background and Motivation

The goal is to allow a desktop GUI (xconv2, a Qt application) to read and plot data from HDF5/NetCDF files on a remote HPC system. The key constraints are:

- No persistent server processes on HPC login nodes
- No port forwarding or tunnels (often blocked by HPC SSH config: `AllowTcpForwarding no`)
- No HPC admin involvement
- Works with standard HPC SSH access
- Server process lives and dies with the client GUI session

The design was inspired by VSCode Remote, which bootstraps a small server over SSH stdio and communicates entirely over the SSH session's stdin/stdout — no extra ports needed.

---

## Core Concept: Remote pyfive Proxy

The fundamental insight is that p5rem is a **thin proxy over a remote pyfive instance**. Rather than trying to emulate file handles or smuggle metadata across the wire, the server simply runs a real pyfive instance locally on the HPC — where HDF5 metadata reads are fast and free — and the client proxies the results.

```
Client (desktop)                        Server (HPC)

rFile                                   pyfive.File (local, fast Lustre I/O)
    │                                       │
    ├── file_open()   ──────────────────►  f = pyfive.File(path)
    │   ◄── keys, attrs, mtime ──────────  return serialise(f.keys(), f.attrs)
    │                                       │
    ├── var_open()    ──────────────────►  ds = f[varname]
    │   ◄── shape, dtype, index ─────────  return serialise(ds.shape, ds.id.index...)
    │                                       │
    └── get_chunk()   ──────────────────►  ds.id._get_raw_chunk(storeinfo)
        ◄── raw bytes ───────────────────  return chunk_bytes
```

The server does all HDF5 heavy lifting — superblock, object headers, btree traversal — entirely locally at local fast disk speed. No metadata buffering gymnastics, no file handle emulation, no changes to pyfive's high-level API.

---

## Architecture

The server stub is launched via SSH using QProcess (in the Qt GUI) or subprocess (standalone/test). Communication is over stdin/stdout of the SSH session using CBOR-framed messages.

```
Desktop GUI (xconv2)
    │
    └── QProcess
            └── ssh user@hpc "python -m p5rem.server"
                    │
                    │  stdin/stdout (CBOR framing)
                    │
                    └── p5rem server stub (HPC side)
                            ├── pyfive          (local HDF5 reading, fast)
                            └── pyactivestorage (reductions)
```

p5rem has no Qt dependency — the QProcess/subprocess choice is made by the caller.

---

## Key Design Decisions

### Transport: SSH stdio

- The SSH process is the QProcess target — no ports, no tunnels
- stdin/stdout carry all messages — identical pattern to VSCode Remote
- Server exits automatically when the SSH connection closes
- No HPC admin involvement — rides standard SSH

### Single Persistent SSH Connection

- Established once at GUI startup, reused for all file operations
- FILE_OPEN/CLOSE manage file lifecycle, not connection lifecycle
- HEARTBEAT keeps the SSH session alive during idle periods
- On GUI close, QProcess destruction closes stdin and the server exits on EOF
- On network dropout, bootstrap.py reconnects and restarts the stub transparently

### Bootstrap: SFTP upload + exec

- paramiko uploads the server stub to the HPC at session start via SFTP
- The stub is then executed via SSH
- Longer term: `pip install p5rem` on HPC makes bootstrap trivial
- Version sync guaranteed — client and server are the same package
- paramiko is only needed for bootstrap — all ongoing communication is SSH stdio

### Filesystem Navigation: server-side os calls

- LIST and STAT handled by the server stub via os.listdir/os.stat
- No separate SFTP connection needed — same SSH stdio transport
- Single connection handles file navigation, metadata, chunks, and reductions

### File Access: remote pyfive proxy

- Server runs a real pyfive.File instance locally on the HPC
- All HDF5 metadata reads (superblock, object headers, btree) happen at Lustre speed on the server — zero SSH round-trip latency for metadata
- Client receives serialised pyfive results and caches them
- Chunk reads are the only thing that travel over the wire as raw bytes
- Access is lazy — matching pyfive's natural access pattern:

```
file_open  → server: pyfive.File(path)    → keys, attrs, mtime
var_open   → server: f[varname]           → shape, dtype, chunks, btree index,
                                            fragmentation hint
get_chunk  → server: id._get_raw_chunk() → raw bytes (2MB typical)
```

### h5repack hint

- Server returns btree fragmentation info with VAR_OPEN response
- pyfive already exposes this via btree_range and consolidated_metadata APIs
- Client surfaces a hint to the user when fragmentation is detected
- Informational only — files work correctly, just with more round trips

### Reductions: pyactivestorage

- For large fields, reductions (mean, max, min etc.) are pushed to the HPC
- pyactivestorage is leveraged directly as the reduction component
- Only the reduced result travels over the wire, not raw chunk data

### GUI Integration: Environment Discovery

For GUI applications (e.g. xconv2), p5rem provides a utility to discover available Python/conda environments on the remote HPC **before bootstrapping the server**. This allows the GUI to:

1. **Detect available environments** via `discover_remote_conda_envs()` — queries `conda env list` over SSH
2. **Let users select** which Python environment to use for the server
3. **Bootstrap with the selected environment** using `bootstrap_session()` or `ReconnectingBootstrappedSession`

```python
from p5rem import discover_remote_conda_envs, bootstrap_session

# Step 1: Discover available conda environments (no server needed, just SSH)
envs = discover_remote_conda_envs(
    host="xfer1",
    ssh_config_path="~/.ssh/config",
    login_shell=True,  # Required on HPC systems with modules
)
# envs = {"base": "/path/to/miniforge3", "jas26": "/path/to/miniforge3/envs/jas26", ...}

# Step 2: User selects environment from GUI dropdown → "jas26"
# Step 3: Bootstrap server with selected environment
session = bootstrap_session(
    host="xfer1",
    remote_python="conda run -n jas26 python",
    local_script_path="/path/to/server.py",
    login_shell=True,
)

# Step 4: Use session as normal for file operations
proxy = session.open("/path/to/data.nc")
```

No server process is required to discover environments — only SSH access via Paramiko.

### cf-python / cfdm integration

- Client side uses cf-python (not xarray) for CF-conventions-aware data handling
- Next release of cfdm uses h5netcdf with pyfive by default
- cfdm receives an rFile which looks like a pyfive.File — no cfdm changes needed

### Chunk Caching: diskcache

Chunk-level disk caching survives across sessions — reopening the same file the next day reuses already-fetched chunks. 2MB chunks are expensive to re-fetch and pan/zoom operations reuse the same chunks repeatedly.

`diskcache` is the cache backend — pure Python, persistent, thread-safe, concurrent-safe, handles eviction automatically.

#### Cache key

```python
key = (host, path, byte_offset, size, mtime)
```

`mtime` is the invalidation strategy — file changes on HPC change mtime, old entries are never hit and evict naturally. No explicit invalidation needed. `mtime` is returned free in the FILE_OPEN response.

#### Cache location and sizing

```python
cache = diskcache.Cache('~/.cache/p5rem', size_limit=10 * 2**30)  # 10GB default
```

#### Shared across instances

Cache is shared across all p5rem instances on the same desktop. `diskcache` handles concurrent access safely. Cache stampede protection via transactions:

```python
with cache.transact():
    cached = cache.get(key)
    if cached is None:
        chunk = session.get_chunk(...)
        cache[key] = chunk
return cached or chunk
```

#### Metadata also cached

- FILE_OPEN responses cached by (host, path, mtime)
- VAR_OPEN responses cached by (host, path, varname, mtime)
- Metadata entries are small — no size concern

#### Cache management API

```python
p5rem.cache.clear()          # clear everything
p5rem.cache.clear(host=...)  # clear one host
p5rem.cache.info()           # size, entry count, hit rate
```

### pyfive: no changes needed

(Earlier versions proposed minor changes which are no longer needed)

---

## Wire Format

- **CBOR** via `cbor2` — consistent with pyactivestorage
- Framing: `[4 bytes: message length][N bytes: CBOR payload]` — uniform for all types
- CBOR native binary type carries chunk bytes directly — no special casing
- Every message is a CBOR map with a `type` field

```python
# metadata response example
{"type": "VAR_INFO", "shape": [180, 360], "dtype": "float32",
 "chunks": [180, 360], "index": {...}, "fragmented": False}

# chunk response — binary is native CBOR
{"type": "CHUNK_DATA", "byte_offset": 1234567, "size": 2097152,
 "filter_mask": 0, "data": b'\x89...'}
```

---

## Protocol Messages

### Requests

- `LIST path` — directory listing
- `STAT path` — file metadata (size, mtime etc.)
- `FILE_OPEN path` — open file, returns keys, attrs, mtime
- `VAR_OPEN path varname` — open variable, returns shape/dtype/chunks/btree/fragmentation
- `GET_CHUNK path varname byte_offset size` — raw chunk bytes
- `REDUCE path varname byte_offset size operation` — pyactivestorage reduction
- `FILE_CLOSE path` — release server-side pyfive.File
- `HEARTBEAT` — keep SSH session alive

### Responses

LIST_RESULT, STAT_RESULT, FILE_INFO, VAR_INFO, CHUNK_DATA, REDUCTION_RESULT, ERROR

---

## Package Structure

```
p5rem/
    ├── pyproject.toml
    ├── README.md
    ├── p5rem/
    │   ├── __init__.py
    │   ├── protocol.py     — CBOR framing, message types, encode/decode
    │   ├── proxy.py        — rFile: looks like pyfive.File to cfdm
    │   ├── session.py      — persistent connection, request/response, heartbeat
    │   ├── cache.py        — diskcache wrapper, stampede protection, management API
    │   ├── bootstrap.py    — SFTP upload and remote process launch (paramiko)
    │   └── server/
    │       ├── __main__.py — entry point: python -m p5rem.server
    │       └── stub.py     — drives local pyfive instance, handles requests
    └── tests/
        ├── test_protocol.py
        ├── test_proxy.py
        ├── test_cache.py
        ├── test_loopback.py        — full client/server, no SSH needed
        └── conftest.py             — synthetic HDF5 test files
```

---

## Key Interfaces

### proxy.py — rFile

Looks like a pyfive.File to cfdm/h5netcdf. Populated from cached FILE_OPEN and VAR_OPEN responses. Chunk reads delegate to session.get_chunk().

```python
class rFile:
    def __init__(self, session, path):
        meta = session.file_open(path)   # one round trip
        self._keys = meta['keys']
        self._attrs = meta['attrs']
        self._mtime = meta['mtime']

    def keys(self): return self._keys

    @property
    def attrs(self): return self._attrs

    def __getitem__(self, varname):
        return rDataset(self._session, self._path, varname)
```

### server/stub.py

Drives a real local pyfive instance. Protocol messages map directly to pyfive API:

```python
open_files = {}

def handle_file_open(path):
    f = pyfive.File(path)
    open_files[path] = f
    return {'keys': list(f.keys()), 'attrs': dict(f.attrs),
            'mtime': os.path.getmtime(path)}

def handle_var_open(path, varname):
    ds = open_files[path][varname]
    return {'shape': ds.shape, 'dtype': str(ds.dtype),
            'chunks': ds.chunks, 'index': serialise_index(ds.id.index),
            'fragmented': not open_files[path].consolidated_metadata}

def handle_get_chunk(path, varname, byte_offset, size):
    ds = open_files[path][varname]
    storeinfo = ds.id.get_chunk_info_by_coord(...)
    return {'data': ds.id._get_raw_chunk(storeinfo)}
```


Considerations:
 - For chunks, we want the client to decompress, and cache
 - For contiguous variables, we use the pseudo chunking option, and cache locally accordingly.
 - For all other variable types, simply serialise, return, and do not cache.
 - For metadata, cache
### session.py

Manages the persistent SSH subprocess, heartbeat, reconnection:

```python
class p5remSession:
    def __init__(self, host, username):
        self._proc = None  # QProcess or subprocess.Popen

    def file_open(self, path) -> dict: ...
    def var_open(self, path, varname) -> dict: ...
    def get_chunk(self, path, varname, byte_offset, size) -> bytes: ...
    def reduce(self, path, varname, byte_offset, size, op): ...
    def list(self, path) -> list: ...
    def stat(self, path) -> dict: ...
    def heartbeat(self): ...
```

### User-facing API

```python
with p5rem.Session(host="hpc.cluster.edu", username="user") as session:
    # filesystem navigation
    files = session.list("/scratch/user/data/")

    # open a file — returns a pyfive.File-like proxy
    with session.open("/scratch/user/data/model.nc") as f:
        fields = cf.read(f)  # cfdm sees a pyfive.File, no changes needed
```

---

## pyproject.toml

```toml
[build-system]
requires = ["setuptools>=64", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "p5rem"
version = "0.1.0"
description = "Remote HDF5 chunk serving and reduction over SSH"
readme = "README.md"
license = {text = "BSD-3-Clause"}
authors = [
    {name = "NCAS-CMS", email = "cms@ncas.ac.uk"}
]
requires-python = ">=3.10"
dependencies = [
    "pyfive>=0.5.0",
    "pyactivestorage",  
    "paramiko",
    "numpy>=2.0.0",
    "diskcache",
    "cbor2",            # declared explicitly as we depend on it directly
]

[project.optional-dependencies]
dev = [
    "pytest",
    "pytest-cov",
]

[tool.setuptools.packages.find]
where = ["."]
include = ["p5rem*"]
```

---

## Testing Strategy

SSH is the last thing to test, not the first. Tests are layered so that the vast majority run locally with no SSH or HPC access required.

### Layer 1: Protocol tests (no processes)

Test CBOR framing in isolation — encode, decode, round-trip fidelity:

```python
def test_round_trip_var_open():
    msg = encode(VAR_OPEN, path="/foo/bar.nc", varname="temperature")
    decoded = decode(msg)
    assert decoded['type'] == 'VAR_OPEN'
    assert decoded['varname'] == 'temperature'
```

### Layer 2: Loopback client/server (no SSH)

Launch the server stub as a local subprocess — full protocol over real HDF5 test files, no SSH involved:

```python
@pytest.fixture
def local_server():
    proc = subprocess.Popen(
        ["python", "-m", "p5rem.server"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    yield proc
    proc.terminate()
```

### Layer 3: Proxy tests (mock session)

Test rFile against a mock session — verifies proxy presents the correct pyfive-like interface to cfdm without any real data or network:

```python
class MockSession:
    def file_open(self, path):
        return {'keys': ['temperature'], 'attrs': {}, 'mtime': 12345}
    def get_chunk(self, path, varname, offset, size):
        return b'\x00' * size
```

### Layer 4: Cache tests (no server, no proxy)

Test diskcache integration — eviction, concurrent access, mtime invalidation, stampede protection, management API:

```python
def test_mtime_invalidation():
    cache = p5remCache(tmp_path)
    cache.store(key_mtime_1, chunk)
    assert cache.get(key_mtime_2) is None  # different mtime = miss
```

### Layer 5: SSH integration (optional, not in CI)

Full end-to-end against a real SSH server. Marked explicitly, excluded from normal CI:

```python
@pytest.mark.integration
def test_full_ssh_session():
    with p5rem.Session(host="hpc.cluster.edu", username="user") as session:
        with session.open("/scratch/user/data/model.nc") as f:
            fields = cf.read(f)
```

### Test data

Small synthetic HDF5 files committed to the repo:

- Well-chunked file — normal case
- Fragmented btree file — tests h5repack hint
- Contiguous variable — tests non-chunked path
- Empty variable — edge case

### Summary

```
Layer 1: protocol    — pure unit tests, no I/O
Layer 2: loopback    — subprocess, real HDF5, no SSH
Layer 3: proxy       — mock session, tests pyfive-like interface
Layer 4: cache       — diskcache integration
Layer 5: SSH         — marked integration, excluded from CI
```

Layers 1-4 give high confidence before SSH is ever involved.

---

## Related Packages (NCAS-CMS ecosystem)

- **pyfive** — pure Python HDF5 reader, no C dependencies
- **pyactivestorage** — remote reductions close to data
- **cfdm** — CF data model, next release uses h5netcdf+pyfive by default
- **cf-python** — CF conventions, built on cfdm
- **xconv2** — Qt GUI that will use p5rem as its remote access layer