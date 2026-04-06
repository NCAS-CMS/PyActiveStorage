# Pyfive Lifecycle Findings (Separated from p5rem Design)

## Purpose

This note isolates **pyfive-specific lifecycle behavior** from p5rem design work, so discussion can continue independently in another thread/window.

## Scope and source of truth

These findings are based on direct inspection of the pyfive code currently imported in the environment:

- `pyfive` module path: `/Users/bnl28/Repositories/pyfive/pyfive`
- Main files reviewed:
  - `/Users/bnl28/Repositories/pyfive/pyfive/h5d.py`
  - `/Users/bnl28/Repositories/pyfive/pyfive/high_level.py`

This is intended to avoid relying on assumptions about behavior.

## Core issue (in one sentence)

`DatasetID` reopening is implemented primarily as `open(self._filename, "rb")`, which encodes a **filename-based reopen policy** instead of an explicit transport-aware opener/resource policy.

## Exact pyfive code locations and why they matter

### 1) Reopen policy in `DatasetID._fh`

- File: `/Users/bnl28/Repositories/pyfive/pyfive/h5d.py`
- Lines: ~914-938

What it does:

- POSIX: returns `open(self._filename, "rb")` each access.
- Non-POSIX: reuses cached `self.__fh`; if closed, reopens with `open(self._filename, "rb")`.

Why this is problematic:

- Assumes path + builtin `open` is sufficient to recreate the original IO semantics.
- Does not explicitly preserve opener context (fsspec filesystem, storage options, auth/session, cache wrappers).

### 2) Filename inference in `DatasetID.__init__`

- File: `/Users/bnl28/Repositories/pyfive/pyfive/h5d.py`
- Lines: ~223 onward; relevant block ~251-278

What it does:

- Determines POSIX by probing `fh.fileno()`.
- For non-POSIX, tries to infer `self._filename` from `fh.path`, then `fh.full_name`, then `fh.fh.path`.

Why this is problematic:

- Reopen identity is reconstructed heuristically from handle attributes.
- There is no explicit opener contract that guarantees equivalent reopen behavior.

### 3) Threaded chunk reads reopen via filename

- File: `/Users/bnl28/Repositories/pyfive/pyfive/h5d.py`
- Lines: ~153-176 (`_read_parallel_threads`)

What it does:

- Opens `open(self._filename, "rb")`, uses `os.pread`, closes local handle.

Why this is problematic:

- Again assumes local path semantics for the read backend.
- Any non-local transport state carried by original handle is bypassed.

### 4) B-tree fetch strategy depends on current handle metadata

- File: `/Users/bnl28/Repositories/pyfive/pyfive/h5d.py`
- Lines: ~614-650 (`_make_btree_fetch_fn`)

What it does:

- For non-POSIX, tries to discover `fs` and `path` from current handle to use `cat_ranges`.

Why this is problematic:

- Fast-path behavior depends on whichever handle instance is present now.
- If reopen path produced a plain file handle, transport-specific features can change/disappear.

### 5) File ownership split in `File` API

- File: `/Users/bnl28/Repositories/pyfive/pyfive/high_level.py`
- `File.__init__`: ~257-308
- `File.close`: ~361-364

What it does:

- If input is file-like: `self._close = False` (pyfive does not own close).
- If input is path: `self._close = True` (pyfive owns close).

Why this matters for lifecycle:

- Ownership is tracked at file construction, but DatasetID reopen later bypasses this with direct filename reopen behavior.

## What is and is not being claimed

Claimed:

1. The code paths above concretely implement filename-based reopen.
2. These paths are sufficient to explain why remote/session/cached semantics are hard to preserve consistently.

Not claimed:

1. That pyfive is fundamentally broken for local POSIX workflows.
2. That p5rem behavior alone proves pyfive intent; this note is code-first.

## Implications for design discussion

This supports a pyfive-side requirement:

- Introduce an explicit opener/reader boundary for `DatasetID` data access, rather than requiring equivalent semantics to be reconstructed from `_filename`.

A minimal future direction (no implementation here):

- Keep current default behavior for local paths.
- Allow injected open/read policy for non-local transports.
- Keep transport/session/cache details outside core parsing logic.

## Suggested next-thread prompt (copy/paste)

"In pyfive, can we prototype a minimal opener boundary for DatasetID so `_fh` no longer hardcodes `open(_filename, 'rb')` for non-local reopen? Please preserve current local-file behavior and identify the smallest backwards-compatible change set."
