# Dataset–File Lifetime Design Discussion

## Background

pyfive allows Datasets to outlive the File that created them:

```python
f = pyfive.File("some_file")
v = f["var"]
f.close()
y = max(v[:])   # works — Dataset reopens the file internally
```

This works on POSIX because reopening a local file is cheap (microseconds).
The parsed B-tree is already cached in the Dataset; reopening only happens
to access data bytes.

The original motivation was to simplify Dask usage and minimise the number
of concurrently open file handles.

## The problem

This "reopen is cheap" assumption breaks in two new contexts:

| Context | Reopen cost | State to preserve |
|---------|-------------|-------------------|
| POSIX   | ~microseconds | Just the path |
| fsspec  | Varies (ms–seconds) | URL + cache/session config |
| p5rem/SSH | Seconds | SSH connection + remote parsed state |

- **fsspec**: If the file was opened with caching, the Dataset would need to
  preserve storage options and cache state to reuse the cache on reopen.
- **p5rem**: Reopening means a full SSH handshake plus remote file parse,
  making the pattern impractically slow.

## Design options

### 1. Pluggable opener in pyfive (recommended long-term)

Instead of Dataset storing a filename string and calling `open()` itself,
have it store a callable or small protocol object:

```python
class FileResource(Protocol):
    def open(self) -> BinaryIO: ...
    def close(self) -> None: ...
```

- **POSIX**: `LocalFileResource(path)` — trivially cheap, current behaviour.
- **fsspec**: `FsspecFileResource(url, storage_options)` — preserves cache config.
- **p5rem**: `RemoteFileResource(connection_pool, remote_path)` — reuses SSH sessions.

This pushes the "how to get bytes" question to where it belongs without
changing pyfive's "datasets are independent" contract.

### 2. Connection pooling in p5rem (pragmatic short-term)

A pool keyed by `(host, user, remote_path)` with reference counting:

- `rFile.__init__` acquires from the pool.
- `rFile.close()` releases back to the pool (not a true close).
- `rDataset` holds a reference to the pool entry, keeping it alive.
- Pool entries have a TTL/idle timeout for cleanup.

This means cfdm calling `close()` on the rFile doesn't kill the SSH session
while any rDataset still references it.

**Tradeoff**: Fixes the issue at the p5rem layer only; doesn't help fsspec.

### 3. Don't solve the Dask problem at the file-handle level

Modern Dask typically expects workers to independently open files through a
serialisable "opener" token. Holding a parsed B-tree in the Dataset and
reopening just for data reads is an optimisation that only helps local files.

For remote/networked contexts a cleaner Dask pattern is pure-function
chunk readers that each worker calls independently, sidestepping the
"dataset outlives file" question entirely.

## Recommendation

- **Short term**: Connection pooling in p5rem (option 2). Self-contained,
  fixes the immediate problem without cross-project coordination.
- **Medium term**: Pluggable opener in pyfive (option 1). Fixes the
  assumption at its root for POSIX, fsspec, and p5rem alike. The opener
  protocol is small and can default to current behaviour for backward
  compatibility.
