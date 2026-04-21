# cfdm File Ownership and Close Behavior

## Summary

When `cf.read(...)` is given an SSH-backed `p5rem.rFile`, the file handle is closed during cfdm cleanup. When `cf.read(...)` is given an HTTPS file-like object from `fsspec`, the original handle remains open.

The difference is not specific to SSH versus HTTPS. The difference is whether cfdm/p5netcdf treats the object passed in as the dataset object it owns and later closes.

## What Was Traced

A runtime trace was added around:

- `cfdm.read_write.netcdf.netcdfread.NetCDFRead.dataset_close()`
- `cfdm.read_write.netcdf.p5netcdf.p5netcdf.File.close()`
- `p5rem.proxy.rFile.close()`

This was run in the `work26t` environment.

## Result

### HTTPS case

For the HTTPS input, the trace showed:

- `dataset_representation = file_handle`
- backend = `p5netcdf`
- cleanup called `NetCDFRead.dataset_close()`
- that called `p5netcdf.File.close()`
- but the original HTTPS handle remained open afterwards

Observed runtime state after `cf.read(https_handle)`:

- `https_handle.closed == False`

### SSH case

For the SSH input, the trace showed:

- `dataset_representation = pyfive.File`
- backend = `p5netcdf`
- cleanup called `NetCDFRead.dataset_close()`
- that called `p5netcdf.File.close()`
- `p5netcdf.File.close()` then called `_h5_file.close()`
- `_h5_file` was the original `p5rem.rFile`
- that triggered `rFile.close()` on the caller-owned handle

Observed runtime state after `cf.read(ssh_handle)`:

- `ssh_handle.closed == True`

## Why The Behaviors Differ

### HTTPS input

The HTTPS object is a plain file-like handle.

`p5netcdf.File.__init__` does not treat it as an existing `pyfive.File`, so it constructs a new `pyfive.File(dataset, mode="r")` wrapper around it.

Later, during cleanup, cfdm closes the wrapper object it created. That does not mark the original HTTPS handle as closed.

### SSH input

The SSH object (`p5rem.rFile`) is recognized as a `pyfive.File`.

`p5netcdf.File.__init__` therefore uses the passed-in object directly as `_h5_file` instead of wrapping it.

Later, during cleanup, cfdm closes `_h5_file`, which is the original caller-supplied `rFile`.

## Exact Close Path

The close path for the SSH case is:

1. `cf.read(...)`
2. `cfdm.read_write.netcdf.netcdfread.NetCDFRead.read(...)`
3. `NetCDFRead.dataset_close()`
4. `nc.close()` on the p5netcdf dataset wrapper
5. `cfdm.read_write.netcdf.p5netcdf.p5netcdf.File.close()`
6. `self._h5_file.close()`
7. `p5rem.proxy.rFile.close()`

## Conclusion

The root cause is ownership semantics.

cfdm/p5netcdf closes what it considers to be the dataset object:

- for HTTPS input, that is a wrapper object created by p5netcdf
- for SSH input, that is the original caller-supplied `rFile`

So the problematic behavior is not that cfdm specially closes remote files. It is that caller-supplied `pyfive.File`-like objects are treated as owned objects and are closed during cleanup.

## Implication

If lazy reads need to remain valid after `cf.read(remote_file)`, then the ownership rule must change somewhere in the pyfive/p5netcdf path.

Possible fix points include:

1. cfdm/p5netcdf should not auto-close caller-supplied `pyfive.File`-like objects
2. there should be an explicit distinction between wrapper-owned handles and borrowed external handles
3. p5rem could expose a non-owning adapter instead of passing the true `rFile` directly
