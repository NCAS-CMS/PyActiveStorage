[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
[![Documentation Status](https://app.readthedocs.org/projects/pyactivestorage/badge/?version=latest)](https://pyactivestorage.readthedocs.io/en/latest/?badge=latest)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Test](https://github.com/NCAS-CMS/PyActiveStorage/actions/workflows/run-tests.yml/badge.svg)](https://github.com/NCAS-CMS/PyActiveStorage/actions/workflows/run-tests.yml)
[![codecov](https://codecov.io/gh/NCAS-CMS/PyActiveStorage/graph/badge.svg?token=1olGjnvAOp)](https://codecov.io/gh/NCAS-CMS/PyActiveStorage)

![pyactivestoragelogo](https://raw.githubusercontent.com/NCAS-CMS/PyActiveStorage/main/doc/figures/PyActiveStorage-logo-complete.jpg)

## PyActiveStorage

- [Latest documentation on ReadTheDocs (RTD)](https://pyactivestorage.readthedocs.io/en/latest/)
- [RTD latest builds](https://app.readthedocs.org/projects/pyactivestorage/)
- [GHA Tests](https://github.com/NCAS-CMS/PyActiveStorage/actions)

### Create virtual environment

Use a Miniconda3 package maintainer tool, [download for Linux](https://docs.conda.io/en/latest/miniconda.html#linux-installers).

```bash
(base) conda install -c conda-forge mamba
(base) mamba env create -n activestorage -f environment.yml
conda activate activestorage
```

### Install with `pip`

```bash
pip install -e .
```

### Run tests

```bash
pytest -n 2
```

### Main dependencies

- Python versions supported: 3.10, 3.11, 3.12, 3.13. Fully compatible with `numpy >=2.0.0`.
- [Pyfive](https://anaconda.org/conda-forge/pyfive) needs to be pinned `>=0.5.0` (first fully upgraded Pyfive version).

## Active Storage Data Interface

This package provides 

1. the class `Active`, which is a shimmy to NetCDF4 (and HDF5) via a [`Pyfive.File`](https://github.com/NCAS-CMS/pyfive) file object
2. The actual reads are done in the methods of `storage.py` or `reductionist.py`, which are called from within an `Active.__getitem__`.

Example usage is in the test files, depending on the case:

- [`tests/test_harness.py`](https://github.com/NCAS-CMS/PyActiveStorage/blob/main/tests/test_harness.py)
- [`test_real_s3.py`](https://github.com/NCAS-CMS/PyActiveStorage/blob/main/tests/test_real_s3.py)
- [`test_real_https.py`](https://github.com/NCAS-CMS/PyActiveStorage/blob/main/tests/test_real_https.py)

but it's basically this simple:

```python
active = Active(file.Path | Pyfive.Dataset, ncvar="some_var")
active._version = 2
result = active.mean[0:2, 4:6, 7:9]
```

where `result` will be the mean of the appropriate slice of the hyperslab in `some_var` variable data.

There are some (relatively obsolete) documents from our exploration of zarr internals in the docs4understanding, but they are not germane to the usage of the Active class.

## Storage types

PyActiveStorage is designed to interact with various storage backends.
The storage backend is automatically detected, but can still be specified using the `storage_type` argument to the `Active` constructor.
There are two main integration points for a storage backend:

#. Load netCDF metadata
#. Perform a reduction on a storage chunk (the `reduce_chunk` function)

### Local file

The default storage backend is a local file.
To use a local file, use a `storage_type` of `None`, which is its default value.
netCDF metadata is loaded using the [netCDF4](https://pypi.org/project/netCDF4/) library.
The chunk reductions are implemented in `activestorage.storage` using NumPy.

### S3-compatible object store

We now have support for Active runs with netCDF4 files on S3, from [PR 89](https://github.com/NCAS-CMS/PyActiveStorage/pull/89).
To achieve this we integrate with [Reductionist](https://github.com/stackhpc/reductionist-rs), an S3 Active Storage Server.
Reductionist is typically deployed "near" to an S3-compatible object store and provides an API to perform numerical reductions on object data.
To use Reductionist, use a `storage_type` of `s3`.

To load metadata, netCDF files are opened using `s3fs`, with `h5netcdf` used to put the open file (which is nothing more than a memory view of the netCDF file) into an hdf5/netCDF-like object format.
Chunk reductions are implemented in `activestorage.reductionist`, with each operation resulting in an API request to the Reductionist server.
From there on, `Active` works as per normal.

### HTTPS-compatible on an NGINX server

The same infrastructure as for S3, but the file is passed in as an `https` URI.

## Testing overview

We have written unit and integration tests, and employ a coverage measurement tool - Codecov, see PyActiveStorage [test coverage](https://app.codecov.io/gh/NCAS-CMS/PyActiveStorage) with current coverage of 87%; our Continuous Integration (CI) testing is deployed on [Github Actions](https://github.com/NCAS-CMS/PyActiveStorage/actions), and we have nightly tests that run the entire testing suite, to be able to detect any issues introduced by updated versions of our dependencies. Github Actions (GA) tests also test the integration of various storage types we currently support; as such, we have dedicated tests that test Active Storage with S3 storage (by creating and running a MinIO client from within the test, and deploying and testing PyActiveStorage with data shipped to the S3 client).

Of particular interest are performance tests, and we have started using tests that measure system run time and resident memory (RES); we use ``pytest-monitor`` for this purpose, inside the GA CI testing environemnt. So far, performance testing showed us that HDF5 chunking is paramount for performance `ie` a large number of small HDF5 chunks leads to very long system run times, and high memory consumption; however, larger HDF5 chunks significantly increase performance – as an example, running PyActiveStorage on an uncompressed netCDF4 file of size 1GB on disk (500x500x500 data elements, float64 each), with optimal HDF5 chunking (eg 75 data elements per chunk, on each dimesnional axis) takes order 0.1s for a local POSIX storage and 0.3s for the case when the file is on an S3 server; the same run needs only order approx. 100MB of RES memory for each of the two storage options see [test result](https://github.com/NCAS-CMS/PyActiveStorage/actions/runs/6313871715/job/17142905423?pr=146); the same types of runs with much smaller HDF5 chunks (eg 20x smaller) will need order a factor of 300 more time to complete, and order a few GB of RES memory.

## Testing HDF5 chunking

### Test No. 1 specs

- netCDF4 1.1GB file (on disk, local)
- no compression, no filters
- data shape = (500, 500, 500)
- chunks = (75, 75, 75)

### Ran a null test

(only test module for imports and fixtures)

Ran 30 instances = 101-102M max RES

### Run kerchunk's translator to JSON

Ran 30 instances = 103M max RES

### Ran an Active v1 test

30 tests = 107-108M max RES

So kerchunking only takes 1-2M of RES memory; Active in total ~7M RES memory!


### Test No. 2 specs

- netCDF4 1.1GB file (on disk, local)
- no compression, no filters
- data shape = (500, 500, 500)
- chunks = (25, 25, 25)

### Run kerchunk's translator to JSON

Ran 30 instances = 111M max RES

### Ran an Active v1 test

30 tests = 114-115M max RES

Kerchunking needs 9MB and Active v1 in total 13-14M of max RES memory


### Test No. 3 specs

- netCDF4 1.1GB file (on disk, local)
- no compression, no filters
- data shape = (500, 500, 500)
- chunks = (8, 8, 8)

### Run kerchunk's translator to JSON

Ran 30 instances = 306M max RES

### Ran an Active v1 test

30 tests = 307M max RES

Kerchunking needs ~200MB same as Active in total - kerchunking is memory-dominant in the case of tiny HDF5 chunks.


### Some conclusions

- HDF5 chunking is make or break
- Memory appears to grow expentially of form ``F(M) = M0 + C x M ^ b`` where ``M0`` is the startup memory (module imports, test fixtures etc - here, about 100MB RES), ``C`` is a constant (probably close to 1), and ``b`` is the factor at which chunks decrease in size (along one axis, eg 3 here)

## Documentation

See available Sphinx [documentation](https://pyactivestorage.readthedocs.io/en/latest/). To build locally the documentation run:

```
sphinx-build -Ea doc doc/build
```

Docs are webhooked to build on Pull Requests, and pushes.

## Code coverage (test coverage)

We monitor test coverage via the [Codecov app](https://app.codecov.io/gh/NCAS-CMS/PyActiveStorage) and employ a bot that displays coverage changes introduced in every PR; the bot posts a comment directly to the PR, in which coverage variations introduced by the proposed code changes are displayed.
