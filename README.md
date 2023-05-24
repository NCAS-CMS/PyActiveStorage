[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Test](https://github.com/valeriupredoi/PyActiveStorage/actions/workflows/run-tests.yml/badge.svg)](https://github.com/valeriupredoi/PyActiveStorage/actions/workflows/run-tests.yml)
[![codecov](https://codecov.io/gh/valeriupredoi/PyActiveStorage/branch/main/graph/badge.svg?token=1VGKP4L3S3)](https://codecov.io/gh/valeriupredoi/PyActiveStorage)

![pyactivestoragelogo](https://github.com/valeriupredoi/PyActiveStorage/blob/main/doc/figures/PyActiveStorage-logo-complete.jpg)

## Active Storage Prototype

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

Python versions supported: 3.9, 3.10, 3.11.

## Active Storage Data Interface

This package provides 

1. the class `Active`, which is a shimmy to NetCDF4 (and HDF5) storage via kerchunk metadata and the zarr indexer. It does not however, use zarr for the actual read.
2. The actual reads are done in the methods of `storage.py`, which are called from within an `Active.__getitem__`.

Example usage is in the file `tests/test_harness.py`, but it's basically this simple:

```python
active = Active(self.testfile, "data")
active.method = "mean"
result = active[0:2, 4:6, 7:9]
```

where `result` will be the mean of the appropriate slice of the hyperslab in `var`.

There are some (relatively obsolete) documents from our exploration of zarr internals in the docs4understanding, but they are not germane to the usage of the Active class.


## Documentation

See available Sphinx [documentation](https://htmlpreview.github.io/?https://github.com/valeriupredoi/PyActiveStorage/blob/main/doc/build/index.html). To build locally the documentation run:

```
sphinx-build -Ea doc doc/build
```
## Code coverage (test coverage)

We monitor test coverage via the [Codecov app](https://app.codecov.io/gh/valeriupredoi/PyActiveStorage) and employ a bot that displays coverage changes introduced in every PR; the bot posts a comment directly to the PR, in which coverage variations introduced by the proposed code changes are displayed.
