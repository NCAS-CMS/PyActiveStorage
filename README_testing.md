# PyActiveStorage Testing and CI

This document collects testing, CI, coverage, and performance-testing details.

## Run Tests

Basic local run:

```bash
pytest -n 2
```

Common focused runs used in development:

```bash
conda run -n work26 python -m pytest tests/unit tests/integration -q
pytest tests/test_real_https.py -s -k test_https
```

## Test Coverage

Coverage is tracked with Codecov:

- https://app.codecov.io/gh/NCAS-CMS/PyActiveStorage

CI posts coverage changes for pull requests.

## CI

CI is run on GitHub Actions:

- https://github.com/NCAS-CMS/PyActiveStorage/actions

Nightly and PR workflows are used to catch regressions and dependency-related
breakages.

## Testing Overview

The project includes:

- unit tests
- integration tests
- storage-backend integration tests
- performance-oriented checks in CI

Backends and interfaces covered include local, S3-compatible object storage,
and HTTPS pathways.

## HDF5 Chunking Performance Notes

Performance tests show that HDF5 chunking strongly affects runtime and memory.
Large numbers of very small chunks can significantly increase both wall-clock
runtime and resident memory.

### Test No. 1 specs

- netCDF4 1.1GB file (on disk, local)
- no compression, no filters
- data shape = (500, 500, 500)
- chunks = (75, 75, 75)

Results:

- null test: 101-102M max RES
- kerchunk translator: 103M max RES
- Active v1 test: 107-108M max RES

### Test No. 2 specs

- netCDF4 1.1GB file (on disk, local)
- no compression, no filters
- data shape = (500, 500, 500)
- chunks = (25, 25, 25)

Results:

- kerchunk translator: 111M max RES
- Active v1 test: 114-115M max RES

### Test No. 3 specs

- netCDF4 1.1GB file (on disk, local)
- no compression, no filters
- data shape = (500, 500, 500)
- chunks = (8, 8, 8)

Results:

- kerchunk translator: 306M max RES
- Active v1 test: 307M max RES

### Conclusions

- HDF5 chunking is critical for performance.
- Smaller chunks increase metadata and access overhead.
- Memory growth in this experiment was approximated as:

  ```
  F(M) = M0 + C x M ^ b
  ```

  where ``M0`` is startup memory, ``C`` is an empirical constant, and ``b``
  reflects chunk-count growth as chunk size decreases.
