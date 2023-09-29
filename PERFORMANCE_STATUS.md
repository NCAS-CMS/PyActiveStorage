## Testing overview

We have written unit and integration tests, and employ a coverage measurement tool - Codecov, see PyActiveStorage [test coverage](https://app.codecov.io/gh/valeriupredoi/PyActiveStorage) with current coverage of 87%; our Continuous Integration (CI) testing is deployed on [Github Actions](https://github.com/valeriupredoi/PyActiveStorage/actions), and we have nightly tests that run the entire testing suite, to be able to detect any issues introduced by updated versions of our dependencies. Github Actions (GA) tests also test the integration of various storage types we currently support; as such, we have dedicated tests that test Active Storage with S3 storage (by creating and running a MinIO client from within the test, and deploying and testing PyActiveStorage with data shipped to the S3 client).

Of particular interest are performance tests, and we have started using tests that measure system run time and resident memory (RES); we use ``pytest-monitor`` for this purpose, inside the GA CI testing environemnt. So far, performance testing showed us that HDF5 chunking is paramount for performance `ie` a large number of small HDF5 chunks leads to very long system run times, and high memory consumption; however, larger HDF5 chunks significantly increase performance – as an example, running PyActiveStorage on an uncompressed netCDF4 file of size 1GB on disk (500x500x500 data elements, float64 each), with optimal HDF5 chunking (eg 75 data elements per chunk, on each dimesnional axis) takes order 0.1s for a local POSIX storage and 0.3s for the case when the file is on an S3 server; the same run needs only order approx. 100MB of RES memory for each of the two storage options see [test result](https://github.com/valeriupredoi/PyActiveStorage/actions/runs/6313871715/job/17142905423?pr=146); the same types of runs with much smaller HDF5 chunks (eg 20x smaller) will need order a factor of 300 more time to complete, and order a few GB of RES memory.

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
