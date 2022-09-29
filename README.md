# Active Storage Data Interface

This package provides 

1. the class `Active`, which is a shimmy to NetCDF4 (and HDF5) storage via kerchunk metadata and the zarr indexer. It does not however, use zarr for the actual read.
2. The actual reads are done in the methods of `storage.py`, which are called from within an `Active __getitem__`.

Example usage is in the file test_harness.py, but it's basically this simple:

```python
active = Active(self.testfile)
active.method=np.mean
var = active['data']
result2 = var[0:2,4:6,7:9]
```
where `result2` will be the mean of the appropriate slice of the hyperslab in `var`.

There are some (relatively obsolete) documents from our exploration of zarr internals in the docs4understanding, but they are not germane to the usage of the Active class.

