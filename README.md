## HDF5, Kerchunk, and Zarr guff

### Get the env done

```bash
(base) conda install -c conda-forge mamba
(base) mamba env create -n zarr-kerchunk -f environment.yml
conda activate zarr-kerchunk
```

### Analyze chunks and slices

`chunks_slices.py` constructs a 3D data array and takes it through h5py, kerchunk, and Zarr
each time getting information (byte offsets and sizes) for chunks and given slices.
