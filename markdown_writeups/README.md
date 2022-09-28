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

### Output

Output is written to stdout like:

```
H5Py stuffs
==================
Dataset number of chunks is 180

Analyzing 0th and 5th chunks
Chunk index 0
Chunk index offset (0, 0, 0)
Chunk byte offset 4536
Chunk size 1228800
Chunk index 5
Chunk index offset (0, 0, 800)
Chunk byte offset 6148536
Chunk size 1228800

Total chunks size 155523840

Now looking at some slices:
Slice Dataset[0:2] shape (2, 404, 802)
Slice offset and size: 0 and 2592064


Slice Dataset[4:7] shape (3, 404, 802)
Slice offset and size 1296032 and 7776192



Kerchunk-IT stuffs
======================
Dataset number of chunks is 180
(0, 0, 0) Chunk: offset and size: 4536 1228800
(0, 0, 5) Chunk: offset and size: 6148536 1228800
Min chunk size 1228800
Max chunk size 1228800
Total size (sum of chunks), UNCOMPRESSED: 221184000


Zarr stuffs
==================
Data file loaded by Zarr
: <zarr.core.Array (60, 404, 802) float64>
Info of Data file loaded by Zarr
: Type               : zarr.core.Array
Data type          : float64
Shape              : (60, 404, 802)
Chunk shape        : (12, 80, 160)
Order              : C
Read-only          : False
Compressor         : None
Store type         : zarr.storage.DirectoryStore
No. bytes          : 155523840 (148.3M)
No. bytes stored   : 221184248 (210.9M)
Storage ratio      : 0.7
Chunks initialized : 180/180

Data chunks: (12, 80, 160)
Chunk offsets [(0, 12, 24, 36, 48), (0, 80, 160, 240, 320, 400), (0, 160, 320, 480, 640, 800)]
Zarr number of chunks 180

Analyzing 0th and 5th chunks
Chunk index 0
Chunk index offset: (0, 0, 0)
Chunk position: (slice(0, 12, None), slice(0, 80, None), slice(0, 160, None))
Chunk size: 1228800
Chunk index 5
Chunk index offset: (0, 0, 800)
Chunk position: (slice(0, 12, None), slice(0, 80, None), slice(800, 802, None))
Chunk size: 15360

Chunks information
Min chunk size: 768
Max chunk size: 1228800
Total chunks size COMPRESSED: 155523840

Total size (sum of chunks), COMPRESSED: 155523840

Now looking at some slices:
Slice Dataset[0:2] shape (2, 404, 802)
Slice offset and size 0 and 5184128


Slice Dataset[4:7] shape (3, 404, 802)
Slice offset and size 1296032 and 7776192
```
