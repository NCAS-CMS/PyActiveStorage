
Kerchunk effectively sets things up for [this](https://github.com/fsspec/kerchunk/blob/06a85a3c0c09807f438517eaad56e2549a53f17c/kerchunk/hdf.py#L509) to work:
```python
def _read_block(open_file, offset, size):
    place = open_file.tell()
    open_file.seek(offset)
    data = open_file.read(size)
    open_file.seek(place)
    return data
```

It is effectively called by something that looks like this:
```python
cdata = self.chunk_store[ckey]
```
where `chunk_store` is an instance of a `KVStore`, which is an `fsspec.mapping.FSMap` object.

Once the chunk is in memory, the PCI iterator is used to pull out the sub-chunks and load them into the target array. This is all explained in [[zarr getitem#Processing the Chunk]]

#### Setting up the Chunkfile

Let's have a look at how that `chunk_store` instance was created, and a bit more about it.

The arguments passed to the `Array` [class](https://github.com/zarr-developers/zarr-python/blob/44de0e4a017b8919bb5caba41eadcd67e18abdb9/zarr/core.py#L58) are initally [assigned](https://github.com/zarr-developers/zarr-python/blob/44de0e4a017b8919bb5caba41eadcd67e18abdb9/zarr/core.py#L182) to private variables:

```python
self._chunk_store = chunk_store
self._store = store
```
The arguments themselves are documented as
```python
"""
store : MutableMapping
   Array store, already initialized.
chunk_store : MutableMapping, optional
    Separate storage for chunks. If not provided, `store` will be used
    for storage of both chunks and metadata.
"""
```

NB, wrt compression, note this
```python
""" partial_decompress : bool, optional
        If True and while the chunk_store is a FSStore and the compression used
        is Blosc, when getting data from the array chunks will be partially
        read and decompressed when possible.
"""
```

The store and chunk store are [bound together via a property](https://github.com/zarr-developers/zarr-python/blob/44de0e4a017b8919bb5caba41eadcd67e18abdb9/zarr/core.py#L359):
```python
@property
    def chunk_store(self):
        """A MutableMapping providing the underlying storage for array chunks."""
        if self._chunk_store is None:
            return self._store
        else:
            return self._chunk_store
```

[V's code](https://github.com/valeriupredoi/hdf5_kerchunk_zarr/blob/main/netcdf_to_zarr.py) creates the chunk store from an hdffile  like this
```python
h5chunks = SingleHdf5ToZarr(infile, file_url, inline_threshold=300)  
# inline threshold adjusts the Size below which binary blocks are  
# included directly in the output  
# a higher inline threshold can result in a larger json file but  
# faster loading time  
fname = os.path.splitext(file_url)[0]  
out_json = ujson.dumps(h5chunks.translate()).encode()
# skipping the write to disk of the json, which we could do via StringIO ... 
fs = fsspec.filesystem("reference", fo=out_json)  
mapper = fs.get_mapper("")  # local FS mapper  
zarr_group = zarr.open_group(mapper)  
zarr_array = zarr_group.data  
```
