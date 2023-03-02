### Orthogonal selector

Suppose we have a selection like `S = (slice(0, 2, 1), slice(4, 6, 1), slice(7, 9, 1))`; the
procedure to return it, as a (2, 2, 2) array is:

- stuff happening in `zarr.indexing.OrthogonalIndexer`
- first builds an Indexer=Indexer(data.shape=(10, 10, 10), chunks=(3, 3, 1), selection=S)
- this returns the chunks that overlap the selected selection:
  - chunks coordinates (x(chunk selection slice), y(chunk selection slice), z(chunk selection slice)),
    example for our case:
    ```
    Chunk1:
    ChunkDimProjection(dim_chunk_ix=0, dim_chunk_sel=slice(0, 2, 1), dim_out_sel=slice(0, 2, None)),
    ChunkDimProjection(dim_chunk_ix=1, dim_chunk_sel=slice(1, 3, 1), dim_out_sel=slice(0, 2, None)),
    ChunkDimProjection(dim_chunk_ix=7, dim_chunk_sel=slice(0, 1, 1), dim_out_sel=slice(0, 1, None))

    Chunk2:
    ChunkDimProjection(dim_chunk_ix=0, dim_chunk_sel=slice(0, 2, 1), dim_out_sel=slice(0, 2, None)),
    ChunkDimProjection(dim_chunk_ix=1, dim_chunk_sel=slice(1, 3, 1), dim_out_sel=slice(0, 2, None)),
    ChunkDimProjection(dim_chunk_ix=8, dim_chunk_sel=slice(0, 1, 1), dim_out_sel=slice(1, 2, None)))
    ```
    this would be chunk (0, 1, 7), with selected bits on x=x(0:2), y=y(1:3), z=z(0:1) of **chunk**, and
    chunk (0, 1, 8), with selected bits on x=x(0:2), y=y(1:3), z=z(0:1) of **chunk**; this also
    contains the info on the indexing on the actual output (2, 2, 2) data too in `dim_out_sel`;
  - in our case we have two chunks overlapping the desired selection S:
    xyz = (0, 1, 7) and xyz = (0, 1, 8);
  - this is returned as `ChunkProjection(chunk_coords, chunk_selection, out_selection)`
- then we switch modules and go to `zarr.core` in `zarr.core.Array.get_orthogonal_selection`
  that takes that (iterator) of ChunkProjections straight to `_get_selection(indexer=indexer, out=out...)`
- - the **key info** here is in the docstring of `_get_selection()` func:
```python
        # We iterate over all chunks which overlap the selection and thus contain data
        # that needs to be extracted. Each chunk is processed in turn, extracting the
        # necessary data and storing into the correct location in the output array.

        # N.B., it is an important optimisation that we only visit chunks which overlap
        # the selection. This minimises the number of iterations in the main for loop.
```
- then items from chunks are retrieved via `_chunk_getitem()` which:
  - uses the chunk coordinates `chunk_coords` eg (0, 1, 7) to locate the chunk of interest;
  - then uses the `chunk_selection` tuple eg `chunk_selection = (slice(0, 2, 1), slice(1, 3, 1), slice(0, 1, 1))` to locate the region in the chunk
    where the data of the selection we need that the chunk overlaps lives;
  - there in lies my Trojan `self._process_chunk_V(chunk_selection)` that performs the PCI computation;
  - the PCI gets that very same `chunk_selection`, (and `self.chunks`), of the particular chunk with data;
  - the PCI says it straight:
```
Iterator to retrieve the specific coordinates of requested data
    from within a compressed chunk
```
  - and what it returns is numbers of elements:
```
    start: int
        elements offset in the chunk to read from
    nitems: int
        number of elements to read in the chunk from start
```
    but also the slices (positions) of data from the chunk in the new array:
```
    partial_out_selection: list of slices
        indices of a temporary empty array of size `Array._chunks` to assign
        the decompressed data to after the partial read.
```


  
