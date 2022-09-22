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

  
