## Stack explained

### Problem at hand

We have a netCDF file, containing data for one or more variables:

- we want to apply a selection `S=S(slices)` where slices are `slice` objects;
- we have two alternatives:
  - on a normal day we'd do `selection = xarray.open_mfdataset(netCDF)["var"][S]` and be done with it;
  - on a crazy day we'd need to ask a machine to do that; the machine doesn't know `xarray`, doesn't even know Python,
    so we'd have to tell the machine some stuff it can understand: given any (compressed) chunk `C(x_i, y_i, z_i)`,
    we'd have to know:
      - A: chunk coordinates `(x_i, y_i, z_i)` (or, position in the chunks matrix) of the chunk that contains selection data;
      - B: where does the partial selection data that the chunk contains:
        - start (the `j`th element where the selection starts, or `j x dtype.size` for the actual bytes offset);
        - and how many `k` elements it spans, so `j + k` elements belonging to the data selection inside the chunk
          (or, `(j + k) x dtype.size` for a span bytes size);
      - C: chunk offset (bytes) in the chunk matrix;

### Solving the "crazy day" case (Active Storage)

Script of interest [`netcdf_to_zarr.py`](https://github.com/valeriupredoi/hdf5_kerchunk_zarr/blob/main/netcdf_to_zarr.py)

#### Preamble

We take a netCDF file, spin it through Kerchunk, then through fsspec, and eventually get a `zarr.core.Array` fully
utilizable with anything zarr. See [how we load it](https://github.com/valeriupredoi/hdf5_kerchunk_zarr/blob/107e80d2939e35a29ececcc17acc7468cbf1e0bf/netcdf_to_zarr.py#L54).

#### Solving the problem

To start with, we need to understand a bit about Zarr's indexing stack when doing an orthogonal selection, that info is
summarized in this [document](https://github.com/valeriupredoi/hdf5_kerchunk_zarr/blob/main/indexer_for_selection.md).

Then we can "hack" a bit `zarr.core` to expose some previously unexposed to the API functionalities, specifically we are interested in
obtaining the chunk coordinates `(x_i, y_i, z_i)` in Zarr chunk space of each chunk that will contain bits of the selection `S`, i.e. where
slices of our desired selection overlap chunks: to do that, look at the call to [`get_orthogonal_selection`](https://github.com/valeriupredoi/hdf5_kerchunk_zarr/blob/107e80d2939e35a29ececcc17acc7468cbf1e0bf/netcdf_to_zarr.py#L85) function from Zarr. This slightly changed from the stock Zarr function since it returns a nit more than the stock function:

- `data_selection`: an array-like object of the shape of our desired selection `S`, populated with crap (no real data, just its shape matters)
- `chunk_info`: a tuple containing master chunks, chunks selection indices (not important), and PCI=**PartialChunkIterator** (see below);
- `chunk_coords`: **chunks coordinates `(x_i, y_i, z_i)`** so parameter A comes straight from Zarr (of course, with the [hacked core](https://github.com/valeriupredoi/hdf5_kerchunk_zarr/blob/main/core.py) );

The set of parameters B can be extracted [from the PCI](https://github.com/valeriupredoi/hdf5_kerchunk_zarr/blob/107e80d2939e35a29ececcc17acc7468cbf1e0bf/netcdf_to_zarr.py#L93) - the PCI returns numbers of elements:
```
    start: int
        elements offset in the chunk to read from
    nitems: int
        number of elements to read in the chunk from start
```
- these are applies to each chunk that contains data from the selection (note that Zarr is **only** looking ath the chunks that contain
data from the selection, and discards all others uninteresting chunks); an application of the PCI=PCI(start, nitems) is done in the
[iteration over relevant chunks](https://github.com/valeriupredoi/hdf5_kerchunk_zarr/blob/107e80d2939e35a29ececcc17acc7468cbf1e0bf/netcdf_to_zarr.py#L120)
after we have decoded (with decompression) each of these relevant chunks to obtain the actual data sitting in there;

Parameters C


