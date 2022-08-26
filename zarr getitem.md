Working from [this piece of code](https://github.com/zarr-developers/zarr-python/blob/44de0e4a017b8919bb5caba41eadcd67e18abdb9/zarr/core.py#L1872):

- [getitem](https://github.com/zarr-developers/zarr-python/blob/44de0e4a017b8919bb5caba41eadcd67e18abdb9/zarr/core.py#L648)
	- first pops the field to get at the selection and the field via `self.`[pop_fields](https://github.com/zarr-developers/zarr-python/blob/44de0e4a017b8919bb5caba41eadcd67e18abdb9/zarr/indexing.py#L874)
	- `self.ndim`  is a property which is the number of dimensions in the shape
	- the simplest case uses `self.`[get_basic_selection](https://github.com/zarr-developers/zarr-python/blob/44de0e4a017b8919bb5caba41eadcd67e18abdb9/zarr/core.py#L791)
	```
	def get_basic_selection(self, selection=Ellipsis, out=None, fields=None):
        """
        Retrieve data for an item or region of the array.
        Parameters
        ----------
        selection : tuple
            A tuple specifying the requested item or region for each dimension of 
            the array. May be any combination of int and/or slice for
            multidimensional arrays.
        out : ndarray, optional
            If given, load the selected data directly into this array.
        fields : str or sequence of str, optional
            For arrays with a structured dtype, one or more fields can be specified
            to extract data for.
        Returns
        -------
        out : ndarray
            A NumPy array containing the data for the requested region.

   ```
	- The first thing this does is refresh metadata via `self.`\_[load_metadata](https://github.com/zarr-developers/zarr-python/blob/44de0e4a017b8919bb5caba41eadcd67e18abdb9/zarr/core.py#L217) and in the non zero-dimensional case it calls `self.`\_[get _basic_selection_nd](https://github.com/zarr-developers/zarr-python/blob/44de0e4a017b8919bb5caba41eadcd67e18abdb9/zarr/core.py#L951)
	- That calls [BasicIndexer](https://github.com/zarr-developers/zarr-python/blob/44de0e4a017b8919bb5caba41eadcd67e18abdb9/zarr/indexing.py#L326)(selection,self) then calls `self.`\_[get_selection](https://github.com/zarr-developers/zarr-python/blob/44de0e4a017b8919bb5caba41eadcd67e18abdb9/zarr/core.py#L1219)(indexer=indexer, out=out,fields=fields).

- `self._get_selection` ([here](https://github.com/zarr-developers/zarr-python/blob/44de0e4a017b8919bb5caba41eadcd67e18abdb9/zarr/core.py#L1219)) does the business!
	```
	# We iterate over all chunks which overlap the selection and thus contain data
    # that needs to be extracted. Each chunk is processed in turn, extracting the
    # necessary data and storing into the correct location in the output array.

    # N.B., it is an important optimisation that we only visit chunks which overlap
    # the selection. This minimises the number of iterations in the main for loop.
    
    ... determine output shape and setup output array if it isn't there already
    out_shape = indexer.shape
    out = np.empty(out_shape ...)

	# iterate over chunks, two versions, one which sequentially gets one at a time,
	# one which allows storage to get multiple items at once. The first version:

	for chunk_coords, chunk_selection, out_selection in indexer:

                # load chunk selection into output array
                self._chunk_getitem(chunk_coords, chunk_selection, out, 
					                out_selection,
                                    drop_axes=indexer.drop_axes, fields=fields)

	return out
    ```
We have two threads to chase down now, the indexer and the chunk get item. Doing the latter
first:

- `self._chunk_getitem` ([here](https://github.com/zarr-developers/zarr-python/blob/44de0e4a017b8919bb5caba41eadcd67e18abdb9/zarr/core.py#L1906)): 
    ```
    ... neglecting setup and error handling ...
    # obtain key for chunk
    ckey = self._chunk_key(chunk_coords)

    # obtain compressed data for chunk
    cdata = self.chunk_store[ckey]
    self._process_chunk(out, cdata, chunk_selection, drop_axes, out_is_ndarray, 
                        fields, out_selection)
   ```
- `self._process_chunk` ([here](https://github.com/zarr-developers/zarr-python/blob/44de0e4a017b8919bb5caba41eadcd67e18abdb9/zarr/core.py#L1823)):
    ```
    """Take binary data from storage and fill output array"""
    ... has two routes through, one where the whole chunk is wanted ant the 
    destination is contiguous so decompress can go directly to the destination
    array, and the more complicated one we are interested in ...

    ... this is the bit that uses partial chunk iterator ... 

	... but it use interesting methods on the input cdata variable to 
	do the actual reading ...

    cdata.prepare_chunk()
    # size of chunk
    tmp = np.empty(self._chunks, dtype=self.dtype)
    index_selection = PartialChunkIterator(chunk_selection, self.chunks)
    for start, nitems, partial_out_selection in index_selection:
	    expected_shape = [
            len(range(*partial_out_selection[i].indices(self.chunks[0] + 1)))
	        if i < len(partial_out_selection)
            else dim
            for i, dim in enumerate(self.chunks)
                ]
            cdata.read_part(start, nitems)
            chunk_partial = self._decode_chunk(
                cdata.buff,
                start=start,
	            nitems=nitems,
                expected_shape=expected_shape,
                )
            tmp[partial_out_selection] = chunk_partial
    out[out_selection] = tmp[chunk_selection]
    return
   ```

Now let's return to the BasicIndexer

- [BasicIndexer](https://github.com/zarr-developers/zarr-python/blob/44de0e4a017b8919bb5caba41eadcd67e18abdb9/zarr/indexing.py#L326)(selection,self). This appears to just do all the indexing between chunks and slices, but doesn't know about compression. Where is that coming from? It's coming from the chunk store. Let's go look there.
- The chunk_store is [provided](https://github.com/zarr-developers/zarr-python/blob/44de0e4a017b8919bb5caba41eadcd67e18abdb9/zarr/core.py#L167) "fully initialized with all configuration metadata fully specified and normalised". Since cdata is returned by a getitem on this, that's what we have to find.


Epiphany!
	- Zarr stores every chunk in it's own file, and the point of the store is to provide an index to each file, and then the entire file [is read](https://github.com/zarr-developers/zarr-python/blob/44de0e4a017b8919bb5caba41eadcd67e18abdb9/zarr/storage.py#L1026).
	- The point of kerchunk is to mimic that so we can read an entire chunk from somewhere ina  netcdf file, so kerchunk is the thing we need to think about.
	- The problem we have is that all the gubbins through the stack above is to deal with all the edge cases. How many edge cases do we care about?



