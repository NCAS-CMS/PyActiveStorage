import os
import numpy as np
import itertools
import zarr
from active_tools import make_an_array_instance_active

from pathlib import Path
import ujson
import fsspec

from kerchunk.hdf import SingleHdf5ToZarr


def gen_json(file_url, fs, fs2, **so):
    """Generate a json file that contains the kerchunk-ed data for Zarr."""
    # set some name for the output json file
    fname = os.path.splitext(file_url)[0]
    variable = file_url.split('/')[-1].split('.')[0]  # just as an example
    outf = f'{fname}_{variable}.json' # vanilla file name

    # write it out if it's not there
    if not os.path.isfile(outf):
        with fs.open(file_url, **so) as infile:
            h5chunks = SingleHdf5ToZarr(infile, file_url, inline_threshold=300)
            # inline threshold adjusts the Size below which binary blocks are
            # included directly in the output
            # a higher inline threshold can result in a larger json file but
            # faster loading time
            fname = os.path.splitext(file_url)[0]
            outf = f'{fname}_{variable}.json' # vanilla file name
            with fs2.open(outf, 'wb') as f:
                f.write(ujson.dumps(h5chunks.translate()).encode())

    return outf


def open_zarr_group(out_json):
    """
    Do the magic opening

    Open a json file read and saved by the reference file system
    into a Zarr Group, then extract the Zarr Array you need.
    That Array is in the 'data' attribute.
    """
    fs = fsspec.filesystem("reference", fo=out_json)
    mapper = fs.get_mapper("")  # local FS mapper
    zarr_group = zarr.open_group(mapper)
    print("Zarr group info:", zarr_group.info)
    zarr_array = zarr_group.data
    print("Zarr array info:",  zarr_array.info)

    return zarr_array


def load_netcdf_zarr_generic(fileloc, varname=None, build_dummy=True):
    """Pass a netCDF4 file to be shaped as Zarr file by kerchunk."""
    so = dict(mode='rb', anon=True, default_fill_cache=False,
              default_cache_type='first') # args to fs.open()
    # default_fill_cache=False avoids caching data in between
    # file chunks to lower memory usage
    fs = fsspec.filesystem('')  # local, for S3: ('s3', anon=True)
    fs2 = fsspec.filesystem('')  # local file system to save final json to
    out_json = gen_json(fileloc, fs, fs2)

    # open this monster
    ref_ds = open_zarr_group(out_json)

    return ref_ds


def slice_offset_size(fileloc, varname, selection):
    """
    Return a Zarr Array slice offset and size via PartialChunkIterator.

    Also return the indices of the chunks where the slice sits in, and
    those chunks' sizes."""
    

    # load the netCDF file into an image of a Zarr array via
    # kerchunk HDF5->Zarr translation and a reference file system
    ds = load_netcdf_zarr_generic(fileloc, varname)
    ds = make_an_array_instance_active(ds)

    data_selection, chunk_info, chunk_coords = ds.get_orthogonal_selection(selection,
                                                 out=None, fields=None)

    chunks, chunk_sel, PCI = chunk_info[0]

    # get offsets and sizes from PCI
    offsets = []
    sizes = []
    for offset, size, _ in list(PCI):
        offsets.append(offset)
        sizes.append(size)

    # get chunks info from chunk store
    chunk_store = ds.chunk_store
    chunk_coords_formatted = []
    for ch_coord in chunk_coords:
        new_key = "data/" + ".".join([str(ch_coord[0]),
                                      str(ch_coord[1]),
                                      str(ch_coord[2])])
        chunk_coords_formatted.append(new_key)

    # decode bytes from chunks
    chunks_with_data = [ds._decode_chunk(chunk_store[k]) for k in chunk_coords_formatted]
    flat_chunks_with_data = np.ndarray.flatten(np.array(chunks_with_data))

    print("Data selection shape as returned by Zarr (out array shape):", data_selection.shape)
    print(f"Requested data selection (slices): {selection}")
    print(f"Master chunks: {chunks}")
    print(f"Data coordinates inside each chunk that overlaps selection: {chunk_sel}")
    print(f"Zarr PartialChunkIterator (PCI): {list(PCI)}")
    print(f"Chunks (containing all data in selection) coordinates: {chunk_coords}")
    print(f"Data from the Chunks (containing all data in selection): {flat_chunks_with_data}")
    print(f"Number of offset elements where selected data starts per chunk: {offsets}")
    print(f"Number of elements in data inside chunk per chunk: {sizes}")
    chunks_dict = {}
    for (i, k), f in zip(enumerate(chunk_coords_formatted), chunk_coords):
        flat_decoded = np.ndarray.flatten(ds._decode_chunk(chunk_store[k]))
        print(f"Flattened chunk {i} data: {flat_decoded}")
        selection_in_chunk = []
        # NB: very important to remember that each start in "offsets" is to be
        # used for each chunk; it's not one start from "offsets" is per chunk
        for j, k in zip(offsets, sizes):
            print(f"Local start of selection data for chunk {i}: {flat_decoded[j]}")
            partial_data = flat_decoded[j:j+k]
            selection_in_chunk.extend(partial_data)
            print(f"Data comprised in segment in chunk {i}: {partial_data}")
        print(f"Selection of data comprised in chunk {i}: {selection_in_chunk}")
        chunks_dict[f] = selection_in_chunk
    print(f"Chunks and their selected data: {chunks_dict}")


    return (ds, chunks, chunk_sel, offsets, sizes,
            chunk_coords, chunks_dict)
