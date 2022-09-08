import os
import itertools
import zarr
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


def load_netcdf_zarr_generic(fileloc, varname, build_dummy=True):
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


def zarr_chunks_info(ds):
    """Get offset and sizes of Zarr chunks."""
    # ds: Zarr dataset
    # Zarr chunking information; inspired from
    # zarr.convenience._copy(); convenience.py module l.964 (zarr=2.12.0)
    # https://zarr.readthedocs.io/en/stable/api/convenience.html#zarr.convenience.copy
    shape = ds.shape
    chunks = ds.chunks
    chunk_offsets = [range(0, s, c) for s, c in zip(shape, chunks)]
    print("Chunks index offsets:", [tuple(k) for k in chunk_offsets])
    print("Zarr Array total number of chunks:", len(list(itertools.product(*chunk_offsets))))
    offsets = []  # chunks keys zarray/i.j.k
    ch_sizes = []  # chunk sizes
    for offset in itertools.product(*chunk_offsets):
        offsets.append(offset)
        sel = tuple(slice(o, min(s, o + c))
                    for o, s, c in zip(offset, shape, chunks))
        # remember that a slice returns a tuple now
        # with data in slice: elem1, chunks selection: elem2, PCI: elem3
        islice = ds[sel][0]
        slice_size = islice.size * islice.dtype.itemsize
        ch_sizes.append(slice_size)

    # return a Chunks dict keyed by chunk coordinate indices (x.x.x)
    # with values the size of that chunk
    chunks_dict = dict()
    for offset, chunk_size in zip(offsets, ch_sizes):
        chunks_dict[offset] = chunk_size

    print(f"Chunks idx coordinate-size dictionary: {chunks_dict}")
    return chunks_dict


def slice_offset_size(fileloc, varname, selection):
    """
    Return a Zarr Array slice offset and size via PartialChunkIterator.

    Also return the indices of the chunks where the slice sits in, and
    those chunks' sizes."""
    # toggle compute_data
    # this turns on and off the actual data payload loading into Zarr store
    compute_data = False

    # load the netCDF file into an image of a Zarr array via
    # kerchunk HDF5->Zarr translation and a reference file system
    ds = load_netcdf_zarr_generic(fileloc, varname)

    data_selection, chunk_info = \
        zarr.core.Array.get_orthogonal_selection(ds, selection,
                                                 out=None, fields=None,
                                                 compute_data=compute_data)

    chunks, chunk_sel, PCI = chunk_info[0]

    offsets = []
    sizes = []
    for offset, size, _ in list(PCI):
        offsets.append(offset)
        sizes.append(size)

    print(f"Requested data selection (slices): {selection}")
    print(f"Master chunks: {chunks}, chunks selection: "
          f"{chunk_sel}, \nZarr PCI: {list(PCI)}\n")
    print(f"Slices offsets: {offsets}, \nslices sizes: {sizes}")

    chunks_dict = zarr_chunks_info(ds)
    chunks_coords = list(chunks_dict.keys())
    chunks_sizes = list(chunks_dict.values())
    # print("Chunks keys (i, j, k):", chunks_coords)
    # print("Chunks sizes:", chunks_sizes)
    
    # select chunks in selection
    selected_chunks = [chunks_coords[sl] for sl in selection]
    selected_chunk_sizes = [chunks_sizes[sl] for sl in selection]

    print("Chunks in selections:", selected_chunks)
    print("Chunks sizes in selections:", selected_chunk_sizes)

    return (ds, chunks, chunk_sel, offsets, sizes,
            selected_chunks, selected_chunk_sizes)
