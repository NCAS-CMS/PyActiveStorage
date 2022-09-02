import os
import zarr
from pathlib import Path
import ujson
import fsspec

from kerchunk.hdf import SingleHdf5ToZarr


def gen_json(file_url, fs, fs2, **so):
    """Generate a json file that contains the kerchunk-ed data for Zarr."""
    # set some name for the output json file
    fname = os.path.splitext(file_url)[0]
    variable = file_url.split('/')[-1].split('.')[0]
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

    Open a reference filesystem json file into a Zarr Group
    then extract the Zarr Array you need.
    """
    fs = fsspec.filesystem("reference", fo=out_json)
    mapper = fs.get_mapper("")
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
    fs2 = fsspec.filesystem('')  # local file system to save final jsons to
    out_json = gen_json(fileloc, fs, fs2)

    # open this monster
    ref_ds = open_zarr_group(out_json)

    return ref_ds


def slice_offset_size(fileloc, varname, selection):
    """Return a Zarr Array slice offset and size via PartialChunkIterator."""
    # toggle compute_data
    # this turns on and off the actual data payload loading into Zarr store
    compute_data = False

    # load the Zarr image into an actual Array
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

    return ds, chunks, chunk_sel, offsets, sizes
