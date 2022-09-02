import os
import zarr
from pathlib import Path
import ujson
import fsspec

from kerchunk.hdf import SingleHdf5ToZarr


def extract_dict(strarg):
    """Extract a dict from a string-formatted dict."""
    # FIXME this is a bit silly
    return_dict = dict()
    if "shape" in strarg:
        items = strarg.split('"shape"')
        val = items[1].split(":")[1].strip()
        val = val.split('"zarr_format"')[0].split(',')[0:3]
        val = (int(val[0].lstrip('[').strip()),
               int(val[1].strip()),
               int(val[2].split("]")[0].strip()))
        return_dict["shape"] = val
    if "chunks" in strarg:
        items = strarg.split('"compressor"')
        val = items[0].split(":")[1].strip()
        val = [v.strip() for v in val.split('\n')]
        val = (int(val[1].split(',')[0]),
               int(val[2].split(',')[0]),
               int(val[3]))
        return_dict["chunks"] = val

    return return_dict


def gen_json(file_url, fs, fs2, **so):
    with fs.open(file_url, **so) as infile:
        h5chunks = SingleHdf5ToZarr(infile, file_url, inline_threshold=300)
        # inline threshold adjusts the Size below which binary blocks are
        # included directly in the output
        # a higher inline threshold can result in a larger json file but
        # faster loading time
        variable = file_url.split('/')[-1].split('.')[0]
        fname = os.path.splitext(file_url)[0]
        outf = f'{fname}_{variable}.json' # vanilla file name
        with fs2.open(outf, 'wb') as f:
            f.write(ujson.dumps(h5chunks.translate()).encode())

    return outf


def load_netcdf_zarr_generic(fileloc, varname, build_dummy=True):
    """Pass a netCDF4 file to be shaped as Zarr file by kerchunk."""
    # really, what we should do is pass the Kerchunk reference dict
    # straight to Zarr to get its info from there. and not create any
    # dummy Zarr Array

    # for testing: use a dummy Zarr Array built ad-hoc
    if build_dummy:
        # invoke the Kerchunk Lorde
        ds = SingleHdf5ToZarr(fileloc).translate()
        varkey = varname + '/.zarray'
        if not varkey in ds['refs']:
            varkey = 'data/.zarray'
            chunks = extract_dict(ds['refs'][varkey])["chunks"]
        chunks = extract_dict(ds['refs'][varkey])["shape"]

        store, chunk_store = dict(), dict()
        ref_ds = zarr.create(chunks, chunks=chunks, compressor=None,
                             dtype='f8', order='C',
                             store=store, chunk_store=chunk_store)

    else:
        so = dict(mode='rb', anon=True, default_fill_cache=False,
                  default_cache_type='first') # args to fs.open()
        # default_fill_cache=False avoids caching data in between
        # file chunks to lower memory usage
        fs = fsspec.filesystem('')  # local, for S3: ('s3', anon=True)
        fs2 = fsspec.filesystem('')  # local file system to save final jsons to
        out_json = gen_json(fileloc, fs, fs2)

        # open this monster
        fs = fsspec.filesystem("reference", fo=out_json)
        m = fs.get_mapper("")
        ref_ds = zarr.open_group(m)
        print("XXX", ref_ds.info)

    return ref_ds


def slice_offset_size(fileloc, varname, selection):
    """Return a Zarr Array slice offset and size via PartialChunkIterator."""
    # toggle compute_data
    # this turns on and off the actual data payload loading into Zarr store
    compute_data = False

    # load the Zarr image
    ds = load_netcdf_zarr_generic(fileloc, varname, build_dummy=False)

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
