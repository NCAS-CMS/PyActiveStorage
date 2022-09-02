import os
import zarr

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


def load_netcdf_zarr_generic(fileloc, varname):
    """Pass a netCDF4 file to be shaped as Zarr file by kerchunk."""
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

    return ref_ds


def slice_offset_size(fileloc, varname, selection):
    """Return a Zarr Array slice offset and size via PartialChunkIterator."""
    # toggle compute_data
    # this turns on and off the actual data payload loading into Zarr store
    compute_data = False

    # load the Zarr image
    ds = load_netcdf_zarr_generic(fileloc, varname)

    data_selection, chunk_info = \
        zarr.core.Array.get_orthogonal_selection(ds, selection,
                                                 out=None, fields=None,
                                                 compute_data=compute_data)

    # integrate it in the Zarr mechanism

    print("Running the PCI wrapper integrated in Zarr.Array:")
    print(f"Data selection {selection}")
    chunks, chunk_sel, PCI = chunk_info[0]
    print(f"Chunks {chunks}, chunk selection "
          f"{chunk_sel}, PCI {list(PCI)}\n")

    offsets = []
    sizes = []
    for offset, size, _ in list(PCI):
        offsets.append(offset)
        sizes.append(size)

    return ds, chunks, chunk_sel, offsets, sizes
