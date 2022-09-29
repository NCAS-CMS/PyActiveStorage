import os
import zarr

from kerchunk.hdf import SingleHdf5ToZarr


# example valid netCDF4 file
# can be loaded through h5py
cmip6_test_file = r"/home/valeriu/climate_data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-HR/historical/r1i1p1f1/Omon/sos/gn/v20190710/sos_Omon_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_198001-198412.nc"

def build_zarr_dataset():
    """Create a zarr array and save it."""
    dsize = (60, 404, 802)
    dchunks = (12, 80, 160)
    dvalue = 42.
    store = zarr.DirectoryStore('data/array.zarr')
    z = zarr.zeros(dsize, chunks=dchunks, store=store, overwrite=True)
    z[...] = dvalue
    zarr.save_array("example.zarr", z, compressor=None)


def load_zarr_native():
    """Load the Zarr file we have built, actual/native format."""
    zarr_dir = "./example.zarr"
    if not os.path.isdir(zarr_dir):
        build_zarr_dataset()
    ds = zarr.open("./example.zarr")

    return ds


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


def load_netcdf_zarr():
    """Pass a netCDF4 file to be shaped as Zarr file."""
    ds = SingleHdf5ToZarr(cmip6_test_file).translate()
    chunks = extract_dict(ds['refs']['sos/.zarray'])["shape"]

    store, chunk_store = dict(), dict()
    ref_ds = zarr.create(chunks, chunks=chunks, compressor=None,
                         dtype='f8', order='C',
                         store=store, chunk_store=chunk_store)

    return ref_ds


def load_netcdf_zarr_generic(fileloc, varname):
    """Pass a netCDF4 file to be shaped as Zarr file."""
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


def PCI_wrapper_standalone(chunk_selection, native_zarr=False):
    """Build a wrapper of PartialChunkIterator and run it as standalone."""
    from zarr import core
    if not native_zarr:
        ds = load_netcdf_zarr()
    else:
        ds = load_zarr_native()
    # functional arguments example
    # chunk_selection: (slice(0, 2, 1), slice(0, 4, 1), slice(0, 2, 1))
    pc = core.Array._process_chunk_V(ds, chunk_selection)

    return pc


def PCI_wrapper(selection, compute_data, native_zarr=False):
    """Use all the Zarr ins."""
    from zarr import core
    if not native_zarr:
        ds = load_netcdf_zarr()
    else:
        ds = load_zarr_native()
    pc = core.Array.get_orthogonal_selection(ds, selection,
                                             out=None, fields=None,
                                             compute_data=compute_data)
    return pc


def main():
    """Extract the needed info straight from Zarr."""
    # pass a chunk_selection and run the standalone function
    chunk_selection = (slice(0, 2, 1), slice(0, 4, 1), slice(0, 2, 1))
    chunks, chunk_sel, PCI = PCI_wrapper_standalone(chunk_selection,
                                                    native_zarr=False)
    print("Running the PCI wrapper as standalone:")
    print(f"Chunks {chunks}, chunk selection {chunk_sel}, PCI {list(PCI)}\n")

    # toggle compute_data
    # this turns on and off the actual data loading into store
    compute_data = False

    # integrate it in the Zarr mechanism
    data_selection, chunk_info = PCI_wrapper(chunk_selection,
                                             compute_data,
                                             native_zarr=False)
    print("Running the PCI wrapper integrated in Zarr.Array:")
    print(f"Data selection {data_selection}")
    chunks_full, chunk_sel_full, PCI_full = chunk_info[0]
    print(f"Chunks {chunks}, chunk selection {chunk_sel_full}, PCI {list(PCI_full)}\n")

    # cross-check
    assert chunk_sel == chunk_sel_full
    assert list(PCI) == list(PCI_full)
        


if __name__ == '__main__':
    main()
