import os
import numpy as np
import zarr
import ujson
import fsspec
import s3fs

from kerchunk.hdf import SingleHdf5ToZarr


def gen_json(file_url, fs, fs2, varname, **so):
    """Generate a json file that contains the kerchunk-ed data for Zarr."""
    # set some name for the output json file
    fname = os.path.splitext(file_url)[0]
    if "s3:" in fname:
        fname = os.path.basename(fname)
    outf = f'{fname}_{varname}.json' # vanilla file name

    # write it out if it's not there
    if not os.path.isfile(outf):
        with fs.open(file_url, **so) as infile:
            # FIXME need to disentangle HDF5 errors if not OSError (most are)
            try:
                h5chunks = SingleHdf5ToZarr(infile, file_url, inline_threshold=0)
            except OSError as exc:
                raiser_1 = f"Unable to open file {file_url}. "
                raiser_2 = "Check if file is netCDF3 or netCDF-classic"
                print(raiser_1 + raiser_2)
                raise exc

            # inline threshold adjusts the Size below which binary blocks are
            # included directly in the output
            # a higher inline threshold can result in a larger json file but
            # faster loading time
            # for active storage, we don't want anything inline
#            fname = os.path.splitext(file_url)[0]
#            outf = f'{fname}_{varname}.json' # vanilla file name
            with fs2.open(outf, 'wb') as f:
                f.write(ujson.dumps(h5chunks.translate()).encode())

    return outf


def open_zarr_group(out_json, varname):
    """
    Do the magic opening

    Open a json file read and saved by the reference file system
    into a Zarr Group, then extract the Zarr Array you need.
    That Array is in the 'data' attribute.
    """
    fs = fsspec.filesystem("reference", fo=out_json)
    mapper = fs.get_mapper("")  # local FS mapper
    #mapper.fs.reference has the kerchunk mapping, how does this propagate into the Zarr array?
    zarr_group = zarr.open_group(mapper)
    try:
        zarr_array = getattr(zarr_group, varname)
    except AttributeError as attrerr:
        print(f"Zarr Group does not contain variable {varname}. "
              f"Zarr Group info: {zarr_group.info}")
        raise attrerr
    #print("Zarr array info:",  zarr_array.info)

    return zarr_array


def load_netcdf_zarr_generic(fileloc, varname, storage_type, build_dummy=True):
    """Pass a netCDF4 file to be shaped as Zarr file by kerchunk."""
    print(f"Storage type {storage_type}")
    object_filesystems = ["s3"]
    if storage_type not in object_filesystems:
        so = dict(mode='rb', anon=True, default_fill_cache=False,
                  default_cache_type='first') # args to fs.open()
        # default_fill_cache=False avoids caching data in between
        # file chunks to lower memory usage
        fs = fsspec.filesystem('')
    elif storage_type == "s3":
        # TODO of course s3 connection params must be off the config
        fs = s3fs.S3FileSystem(key="minioadmin",
                               secret="minioadmin",
                               client_kwargs={'endpoint_url': "http://localhost:9000"})
        so = {}

    fs2 = fsspec.filesystem('')  # local file system to save final json to
    out_json = gen_json(fileloc, fs, fs2, varname)

    # open this monster
    print(f"Attempting to open and convert {fileloc}.")
    ref_ds = open_zarr_group(out_json, varname)

    return ref_ds
