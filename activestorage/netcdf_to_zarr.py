import os
import numpy as np
import zarr
import ujson
import fsspec
import s3fs
import tempfile

from activestorage.config import *
from kerchunk.hdf import SingleHdf5ToZarr

import time
import h5py


def _correct_compressor_and_filename(content, varname, bryan_bucket=False):
    """
    Correct the compressor type as it comes out of Kerchunk.
    Also correct file name as Kerchnk now prefixes it with "s3://"
    and for special buckets like Bryan's bnl the correct file is bnl/file.nc
    not s3://bnl/file.nc
    """
    tc1 = time.time()
    new_content = content.copy()
    try:
        new_zarray =  ujson.loads(new_content['refs'][f"{varname}/.zarray"])
        group = False
    except KeyError:
        new_zarray =  ujson.loads(new_content['refs'][f"{varname} /{varname}/.zarray"])
        group = True

    # re-add the correct compressor if it's in the "filters" list
    if new_zarray["compressor"] is None:
        for zfilter in new_zarray["filters"]:
            if zfilter["id"] == "zlib":
                new_zarray["compressor"] = zfilter
                new_zarray["filters"].remove(zfilter)

    if not group:
        new_content['refs'][f"{varname}/.zarray"] = ujson.dumps(new_zarray)
    else:
        new_content['refs'][f"{varname} /{varname}/.zarray"] = ujson.dumps(new_zarray)

    # FIXME TODO this is an absolute nightmate: the type of bucket on UOR ACES
    # this is a HACK and it works only with the crazy Bryan S3 bucket "bnl/file.nc"
    # the problem: filename gets written to JSON as "s3://bnl/file.nc" but Reductionist doesn't
    # find it since it needs url=bnl/file.nc, with endpoint URL being extracted from the
    # endpoint_url of storage_options. BAH!
    if bryan_bucket:
        for key in new_content['refs'].keys():
            if varname in key and isinstance(new_content['refs'][key], list) and "s3://" in new_content['refs'][key][0]:
                new_content['refs'][key][0] = new_content['refs'][key][0].replace("s3://", "")

    tc2 = time.time()
    print("Time to manipulate Kerchunk Zarr output", tc2 - tc1)
    return new_content


def gen_json(file_url, varname, outf, storage_type, storage_options):
    """Generate a json file that contains the kerchunk-ed data for Zarr."""
    # S3 configuration presets
    if storage_type == "s3" and storage_options is None:
        fs = s3fs.S3FileSystem(key=S3_ACCESS_KEY,
                               secret=S3_SECRET_KEY,
                               client_kwargs={'endpoint_url': S3_URL},
                               default_fill_cache=False,
                               default_cache_type="first"
        )
        fs2 = fsspec.filesystem('')
        with fs.open(file_url, 'rb') as s3file:
            h5chunks = SingleHdf5ToZarr(s3file, file_url,
                                        inline_threshold=0)
            bryan_bucket = False
            if "bnl" in file_url:
                bryan_bucket = True
            with fs2.open(outf, 'wb') as f:
                content = h5chunks.translate()
                content = _correct_compressor_and_filename(content,
                                                           varname,
                                                           bryan_bucket=bryan_bucket)
                f.write(ujson.dumps(content).encode())

    # S3 passed-in configuration
    elif storage_type == "s3" and storage_options is not None:
        storage_options = storage_options.copy()
        storage_options['default_fill_cache'] = False
        storage_options['default_cache_type'] = "first"
        fs = s3fs.S3FileSystem(**storage_options)
        fs2 = fsspec.filesystem('')
        tk1 = time.time()
        with fs.open(file_url, 'rb') as s3file:
            s3file = h5py.File(s3file, mode="w")
            if isinstance(s3file[varname], h5py.Dataset):
                print("Looking only at a single Dataset", s3file[varname])
                s3file.create_group(varname + " ")
                s3file[varname + " "][varname] = s3file[varname]
                s3file = s3file[varname + " "]
            elif isinstance(s3file[varname], h5py.Group):
                print("Looking only at a single Group", s3file[varname])
                s3file = s3file[varname]
            # Kerchunk wants the correct file name in S3 format
            if not file_url.startswith("s3://"):
                file_url = "s3://" + file_url
            bryan_bucket = False
            if "bnl" in file_url:
                bryan_bucket = True
            h5chunks = SingleHdf5ToZarr(s3file, file_url,
                                        inline_threshold=0,
                                        storage_options=storage_options)
            tk2 = time.time()
            with fs2.open(outf, 'wb') as f:
                content = h5chunks.translate()
                content = _correct_compressor_and_filename(content,
                                                           varname,
                                                           bryan_bucket=bryan_bucket)
                f.write(ujson.dumps(content).encode())
            tk3 = time.time()
            print("Time to Kerchunk and write JSON file", tk3 - tk2)
    # not S3
    else:
        fs = fsspec.filesystem('')
        with fs.open(file_url, 'rb') as local_file:
            try:
                h5chunks = SingleHdf5ToZarr(local_file, file_url,
                                            inline_threshold=0)
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
            with fs.open(outf, 'wb') as f:
                content = h5chunks.translate()
                f.write(ujson.dumps(content).encode())

    # account for both Group and Dataset
    try:
        zarray =  ujson.loads(content['refs'][f"{varname}/.zarray"])
        zattrs =  ujson.loads(content['refs'][f"{varname}/.zattrs"])
    except KeyError:
        zarray =  ujson.loads(content['refs'][f"{varname} /{varname}/.zarray"])
        zattrs =  ujson.loads(content['refs'][f"{varname} /{varname}/.zattrs"])

    return outf, zarray, zattrs


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

    not_group = False
    try:
        zarr_array = getattr(zarr_group, varname + " ")
    except AttributeError:
        not_group = True
        pass
    if not_group:
        try:
            zarr_array = getattr(zarr_group, varname)
        except AttributeError:
            print(f"Zarr Group does not contain variable {varname}. "
                  f"Zarr Group info: {zarr_group.info}")
            raise
    
    return zarr_array


def load_netcdf_zarr_generic(fileloc, varname, storage_type, storage_options, build_dummy=True):
    """Pass a netCDF4 file to be shaped as Zarr file by kerchunk."""
    print(f"Storage type {storage_type}")

    # Write the Zarr group JSON to a temporary file.
    with tempfile.NamedTemporaryFile() as out_json:
        _, zarray, zattrs = gen_json(fileloc,
                                     varname,
                                     out_json.name,
                                     storage_type,
                                     storage_options)

        # open this monster
        print(f"Attempting to open and convert {fileloc}.")
        ref_ds = open_zarr_group(out_json.name, varname)

    return ref_ds, zarray, zattrs
