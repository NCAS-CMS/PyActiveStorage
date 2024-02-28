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


def _correct_compressor(content, varname):
    """Correct the compressor type as it comes out of Kerchunk."""
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
            with fs2.open(outf, 'wb') as f:
                content = h5chunks.translate()
                f.write(ujson.dumps(content).encode())

    # S3 passed-in configuration
#### this implementation works with a minimally changed kerchunk/hdf.py #####
#############################################################################
###    def __init__(
###        self,
###        h5f: "BinaryIO | str",
###        url: str = None,
###        spec=1,
###        inline_threshold=500,
###        storage_options=None,
###        error="warn",
###        vlen_encode="embed",
###    ):
###
###        # Open HDF5 file in read mode...
###        lggr.debug(f"HDF5 file: {h5f}")
###
###        if isinstance(h5f, str):
###            fs, path = fsspec.core.url_to_fs(h5f, **(storage_options or {}))
###            self.input_file = fs.open(path, "rb")
###            url = h5f
###            self._h5f = h5py.File(self.input_file, mode="r")
###        elif isinstance(h5f, io.IOBase):
###            self.input_file = h5f
###            self._h5f = h5py.File(self.input_file, mode="r")
###        elif isinstance(h5f, (h5py.File, h5py.Group)):
###            self._h5f = h5f
###
###        self.spec = spec
###        self.inline = inline_threshold
###        if vlen_encode not in ["embed", "null", "leave", "encode"]:
###            raise NotImplementedError
###        self.vlen = vlen_encode
###
###        self.store = {}
###        self._zroot = zarr.group(store=self.store, overwrite=True)
###
###        self._uri = url
###        self.error = error
###        lggr.debug(f"HDF5 file URI: {self._uri}")
###############################################################################
    elif storage_type == "s3" and storage_options is not None:
        storage_options = storage_options.copy()
        storage_options['default_fill_cache'] = False
        storage_options['default_cache_type'] = "first"
        fs = s3fs.S3FileSystem(**storage_options)
        fs2 = fsspec.filesystem('')
        tk1 = time.time()
        print("Storage options dict", storage_options)
        print("File url", file_url)
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
            if not file_url.startswith("s3://"):
                file_url = "s3://" + file_url
            print("File_url to Kerchunk", file_url)
            h5chunks = SingleHdf5ToZarr(s3file, file_url,
                                        inline_threshold=0,
                                        storage_options=storage_options)
            tk2 = time.time()
            print("Time to set up Kerchunk", tk2 - tk1)
            with fs2.open(outf, 'wb') as f:
                content = h5chunks.translate()
                content = _correct_compressor(content, varname)
                f.write(ujson.dumps(content).encode())
            tk3 = time.time()
            print("Time to Translate and Dump Kerchunks to json file", tk3 - tk2)
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
   
    try:
        zarr_array = getattr(zarr_group, varname)
    except AttributeError as attrerr:
        print(f"Zarr Group does not contain variable {varname}. "
              f"Zarr Group info: {zarr_group.info}")
        raise attrerr
    #print("Zarr array info:",  zarr_array.info)
    
    return zarr_array


def load_netcdf_zarr_generic(fileloc, varname, storage_type, storage_options, build_dummy=True):
    """Pass a netCDF4 file to be shaped as Zarr file by kerchunk."""
    print(f"Storage type {storage_type}")

    # Write the Zarr group JSON to a temporary file.
    save_json = "test_file.json"
    # with tempfile.NamedTemporaryFile() as out_json:
    with open(save_json, "wb") as out_json:
        _, zarray, zattrs = gen_json(fileloc,
                                     varname,
                                     out_json.name,
                                     storage_type,
                                     storage_options)

        # open this monster
        print(f"Attempting to open and convert {fileloc}.")
        try:
            ref_ds = open_zarr_group(out_json.name, varname)
        except AttributeError:
            ref_ds = open_zarr_group(out_json.name, varname + " ")

    return ref_ds, zarray, zattrs


#d = {'version': 1,
# 'refs': {
#     '.zgroup': '{"zarr_format":2}',
#     '.zattrs': '{"Conventions":"CF-1.6","access-list":"grenvillelister simonwilson jeffcole","awarning":"**** THIS SUITE WILL ARCHIVE NON-DUPLEXED DATA TO MOOSE. FOR CRITICAL MODEL RUNS SWITCH TO DUPLEXED IN: postproc --> Post Processing - common settings --> Moose Archiving --> non_duplexed_set. Follow guidance in http:\\/\\/www-twiki\\/Main\\/MassNonDuplexPolicy","branch-date":"1950-01-01","calendar":"360_day","code-version":"UM 11.6, NEMO vn3.6","creation_time":"2022-10-28 12:28","decription":"Initialised from EN4 climatology","description":"Copy of u-ar696\\/trunk@77470","email":"r.k.schieman@reading.ac.uk","end-date":"2015-01-01","experiment-id":"historical","forcing":"AA,BC,CO2","forcing-info":"blah, blah, blah","institution":"NCAS","macro-parent-experiment-id":"historical","macro-parent-experiment-mip":"CMIP","macro-parent-variant-id":"r1i1p1f3","model-id":"HadGEM3-CG31-MM","name":"\\/work\\/n02\\/n02\\/grenvill\\/cylc-run\\/u-cn134\\/share\\/cycle\\/19500101T0000Z\\/3h_","owner":"rosalynhatcher","project":"Coupled Climate","timeStamp":"2022-Oct-28 12:20:33 GMT","title":"[CANARI] GC3.1 N216 ORCA025 UM11.6","uuid":"51e5ef20-d376-4aa6-938e-4c242886b7b1"}',
#     'lat/.zarray': '{"chunks":[324],"compressor":{"id":"zlib","level":1},"dtype":"<f4","fill_value":null,"filters":[{"elementsize":4,"id":"shuffle"}],"order":"C","shape":[324],"zarr_format":2}', 'lat/.zattrs': '{"_ARRAY_DIMENSIONS":["lat"],"axis":"Y","long_name":"Latitude","standard_name":"latitude","units":"degrees_north"}',
#     'lat/0': ['/home/david/Downloads/3h__19500101-19500110.nc', 26477, 560],
#     'lon/.zarray': '{"chunks":[432],"compressor":{"id":"zlib","level":1},"dtype":"<f4","fill_value":null,"filters":[{"elementsize":4,"id":"shuffle"}],"order":"C","shape":[432],"zarr_format":2}',
#     'lon/.zattrs': '{"_ARRAY_DIMENSIONS":["lon"],"axis":"X","long_name":"Longitude","standard_name":"longitude","units":"degrees_east"}',
#     'lon/0': ['/home/david/Downloads/3h__19500101-19500110.nc', 27037, 556],
#     'm01s00i507_10/.zarray': '{"chunks":[1,324,432],"compressor":{"id":"zlib","level":1},"dtype":"<f4","fill_value":-1073741824.0,"filters":[{"elementsize":4,"id":"shuffle"}],"order":"C","shape":[80,324,432],"zarr_format":2}',
#     'm01s00i507_10/.zattrs': '{"_ARRAY_DIMENSIONS":["time_counter","lat","lon"],"cell_methods":"time: mean (interval: 900 s)","coordinates":"time_centered","interval_offset":"0ts","interval_operation":"900 s","interval_write":"3 h","long_name":"OPEN SEA SURFACE TEMP AFTER TIMESTEP","missing_value":-1073741824.0,"online_operation":"average","standard_name":"surface_temperature","units":"K"}',
#     }}
