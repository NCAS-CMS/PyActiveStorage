import os
import numpy as np
import zarr
import ujson
import fsspec
import s3fs
import tempfile

from activestorage.config import *
from kerchunk.hdf import SingleHdf5ToZarr


def gen_json(file_url, varname, outf, storage_type, storage_options):
    """Generate a json file that contains the kerchunk-ed data for Zarr."""
    # S3 configuration presets
    if storage_type == "s3" and not storage_options:
        fs = s3fs.S3FileSystem(key=S3_ACCESS_KEY,
                               secret=S3_SECRET_KEY,
                               client_kwargs={'endpoint_url': S3_URL},
                               default_fill_cache=False,
                               default_cache_type="none"
        )
        fs2 = fsspec.filesystem('')
        with fs.open(file_url, 'rb') as s3file:
            h5chunks = SingleHdf5ToZarr(s3file, file_url,
                                        inline_threshold=0)
            with fs2.open(outf, 'wb') as f:
                content = h5chunks.translate()
                f.write(ujson.dumps(content).encode())

    # S3 passed-in configuration
    elif storage_type == "s3" and storage_options:
        storage_options = storage_options.copy()
        storage_options['default_fill_cache'] = False
        storage_options['default_cache_type'] = "none"
        fs = s3fs.S3FileSystem(**storage_options)
        fs2 = fsspec.filesystem('')
        with fs.open(file_url, 'rb') as s3file:
            h5chunks = SingleHdf5ToZarr(s3file, file_url,
                                        inline_threshold=0)
            with fs2.open(outf, 'wb') as f:
                content = h5chunks.translate()
                f.write(ujson.dumps(content).encode())
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

    zarray =  ujson.loads(content['refs'][f"{varname}/.zarray"])
    zattrs =  ujson.loads(content['refs'][f"{varname}/.zattrs"])
                
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
