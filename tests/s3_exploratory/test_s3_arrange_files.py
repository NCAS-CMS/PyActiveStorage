import os

import fsspec
import numpy as np
import pytest
import s3fs
import tempfile

from activestorage.active import Active
from activestorage.dummy_data import make_vanilla_ncdata
import activestorage.storage as st
from activestorage.reductionist import reduce_chunk as reductionist_reduce_chunk
from activestorage.netcdf_to_zarr import gen_json
from kerchunk.hdf import SingleHdf5ToZarr

from numpy.testing import assert_allclose, assert_array_equal
from pathlib import Path

from config_minio import *


@pytest.fixture
def test_spec():
    CHUNKS = (10, 10, 10)  # (3, 3, 1)
    NSIZE = 150

    return CHUNKS, NSIZE


@pytest.fixture
def test_data_path():
    """Path to test data."""
    return Path(__file__).resolve().parent / 'test_data'


def make_s3_file(test_spec):
    """Make dummy data."""
    temp_folder = tempfile.mkdtemp()
    CHUNKS, NSIZE = test_spec
    s3_testfile = os.path.join(temp_folder,
                               's3_test_bizarre_large.nc')
    print(f"S3 Test file is {s3_testfile}")
    if not os.path.exists(s3_testfile):
        make_vanilla_ncdata(filename=s3_testfile,
                            chunksize=CHUNKS, n=NSIZE)

    return s3_testfile


def make_local_file(test_data_path, test_spec):
    """Create a vanilla nc file and store in test_data dir here."""
    local_testfile = os.path.join(test_data_path,
                                  'test_bizarre.nc')
    CHUNKS, NSIZE = test_spec

    print(f"Local Test file is {local_testfile}")
    if not os.path.exists(local_testfile):
        make_vanilla_ncdata(filename=local_testfile,
                            chunksize=CHUNKS, n=NSIZE)

    return local_testfile


def upload_to_s3(server, username, password, bucket, object, rfile):
    """Upload a file to an S3 object store."""
    s3_fs = s3fs.S3FileSystem(key=username, secret=password, client_kwargs={'endpoint_url': server})
    # Make sure s3 bucket exists
    try:
        s3_fs.mkdir(bucket)
    except FileExistsError:
        pass

    s3_fs.put_file(rfile, os.path.join(bucket, object))

    return os.path.join(bucket, object)


def test_create_files(test_data_path, test_spec):
    """Create a file, keep it local, and put file in s3."""
    # make dummy data
    s3_testfile  = make_s3_file(test_spec)
    local_testfile = make_local_file(test_data_path, test_spec)

    # put s3 dummy data onto S3
    object = os.path.basename(s3_testfile)
    bucket_file = upload_to_s3(S3_URL, S3_ACCESS_KEY, S3_SECRET_KEY,
                               S3_BUCKET, object, s3_testfile)

    s3_testfile_uri = os.path.join("s3://", bucket_file)

    print("S3 file uri", s3_testfile_uri)
    print("Local file uri", local_testfile)
