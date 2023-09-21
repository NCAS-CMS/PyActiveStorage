import os
import numpy as np
import pytest
import s3fs
import tempfile

from activestorage.active import Active
from activestorage.dummy_data import make_vanilla_ncdata
import activestorage.storage as st
from activestorage.reductionist import reduce_chunk as reductionist_reduce_chunk
from numpy.testing import assert_allclose, assert_array_equal
from pathlib import Path

from config_minio import *


def make_tempfile():
    """Make dummy data."""
    temp_folder = tempfile.mkdtemp()
    s3_testfile = os.path.join(temp_folder,
                               's3_test_bizarre.nc')  # Bryan likes this name
    print(f"S3 Test file is {s3_testfile}")
    if not os.path.exists(s3_testfile):
        make_vanilla_ncdata(filename=s3_testfile,
                            chunksize=(3, 3, 1), n=300)

    local_testfile = os.path.join(temp_folder,
                                  'local_test_bizarre.nc')  # Bryan again
    print(f"Local Test file is {local_testfile}")
    if not os.path.exists(local_testfile):
        make_vanilla_ncdata(filename=local_testfile)

    return s3_testfile, local_testfile


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


def create_files():
    """Create a file, keep it local, and put file in s3."""
    # make dummy data
    s3_testfile, local_testfile = make_tempfile()

    # put s3 dummy data onto S3. then rm from local
    object = os.path.basename(s3_testfile)
    bucket_file = upload_to_s3(S3_URL, S3_ACCESS_KEY, S3_SECRET_KEY,
                               S3_BUCKET, object, s3_testfile)

    s3_testfile_uri = os.path.join("s3://", bucket_file)
    print("S3 file uri", s3_testfile_uri)

    return s3_testfile_uri, local_testfile


@pytest.fixture
def s3_file():
    return create_files()[0]


@pytest.fixture
def local_file():
    return create_files()[1]


def test_Active(s3_file):
    """
    Test truly Active with an S3 file.
    """
    print("S3 file uri", s3_file)

    # run Active on s3 file
    active = Active(s3_file, "data", "s3")
    active.method = "mean"
    result1 = active[0:2, 4:6, 7:9]
    print(result1)


def test_no_Active(local_file):
    """
    Test pulling the data locally.
    """
    # run Active on local file
    active = Active(local_file, "data")
    active._version = 0
    active.method = "mean"
    active.components = True
    result2 = active[0:2, 4:6, 7:9]
    print(result2)
