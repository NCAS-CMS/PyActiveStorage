import os
import numpy as np
import pytest
import s3fs
import tempfile

from activestorage.active import Active
from activestorage.dummy_data import make_vanilla_ncdata
import activestorage.storage as st
from activestorage.s3 import reduce_chunk as s3_reduce_chunk
from numpy.testing import assert_array_equal

from config_minio import *


def make_tempfile():
    """Make dummy data."""
    temp_folder = tempfile.mkdtemp()
    s3_testfile = os.path.join(temp_folder,
                               's3_test_bizarre.nc')  # Bryan likes this name
    print(f"S3 Test file is {s3_testfile}")
    if not os.path.exists(s3_testfile):
        make_vanilla_ncdata(filename=s3_testfile)

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


def test_Active():
    """
    Shows what we expect an active example test to achieve and provides "the right answer" 
    """
    # make dummy data
    s3_testfile, local_testfile = make_tempfile()

    # put s3 dummy data onto S3. then rm from local
    object = os.path.basename(s3_testfile)
    bucket_file = upload_to_s3(S3_URL, S3_ACCESS_KEY, S3_SECRET_KEY,
                               S3_BUCKET, object, s3_testfile)
    os.remove(s3_testfile)
    s3_testfile_uri = os.path.join("s3://", bucket_file)
    print("S3 file uri", s3_testfile_uri)

    # run Active on s3 file
    active = Active(s3_testfile_uri, "data", "s3")
    active.method = "mean"
    result1 = active[0:2, 4:6, 7:9]
    print(result1)

    # run Active on local file
    active = Active(local_testfile, "data")
    active._version = 2
    active.method = "mean"
    active.components = True
    result2 = active[0:2, 4:6, 7:9]
    print(result2)

    assert_array_equal(result1, result2["sum"]/result2["n"])


def test_s3_reduce_chunk():
    """Unit test for s3_reduce_chunk."""
    rfile = "tests/test_data/cesm2_native.nc"
    offset = 2
    size = 128
    object = os.path.basename(rfile)

    # create bucket and upload to Minio's S3 bucket
    upload_to_s3(S3_URL, S3_ACCESS_KEY, S3_SECRET_KEY,
                 S3_BUCKET, object, rfile)
    
    # remove file during test session to be sure
    # workflow uses uploaded file to S3 bucket
    os.remove(rfile)

    # call s3_reduce_chunk
    tmp, count = s3_reduce_chunk(S3_ACTIVE_STORAGE_URL, S3_ACCESS_KEY,
                                 S3_SECRET_KEY, S3_URL, S3_BUCKET,
                                 object, offset, size,
                                 None, None, [],
                                 np.dtype("int32"), (32, ),
                                 "C", [slice(0, 2, 1), ],
                                 "min")
    assert tmp == 134351386
    assert count == None
