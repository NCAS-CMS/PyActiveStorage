import os
import numpy as np
import pytest
import s3fs
import tempfile

from activestorage.active import Active
from activestorage.dummy_data import make_vanilla_ncdata
import activestorage.storage as st
from activestorage.reductionist import reduce_chunk as reductionist_reduce_chunk
from activestorage.reductionist import get_session
from numpy.testing import assert_allclose, assert_array_equal
from pathlib import Path

from config_minio import *


def make_tempfile():
    """Make dummy data."""
    temp_folder = tempfile.mkdtemp()
    s3_testfile = os.path.join(temp_folder,
                               's3_test_bizarre.nc')
    print(f"S3 Test file is {s3_testfile}")
    if not os.path.exists(s3_testfile):
        make_vanilla_ncdata(filename=s3_testfile)

    local_testfile = os.path.join(temp_folder,
                                  'local_test_bizarre.nc')
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
    Done twice: POSIX active and Reductionist; we compare results.

    identical to tests/test_harness.py::testActive()

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
    active = Active(s3_testfile_uri, "data", storage_type="s3")
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


@pytest.fixture
def test_data_path():
    """Path to test data for CMOR fixes."""
    return Path(__file__).resolve().parent / 'test_data'


def test_with_valid_netCDF_file(test_data_path):
    """
    Test as above but with an actual netCDF4 file.
    Also, this has _FillValue and missing_value

    identical to tests/test_bigger_data.py::test_cesm2_native

    """
    ncfile = str(test_data_path / "cesm2_native.nc")

    # run POSIX (local) Active
    active = Active(ncfile, "TREFHT")
    active._version = 2
    active.method = "mean"
    active.components = True
    result2 = active[4:5, 1:2]
    print(result2)

    # put data onto S3. then rm from local
    object = os.path.basename(ncfile)
    bucket_file = upload_to_s3(S3_URL, S3_ACCESS_KEY, S3_SECRET_KEY,
                               S3_BUCKET, object, ncfile)
    s3_testfile_uri = os.path.join("s3://", bucket_file)
    print("S3 file uri", s3_testfile_uri)

    # run Active on s3 file
    active = Active(s3_testfile_uri, "TREFHT", storage_type="s3")
    active._version = 2
    active.method = "mean"
    active.components = True
    result1 = active[4:5, 1:2]
    print(result1)

    # expect {'sum': array([[[2368.3232]]], dtype=float32), 'n': array([[[8]]])}
    # check for typing and structure
    assert_allclose(result1["sum"], np.array([[[2368.3232]]], dtype="float32"), rtol=1e-6)
    assert_array_equal(result1["n"], np.array([[[8]]]))

    assert_allclose(result1["sum"], result2["sum"], rtol=1e-6)
    assert_array_equal(result1["n"], result2["n"])


def test_reductionist_reduce_chunk():
    """Unit test for s3_reduce_chunk."""
    rfile = "tests/test_data/cesm2_native.nc"
    offset = 2
    size = 128
    object = os.path.basename(rfile)

    # create bucket and upload to Minio's S3 bucket
    upload_to_s3(S3_URL, S3_ACCESS_KEY, S3_SECRET_KEY,
                 S3_BUCKET, object, rfile)
    
    # call reductionist_reduce_chunk
    session = get_session(S3_ACCESS_KEY, S3_SECRET_KEY, S3_ACTIVE_STORAGE_CACERT)
    tmp, count = reductionist_reduce_chunk(session, S3_ACTIVE_STORAGE_URL,
                                           S3_URL, S3_BUCKET,
                                           object, offset, size, None, None,
                                           [], np.dtype("int32"), (32, ), "C",
                                           [slice(0, 2, 1), ], None, "min")
    assert tmp == 134351386
    # count is returned as a list even for one element
    assert count == [2]
