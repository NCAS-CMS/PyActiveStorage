import fsspec
import pytest
import s3fs

from pathlib import Path
from kerchunk.hdf import SingleHdf5ToZarr

from activestorage.active import Active
from config_minio import *


@pytest.fixture
def test_data_path():
    """Path to test data."""
    return Path(__file__).resolve().parent / 'test_data'


def test_s3_SingleHdf5ToZarr():
    """Check Kerchunk's SingleHdf5ToZarr when S3."""
    s3_file = "s3://pyactivestorage/s3_test_bizarre_large.nc"
    fs = s3fs.S3FileSystem(key=S3_ACCESS_KEY,
                           secret=S3_SECRET_KEY,
                           client_kwargs={'endpoint_url': S3_URL},
                           default_fill_cache=False,
                           default_cache_type="none"
    )
    with fs.open(s3_file, 'rb') as s3file:
        h5chunks = SingleHdf5ToZarr(s3file, s3_file,
                                    inline_threshold=0)


def test_local_SingleHdf5ToZarr(test_data_path):
    """Check Kerchunk's SingleHdf5ToZarr when NO S3."""
    local_file = str(test_data_path / "test_bizarre.nc")
    fs = fsspec.filesystem('')
    with fs.open(local_file, 'rb') as localfile:
        h5chunks = SingleHdf5ToZarr(localfile, local_file,
                                    inline_threshold=0)


def test_Active_s3_v0():
    """
    Test truly Active with an S3 file.
    """
    # run Active on s3 file
    s3_file = "s3://pyactivestorage/s3_test_bizarre_large.nc"
    active = Active(s3_file, "data", "s3")
    active._version = 0
    active.components = True
    result1 = active[0:2, 4:6, 7:9]

def test_Active_s3_v2():
    """
    Test truly Active with an S3 file.
    """
    # run Active on s3 file
    s3_file = "s3://pyactivestorage/s3_test_bizarre_large.nc"
    active = Active(s3_file, "data", "s3")
    active._version = 2
    active.method = "mean"
    active.components = True
    result1 = active[0:2, 4:6, 7:9]


def test_Active_local_v0(test_data_path):
    """
    Test pulling the data locally.
    """
    # run Active on local file
    local_file = str(test_data_path / "test_bizarre.nc")
    active = Active(local_file, "data")
    active._version = 0
    active.components = True
    result2 = active[0:2, 4:6, 7:9]


def test_Active_local_v1(test_data_path):
    """
    Test pulling the data locally.
    """
    # run Active on local file
    local_file = str(test_data_path / "test_bizarre.nc")
    active = Active(local_file, "data")
    active._version = 1
    active.method = "mean"
    active.components = True
    result2 = active[0:2, 4:6, 7:9]


def test_Active_local_v2(test_data_path):
    """
    Test pulling the data locally.
    """
    # run Active on local file
    local_file = str(test_data_path / "test_bizarre.nc")
    active = Active(local_file, "data")
    active._version = 2
    active.method = "mean"
    active.components = True
    result2 = active[0:2, 4:6, 7:9]
