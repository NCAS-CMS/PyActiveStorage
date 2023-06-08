import os
import numpy as np
import pytest

import activestorage.storage as st
from activestorage.s3 import reduce_chunk as s3_reduce_chunk

from .. import test_bigger_data


def test_reduce_chunk():
    """Test reduce chunk entirely."""
    rfile = "tests/test_data/cesm2_native.nc"
    offset = 2
    size = 128

    # no compression
    rc = st.reduce_chunk(rfile, offset, size,
                         compression=None, filters=None,
                         missing=[None, 2050, None, None],
                         dtype="i2", shape=(8, 8),
                         order="C", chunk_selection=slice(0, 2, 1),
                         method=np.min)
    assert rc[0] == -1
    assert rc[1] == 15

    # with compression
    with pytest.raises(NotImplementedError) as exc:
        rc = st.reduce_chunk(rfile, offset, size,
                             compression="Blosc", filters=None,
                             missing=[None, 2050, None, None],
                             dtype="i2", shape=(8, 8),
                             order="C", chunk_selection=slice(0, 2, 1),
                             method=np.max)
    assert str(exc.value) == "Compression is not yet supported!"

    # with filters
    with pytest.raises(NotImplementedError) as exc:
        rc = st.reduce_chunk(rfile, offset, size,
                             compression=None, filters="filters",
                             missing=[None, 2050, None, None],
                             dtype="i2", shape=(8, 8),
                             order="C", chunk_selection=slice(0, 2, 1),
                             method=np.max)
    assert str(exc.value) == "Filters are not yet supported!"


def test_reduced_chunk_masked_data():
    """Test method with masked data."""
    rfile = "tests/test_data/daily_data_masked.nc"
    offset = 6911
    size = 2976

    # no compression
    ch_sel = (slice(0, 62, 1), slice(0, 2, 1),
              slice(0, 3, 1), slice(0, 2, 1))
    rc = st.reduce_chunk(rfile, offset, size,
                         compression=None, filters=None,
                         missing=(None, 999.0, None, None),
                         dtype="float32", shape=(62, 2, 3, 2),
                         order="C", chunk_selection=ch_sel,
                         method=np.mean)
    # test the output dtype
    np.testing.assert_raises(AssertionError,
                             np.testing.assert_array_equal, rc, (249.459564, 680))
    # test result with correct dtype
    np.testing.assert_array_equal(rc, (np.float32(249.459564), 680))


def test_reduced_chunk_fully_masked_data_fill():
    """Test method with masked data."""
    rfile = "tests/test_data/daily_data_fullmask.nc"
    offset = 6911
    size = 2976

    # no compression
    ch_sel = (slice(0, 62, 1), slice(0, 2, 1),
              slice(0, 3, 1), slice(0, 2, 1))
    rc = st.reduce_chunk(rfile, offset, size,
                         compression=None, filters=None,
                         missing=(None, 999.0, None, None),
                         dtype="float32", shape=(62, 2, 3, 2),
                         order="C", chunk_selection=ch_sel,
                         method=np.mean)
    assert rc[0].size == 0
    assert rc[1] is None


def test_reduced_chunk_fully_masked_data_missing():
    """Test method with masked data."""
    rfile = "tests/test_data/daily_data_fullmask.nc"
    offset = 6911
    size = 2976

    # no compression
    ch_sel = (slice(0, 62, 1), slice(0, 2, 1),
              slice(0, 3, 1), slice(0, 2, 1))
    rc = st.reduce_chunk(rfile, offset, size,
                         compression=None, filters=None,
                         missing=(999., None, None, None),
                         dtype="float32", shape=(62, 2, 3, 2),
                         order="C", chunk_selection=ch_sel,
                         method=np.mean)
    assert rc[0].size == 0
    assert rc[1] is None


def test_reduced_chunk_fully_masked_data_vmin():
    """Test method with masked data."""
    rfile = "tests/test_data/daily_data_fullmask.nc"
    offset = 6911
    size = 2976

    # no compression
    ch_sel = (slice(0, 62, 1), slice(0, 2, 1),
              slice(0, 3, 1), slice(0, 2, 1))
    rc = st.reduce_chunk(rfile, offset, size,
                         compression=None, filters=None,
                         missing=(None, None, 1000., None),
                         dtype="float32", shape=(62, 2, 3, 2),
                         order="C", chunk_selection=ch_sel,
                         method=np.mean)
    assert rc[0].size == 0
    assert rc[1] is None


def test_reduced_chunk_fully_masked_data_vmax():
    """Test method with masked data."""
    rfile = "tests/test_data/daily_data_fullmask.nc"
    offset = 6911
    size = 2976

    # no compression
    ch_sel = (slice(0, 62, 1), slice(0, 2, 1),
              slice(0, 3, 1), slice(0, 2, 1))
    rc = st.reduce_chunk(rfile, offset, size,
                         compression=None, filters=None,
                         missing=(None, None, None, 1.),
                         dtype="float32", shape=(62, 2, 3, 2),
                         order="C", chunk_selection=ch_sel,
                         method=np.mean)
    assert rc[0].size == 0
    assert rc[1] is None


def test_like_s3_reduce_chunk():
    """Test reduce chunk entirely."""
    rfile = "tests/test_data/cesm2_native.nc"
    offset = 2
    size = 128

    # no compression
    rc = st.reduce_chunk(rfile, offset, size,
                         compression=None, filters=None,
                         missing=[None, None, None, None],
                         dtype=np.dtype("int32"), shape=(32, ),
                         order="C", chunk_selection=slice(0, 2, 1),
                         method=np.min)

    assert rc[0] == 134351386
    assert rc[1] == 2


def test_s3_reduce_chunk():
    """Unit test for s3_reduce_chunk."""
    rfile = "tests/test_data/cesm2_native.nc"
    offset = 2
    size = 128

    # no compression, filters, missing
    # identical test to one from test_storage
    object = os.path.basename(rfile)

    # create bucket and upload to Minio's S3 bucket
    S3_ACTIVE_STORAGE_URL = "http://localhost:8080"
    S3_URL = 'http://localhost:9000'
    S3_ACCESS_KEY = 'minioadmin'
    S3_SECRET_KEY = 'minioadmin'
    S3_BUCKET = 'pyactivestorage'
    # local test won't build a bucket
    try:
        test_bigger_data.upload_to_s3(S3_URL, S3_ACCESS_KEY, S3_SECRET_KEY,
                                      S3_BUCKET, object, rfile)
    except:
        pass

    # test on local machine fails; should pass when connected to Minio
    try:
        tmp, count = s3_reduce_chunk(S3_ACTIVE_STORAGE_URL, S3_ACCESS_KEY,
                                     S3_SECRET_KEY, S3_URL, S3_BUCKET,
                                     object, offset, size,
                                     None, None, [],
                                     np.dtype("int32"), (32, ),
                                     "C", [slice(0, 2, 1), ],
                                     "min")
        assert tmp == 134351386
        assert count == 2
    except:
        raise
