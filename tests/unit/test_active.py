import os
import numpy as np
import pytest
import threading

from activestorage.active import Active
from activestorage.config import *
from activestorage.s3 import reduce_chunk as s3_reduce_chunk


def test_uri_none():
    """Unit test for class:Active."""
    # test invalid uri
    some_file = None
    expected = "Must use a valid file for uri. Got None"
    with pytest.raises(ValueError) as exc:
        active = Active(some_file, ncvar="")
    assert str(exc.value) == expected


def test_uri_nonexistent():
    """Unit test for class:Active."""
    # test invalid uri
    some_file = 'cow.nc'
    expected = "Must use existing file for uri. cow.nc not found"
    with pytest.raises(ValueError) as exc:
        active = Active(some_file, ncvar="")
    assert str(exc.value) == expected


def test_getitem():
    """Unit test for class:Active."""
    # no variable passed
    uri = "tests/test_data/emac.nc"
    index = 3
    with pytest.raises(ValueError) as exc:
        active = Active(uri, ncvar=None)
    assert str(exc.value) == "Must set a netCDF variable name to slice"

    # unopenable file
    ncvar = "tas"
    baseexc = "tas not found in /"
    with pytest.raises(IndexError) as exc:
        active = Active(uri, ncvar=ncvar)
    assert baseexc in str(exc.value)

    # openable file and correct variable
    uri = "tests/test_data/cesm2_native.nc"
    ncvar = "TREFHT"
    active = Active(uri, ncvar=ncvar)
    assert active._version == 1
    item = active.__getitem__(index)[0, 0]
    expected = np.array(277.11945, dtype="float32")
    np.testing.assert_array_equal(item, expected)

    # test version iterations
    active._version = 0
    assert active._version == 0
    assert active.method is None
    item = active.__getitem__(index)[0, 0]
    expected = np.array(277.11945, dtype="float32")
    np.testing.assert_array_equal(item, expected)
    active._version = 3
    with pytest.raises(ValueError) as exc:
        item = active.__getitem__(index)[0, 0]
    assert str(exc.value) == "Version 3 not supported"


def test_method():
    """Unit test for class:Active."""
    uri = "tests/test_data/cesm2_native.nc"
    ncvar = "TREFHT"
    active = Active(uri, ncvar=ncvar)
    active._method = "min"
    assert active.method([3,444]) == 3

    active._method = "cow"
    assert active.method is None


def test_active():
    """Test with full complement of args."""
    uri = "tests/test_data/cesm2_native.nc"
    ncvar = "TREFHT"
    active = Active(uri, ncvar=ncvar)
    init = active.__init__(uri=uri, ncvar=ncvar, missing_value=True,
                           _FillValue=1e20, valid_min=-1,
                           valid_max=1200)


def test_lock():
    """Unit test for class:Active."""
    uri = "tests/test_data/cesm2_native.nc"
    ncvar = "TREFHT"
    active = Active(uri, ncvar=ncvar)

    lock = threading.Lock()
    active.lock = lock
    assert active.lock is lock

    # Pass through code that uses the lock
    active.method = None
    index = 3
    for version in (0, 2):
        active._version = version
        active[index]

    active.lock = None
    assert active.lock is False


def test_s3_reduce_chunk():
    """Unit test for s3_reduce_chunk."""
    rfile = "tests/test_data/cesm2_native.nc"
    offset = 2
    size = 128

    # no compression, filters, missing
    object = os.path.basename(rfile)
    with pytest.raises(OSError) as exc:
        tmp, count = s3_reduce_chunk(S3_ACTIVE_STORAGE_URL, S3_ACCESS_KEY,
                                     S3_SECRET_KEY, S3_URL, S3_BUCKET,
                                     object, offset, size,
                                     None, None, [],
                                     np.dtype("int32"), (8, 8),
                                     "C", [slice(0, 2, 1)],
                                     "min")
    url = " /v1/min"
    conn_err = "Failed to establish a new connection:"
    assert conn_err in str(exc.value)
    assert url in str(exc.value)
