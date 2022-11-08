import os
import numpy as np
import pytest

from activestorage.active import Active


def test_uri_none():
    """Unit test for class:Active."""
    # test invalid uri
    some_file = None
    expected = "Must use a valid file for uri. Got None"
    with pytest.raises(ValueError) as exc:
        active = Active(some_file)
    assert str(exc.value) == expected


def test_uri_nonexistent():
    """Unit test for class:Active."""
    # test invalid uri
    some_file = 'cow.nc'
    expected = "Must use existing file for uri. cow.nc not found"
    with pytest.raises(ValueError) as exc:
        active = Active(some_file)
    assert str(exc.value) == expected


def test_getitem():
    """Unit test for class:Active."""
    # no variable passed
    uri = "tests/test_data/emac.nc"
    active = Active(uri, ncvar=None)
    index = 3
    with pytest.raises(ValueError) as exc:
        active.__getitem__(index)
    assert str(exc.value) == "Must set a netCDF variable name to slice"

    # unopenable file
    ncvar = "tas"
    active = Active(uri, ncvar=ncvar)
    baseexc = "Unable to open file (file signature not found)"
    with pytest.raises(OSError) as exc:
        item = active.__getitem__(index)
    assert str(exc.value) == baseexc

    # good file; wrong variable
    uri = "tests/test_data/daily_data.nc"
    ncvar = "tas"
    active = Active(uri, ncvar=ncvar)
    with pytest.raises(AttributeError) as exc:
        item = active.__getitem__(index)

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
    







