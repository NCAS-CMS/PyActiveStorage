import os
import numpy as np
import pytest

from activestorage import dummy_data as dd
from netCDF4 import Dataset


def test_make_ncdata_minmax(tmp_path):
    """Test dummy data's make_ncdata."""
    filename = tmp_path / 'kitchen_sink.nc'
    datfile = dd.make_ncdata(filename,
                             chunksize=(3, 3, 1),
                             n=10,
                             compression=None,
                             missing=1e20,
                             fillvalue=-999.,
                             valid_range=None,
                             valid_min=-1.,
                             valid_max=1.2 * 10 ** 3)
    content = Dataset(filename)
    assert content.dimensions["xdim"].size == 10
    assert content.dimensions["ydim"].size == 10
    assert content.dimensions["zdim"].size == 10
    assert content.variables["data"].missing_value == 1e+20
    assert content.variables["data"].valid_min == -1.0
    assert content.variables["data"].valid_max == 1200.0


def test_make_ncdata_range(tmp_path):
    """Test dummy data's make_ncdata."""
    filename = tmp_path / 'kitchen_sink_2.nc'
    datfile = dd.make_ncdata(filename,
                             chunksize=(3, 3, 1),
                             n=10,
                             compression=None,
                             missing=1e20,
                             fillvalue=-999.,
                             valid_range=[-1.0, 1.2 * 10 ** 3],
                             valid_min=None,
                             valid_max=None)
    content = Dataset(filename)
    assert content.dimensions["xdim"].size == 10
    assert content.dimensions["ydim"].size == 10
    assert content.dimensions["zdim"].size == 10
    assert content.variables["data"].missing_value == 1e+20
    np.testing.assert_array_equal(content.variables["data"].valid_range,
                                  [-1., 1200.])


def test_make_fillvalue_ncdata(tmp_path):
    filename = tmp_path / 'test_fillvalue.nc'
    c = dd.make_fillvalue_ncdata(filename=filename,
                                 chunksize=(3, 3, 1), n=10)


def test_make_missing_ncdata(tmp_path):
    filename = tmp_path / 'test_missing.nc'
    c = dd.make_missing_ncdata(filename=filename,
                                 chunksize=(3, 3, 1), n=10)


def test_make_validmin_ncdata(tmp_path):
    filename = tmp_path / 'test_validmin.nc'
    c = dd.make_validmin_ncdata(filename=filename,
                                chunksize=(3, 3, 1), n=10)


def test_make_validmax_ncdata(tmp_path):
    filename = tmp_path / 'test_validmax.nc'
    c = dd.make_validmax_ncdata(filename=filename,
                                chunksize=(3, 3, 1), n=10)


def test_make_validrange_ncdata(tmp_path):
    filename = tmp_path / 'test_validrange.nc'
    c = dd.make_validrange_ncdata(filename=filename,
                                  chunksize=(3, 3, 1), n=10)


def test_make_vanilla_ncdata(tmp_path):
    filename = tmp_path / 'test_vanilla.nc'
    c = dd.make_vanilla_ncdata(filename=filename,
                               chunksize=(3, 3, 1), n=10)
