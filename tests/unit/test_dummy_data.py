import os
import numpy as np
import pytest

from activestorage import dummy_data as dd
from netCDF4 import Dataset


def test_make_ncdata(tmp_path):
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
