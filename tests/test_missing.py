import os
from this import d
import numpy as np
import pytest
import shutil
import tempfile
import unittest

from activestorage.active import Active
from activestorage.config import *
from activestorage import dummy_data as dd

import utils


def _doit(testfile, **kwargs):
    """ 
    Compare and contrast vanilla mean with actual means
    """
    uri = utils.write_to_storage(testfile)
    active = Active(uri, "data", utils.get_storage_type())
    active._version = 0
    d = active[0:2, 4:6, 7:9]
    mean_result = np.mean(d)

    active = Active(uri, "data", utils.get_storage_type(), **kwargs)
    active._version = 2
    active.method = "mean"
    active.components = True
    result2 = active[0:2, 4:6, 7:9]
    np.testing.assert_array_equal(mean_result, result2["sum"]/result2["n"])

def test_partially_missing_data(tmp_path):
    testfile = str(tmp_path / 'test_partially_missing_data.nc')
    r = dd.make_partially_missing_ncdata(testfile)
    _doit(testfile, missing_value=-999.)

def test_missing(tmp_path):
    testfile = str(tmp_path / 'test_missing.nc')
    r = dd.make_partially_missing_ncdata(testfile)
    _doit(testfile, missing_value=-999.)

def test_fillvalue(tmp_path):
    testfile = str(tmp_path / 'test_fillvalue.nc')
    r = dd.make_fillvalue_ncdata(testfile)
    _doit(testfile, _FillValue=-999.)

def test_validmin(tmp_path):
    testfile = str(tmp_path / 'test_validmin.nc')
    r = dd.make_validmin_ncdata(testfile)
    _doit(testfile, valid_min=-1.)

def test_validmax(tmp_path):
    testfile = str(tmp_path / 'test_validmax.nc')
    r = dd.make_validmax_ncdata(testfile)
    _doit(testfile, valid_max=1.2 * 10 ** 3)

def test_validrange(tmp_path):
    testfile = str(tmp_path / 'test_validrange.nc')
    r = dd.make_validrange_ncdata(testfile)
    _doit(testfile, valid_min=-1., valid_max=1.2 * 10 ** 3)
