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


def _doit(testfile):
    """ 
    Compare and contrast vanilla mean with actual means
    """
    uri = utils.write_to_storage(testfile)
    active = Active(uri, "data", utils.get_storage_type())
    active._version = 0
    d = active[0:2, 4:6, 7:9]
    mean_result = np.mean(d)
    print("Bogstandard numpy", mean_result)

    active = Active(uri, "data", utils.get_storage_type())
    active._version = 2
    active.method = "mean"
    active.components = True
    result2 = active[0:2, 4:6, 7:9]
    print("Active result", result2["sum"]/result2["n"])
    print(x)
    np.testing.assert_array_equal(mean_result, result2["sum"]/result2["n"])


# @pytest.mark.skipif(USE_S3, reason="Missing data not supported in S3 yet")
def test_partially_missing_data(tmp_path):
    testfile = str(tmp_path / 'test_partially_missing_data.nc')
    r = dd.make_partially_missing_ncdata(testfile)
    _doit(testfile)

# @pytest.mark.skipif(USE_S3, reason="Missing data not supported in S3 yet")
def test_missing(tmp_path):
    testfile = str(tmp_path / 'test_missing.nc')
    r = dd.make_missing_ncdata(testfile)
    _doit(testfile)

# @pytest.mark.skipif(USE_S3, reason="Missing data not supported in S3 yet")
def test_fillvalue(tmp_path):
    testfile = str(tmp_path / 'test_fillvalue.nc')
    r = dd.make_fillvalue_ncdata(testfile)
    _doit(testfile)

# @pytest.mark.skipif(USE_S3, reason="Missing data not supported in S3 yet")
def test_validmin(tmp_path):
    testfile = str(tmp_path / 'test_validmin.nc')
    r = dd.make_validmin_ncdata(testfile)
    _doit(testfile)

# @pytest.mark.skipif(USE_S3, reason="Missing data not supported in S3 yet")
def test_validmax(tmp_path):
    testfile = str(tmp_path / 'test_validmax.nc')
    r = dd.make_validmax_ncdata(testfile)
    _doit(testfile)

# @pytest.mark.skipif(USE_S3, reason="Missing data not supported in S3 yet")
def test_validrange(tmp_path):
    testfile = str(tmp_path / 'test_validrange.nc')
    r = dd.make_validrange_ncdata(testfile)
    _doit(testfile)
