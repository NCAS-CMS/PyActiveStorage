import os
from this import d
import numpy as np
import pytest
import shutil
import tempfile
import unittest

from activestorage.active import Active
from activestorage import dummy_data as dd


def _doit(testfile):
    """ 
    Compare and contrast vanilla mean with actual means
    """
    active = Active(testfile, "data")
    active._version = 0
    d = active[0:2, 4:6, 7:9]
    mean_result = np.mean(d)

    active = Active(testfile, "data")
    active._version = 2
    active.method = "mean"
    active.components = True
    result2 = active[0:2, 4:6, 7:9]
    np.testing.assert_array_equal(mean_result, result2["sum"]/result2["n"])


def test_partially_missing_data(tmp_path):
    testfile = tmp_path / 'test_partially_missing_data.nc'
    r = dd.make_partially_missing_ncdata(testfile)
    _doit(testfile)

def test_missing(tmp_path):
    testfile = tmp_path / 'test_missing.nc'
    r = dd.make_partially_missing_ncdata(testfile)
    _doit(testfile)

def test_fillvalue(tmp_path):
    testfile = tmp_path / 'test_fillvalue.nc'
    r = dd.make_fillvalue_ncdata(testfile)
    _doit(testfile)

def test_validmin(tmp_path):
    testfile = tmp_path / 'test_validmin.nc'
    r = dd.make_validmin_ncdata(testfile)
    _doit(testfile)

def test_validmax(tmp_path):
    testfile = tmp_path / 'test_validmax.nc'
    r = dd.make_validmax_ncdata(testfile)
    _doit(testfile)

def test_validrange(tmp_path):
    testfile = tmp_path / 'test_validrange.nc'
    r = dd.make_validrange_ncdata(testfile)
    _doit(testfile)
