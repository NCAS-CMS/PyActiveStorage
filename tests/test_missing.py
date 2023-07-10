import os
from this import d
import numpy as np
import pytest
import shutil
import tempfile
import unittest

from netCDF4 import Dataset

from activestorage.active import Active
from activestorage.config import *
from activestorage import dummy_data as dd

import utils


"""
There is a high level of duplication in this test module:
this is to avert any issues of I/O toestepping from h5netcdf
when running with USE_S3=True; this will probably have to change
when we start using netCDF4python with S3-enabled
"""

def test_partially_missing_data(tmp_path):
    testfile = str(tmp_path / 'test_partially_missing_data.nc')
    r = dd.make_partially_missing_ncdata(testfile)

    # retrieve the actual numpy-ed result
    ds = Dataset(testfile)
    actual_data = ds["data"][:]
    ds.close()
    numpy_mean = np.ma.mean(actual_data[0:2, 4:6, 7:9])
    print("Numpy masked result (mean)", numpy_mean)

    # write file to storage
    testfile = utils.write_to_storage(testfile)

    # run the two Active instances: transfer data and do active storage
    active = Active(testfile, "data", utils.get_storage_type())
    active._version = 0
    d = active[0:2, 4:6, 7:9]

    # NOT numpy masked to check for correct Active behaviour
    mean_result = np.mean(d)
    print("No active storage result (mean)", mean_result)

    active = Active(testfile, "data", utils.get_storage_type())
    active._version = 2
    active.method = "mean"
    active.components = True
    result2 = active[0:2, 4:6, 7:9]
    active_result = result2["sum"] / result2["n"]
    print("Active storage result (mean)", active_result)

    if not USE_S3:
        np.testing.assert_array_equal(numpy_mean, active_result)
    else:
        np.testing.assert_raises(AssertionError,
                                 np.testing.assert_array_equal,
                                 numpy_mean, active_result)
    np.testing.assert_array_equal(mean_result, active_result)


def test_missing(tmp_path):
    testfile = str(tmp_path / 'test_missing.nc')
    r = dd.make_missing_ncdata(testfile)

    # retrieve the actual numpy-ed result
    ds = Dataset(testfile)
    actual_data = ds["data"][:]
    ds.close()
    numpy_mean = np.ma.mean(actual_data[0:2, 4:6, 7:9])
    print("Numpy masked result (mean)", numpy_mean)

    # write file to storage
    testfile = utils.write_to_storage(testfile)

    # run the two Active instances: transfer data and do active storage
    active = Active(testfile, "data", utils.get_storage_type())
    active._version = 0
    d = active[0:2, 4:6, 7:9]

    # NOT numpy masked to check for correct Active behaviour
    mean_result = np.mean(d)
    print("No active storage result (mean)", mean_result)

    active = Active(testfile, "data", utils.get_storage_type())
    active._version = 2
    active.method = "mean"
    active.components = True
    result2 = active[0:2, 4:6, 7:9]
    active_result = result2["sum"] / result2["n"]
    print("Active storage result (mean)", active_result)

    if not USE_S3:
        np.testing.assert_array_equal(numpy_mean, active_result)
    else:
        np.testing.assert_raises(AssertionError,
                                 np.testing.assert_array_equal,
                                 numpy_mean, active_result)
    np.testing.assert_array_equal(mean_result, active_result)



def test_fillvalue(tmp_path):
    """
    fill_value set to -999 from dummy_data.py
    note: no _FillValue attr set here, just fill_value!
    """
    testfile = str(tmp_path / 'test_fillvalue.nc')
    r = dd.make_fillvalue_ncdata(testfile)

    # retrieve the actual numpy-ed result
    ds = Dataset(testfile)
    actual_data = ds["data"][:]
    ds.close()
    actual_data = np.ma.masked_where(actual_data == -999., actual_data)
    numpy_mean = np.ma.mean(actual_data[0:2, 4:6, 7:9])
    print("Numpy masked result (mean)", numpy_mean)

    # write file to storage
    testfile = utils.write_to_storage(testfile)

    # run the two Active instances: transfer data and do active storage
    active = Active(testfile, "data", utils.get_storage_type())
    active._version = 0
    d = active[0:2, 4:6, 7:9]
    # d is unmasked, contains fill_values=-999 just like any other data points
    d = np.ma.masked_where(d == -999., d)

    # NOT masked
    mean_result = np.mean(d)
    print("No active storage result (mean)", mean_result)

    active = Active(testfile, "data", utils.get_storage_type())
    active._version = 2
    active.method = "mean"
    active.components = True
    result2 = active[0:2, 4:6, 7:9]
    active_result = result2["sum"] / result2["n"]
    print("Active storage result (mean)", active_result)

    if not USE_S3:
        np.testing.assert_array_equal(numpy_mean, active_result)
        np.testing.assert_array_equal(mean_result, active_result)
    else:
        np.testing.assert_raises(AssertionError,
                                 np.testing.assert_array_equal,
                                 numpy_mean, active_result)
        np.testing.assert_raises(AssertionError,
                                 np.testing.assert_array_equal,
                                 mean_result, active_result)


def test_validmin(tmp_path):
    testfile = str(tmp_path / 'test_validmin.nc')
    r = dd.make_validmin_ncdata(testfile)

    # retrieve the actual numpy-ed result
    ds = Dataset(testfile)
    actual_data = ds["data"][:]
    print(actual_data)
    ds.close()
    numpy_mean = np.ma.mean(actual_data[0:2, 4:6, 7:9])
    print("Numpy masked result (mean)", numpy_mean)

    # write file to storage
    testfile = utils.write_to_storage(testfile)

    # run the two Active instances: transfer data and do active storage
    active = Active(testfile, "data", utils.get_storage_type())
    active._version = 0
    d = active[0:2, 4:6, 7:9]

    # NOT numpy masked to check for correct Active behaviour
    mean_result = np.mean(d)
    print("No active storage result (mean)", mean_result)

    active = Active(testfile, "data", utils.get_storage_type())
    active._version = 2
    active.method = "mean"
    active.components = True
    result2 = active[0:2, 4:6, 7:9]
    active_result = result2["sum"] / result2["n"]
    print("Active storage result (mean)", active_result)

    np.testing.assert_array_equal(numpy_mean, active_result)
    np.testing.assert_array_equal(mean_result, active_result)
    print(x)


def test_validmax(tmp_path):
    testfile = str(tmp_path / 'test_validmax.nc')
    r = dd.make_validmax_ncdata(testfile)

    # retrieve the actual numpy-ed result
    ds = Dataset(testfile)
    actual_data = ds["data"][:]
    ds.close()
    numpy_mean = np.ma.mean(actual_data[0:2, 4:6, 7:9])
    print("Numpy masked result (mean)", numpy_mean)

    # write file to storage
    testfile = utils.write_to_storage(testfile)

    # run the two Active instances: transfer data and do active storage
    active = Active(testfile, "data", utils.get_storage_type())
    active._version = 0
    d = active[0:2, 4:6, 7:9]

    # NOT numpy masked to check for correct Active behaviour
    mean_result = np.mean(d)
    print("No active storage result (mean)", mean_result)

    active = Active(testfile, "data", utils.get_storage_type())
    active._version = 2
    active.method = "mean"
    active.components = True
    result2 = active[0:2, 4:6, 7:9]
    active_result = result2["sum"] / result2["n"]
    print("Active storage result (mean)", active_result)

    np.testing.assert_array_equal(numpy_mean, active_result)
    np.testing.assert_array_equal(mean_result, active_result)


def test_validrange(tmp_path):
    testfile = str(tmp_path / 'test_validrange.nc')
    r = dd.make_validrange_ncdata(testfile)

    # retrieve the actual numpy-ed result
    ds = Dataset(testfile)
    actual_data = ds["data"][:]
    ds.close()
    numpy_mean = np.ma.mean(actual_data[0:2, 4:6, 7:9])
    print("Numpy masked result (mean)", numpy_mean)

    # write file to storage
    testfile = utils.write_to_storage(testfile)

    # run the two Active instances: transfer data and do active storage
    active = Active(testfile, "data", utils.get_storage_type())
    active._version = 0
    d = active[0:2, 4:6, 7:9]

    # NOT numpy masked to check for correct Active behaviour
    mean_result = np.mean(d)
    print("No active storage result (mean)", mean_result)

    active = Active(testfile, "data", utils.get_storage_type())
    active._version = 2
    active.method = "mean"
    active.components = True
    result2 = active[0:2, 4:6, 7:9]
    active_result = result2["sum"] / result2["n"]
    print("Active storage result (mean)", active_result)

    np.testing.assert_array_equal(numpy_mean, active_result)
    np.testing.assert_array_equal(mean_result, active_result)
