import os
import numpy as np
import pyfive
import pytest

from activestorage.active import Active


S3_BUCKET = "bnl"

def build_active_test1_file():
    """Run an integration test with real data off S3 but with a small file."""
    storage_options = {
        'key': "f2d55c6dcfc7618b2c34e00b58df3cef",
        'secret': "$/'#M{0{/4rVhp%n^(XeX$q@y#&(NM3W1->~N.Q6VP.5[@bLpi='nt]AfH)>78pT",
        'client_kwargs': {'endpoint_url': "https://uor-aces-o.s3-ext.jc.rl.ac.uk"},  # final proxy
    }
    active_storage_url = "https://reductionist.jasmin.ac.uk/"  # Wacasoft new Reductionist
    bigger_file = "test1.nc"  # tas; 15 (time) x 143 x 144 

    test_file_uri = os.path.join(
        S3_BUCKET,
        bigger_file
    )
    print("S3 Test file path:", test_file_uri)
    active = Active(test_file_uri, 'tas', interface_type="s3",
                    storage_options=storage_options,
                    active_storage_url=active_storage_url)

    active._version = 2

    return active


def build_active_small_file():
    """Run an integration test with real data off S3 but with a small file."""
    storage_options = {
        'key': "f2d55c6dcfc7618b2c34e00b58df3cef",
        'secret': "$/'#M{0{/4rVhp%n^(XeX$q@y#&(NM3W1->~N.Q6VP.5[@bLpi='nt]AfH)>78pT",
        'client_kwargs': {'endpoint_url': "https://uor-aces-o.s3-ext.jc.rl.ac.uk"},  # final proxy
    }
    active_storage_url = "https://reductionist.jasmin.ac.uk/"  # Wacasoft new Reductionist
    bigger_file = "CMIP6-test.nc"  # tas; 15 (time) x 143 x 144 

    test_file_uri = os.path.join(
        S3_BUCKET,
        bigger_file
    )
    print("S3 Test file path:", test_file_uri)
    active = Active(test_file_uri, 'tas', interface_type="s3",
                    storage_options=storage_options,
                    active_storage_url=active_storage_url)

    active._version = 2

    return active


def test_small_file_axis_0_1():
    """Fails: activestorage.reductionist.ReductionistError: Reductionist error: HTTP 502: -"""
    active = build_active_small_file()
    result = active.min(axis=(0, 1))[:]
    print("Reductionist final result", result)
    assert min(result[0][0]) == 197.69595


def test_test1_file_axis_0_1():
    """Fails: activestorage.reductionist.ReductionistError: Reductionist error: HTTP 502: -"""
    active = build_active_test1_file()
    result = active.min(axis=(0, 1))[:]
    print("Reductionist final result", result)
    assert min(result[0][0]) == 198.82859802246094


def test_small_file_axis_0_1_compare_with_numpy():
    """Fails: activestorage.reductionist.ReductionistError: Reductionist error: HTTP 502: -"""
    active = build_active_small_file()
    result = active.min(axis=(0, 1))[:]
    print("Reductionist final result", result)

    # use numpy and local test data
    ds = pyfive.File("tests/test_data/CMIP6-test.nc")["tas"]
    minarr= np.min(ds[:], axis=(0, 1), keepdims=True)
    print(len(minarr))  # 144
    print(min(minarr))  # 197.69595
    assert np.min(result) == np.min(minarr)
    np.testing.assert_array_equal(result, minarr)


def build_active():
    """Run an integration test with real data off S3."""
    storage_options = {
        'key': "f2d55c6dcfc7618b2c34e00b58df3cef",
        'secret': "$/'#M{0{/4rVhp%n^(XeX$q@y#&(NM3W1->~N.Q6VP.5[@bLpi='nt]AfH)>78pT",
        'client_kwargs': {'endpoint_url': "https://uor-aces-o.s3-ext.jc.rl.ac.uk"},  # final proxy
    }
    active_storage_url = "https://reductionist.jasmin.ac.uk/"  # Wacasoft new Reductionist
    bigger_file = "da193a_25_6hr_t_pt_cordex__198807-198807.nc"  # m01s30i111  ## older 3GB 30 chunks

    test_file_uri = os.path.join(
        S3_BUCKET,
        bigger_file
    )
    print("S3 Test file path:", test_file_uri)
    active = Active(test_file_uri, 'm01s30i111', interface_type="s3",  # 'm01s06i247_4', interface_type="s3",
                    storage_options=storage_options,
                    active_storage_url=active_storage_url)

    active._version = 2

    return active


## Active loads a 4dim dataset
## Loaded dataset <HDF5 dataset "m01s30i111": shape (120, 85, 324, 432), type "float32">
## default axis arg (when axis=None): 'axis': (0, 1, 2, 3)

def test_no_axis():
    """
    Fails: it should pass: 'axis': (0, 1, 2, 3) default
    are fine!

    activestorage.reductionist.ReductionistError: Reductionist error: HTTP 400: {"error": {"message": "request data is not valid", "caused_by": ["__all__: Validation error: Number of reduction axes must be less than length of shape - to reduce over all axes omit the axis field completely [{}]"]}}
    """
    active = build_active()
    result = active.min()[:]
    assert result == [[[[164.8125]]]]


@pytest.mark.skip(reason="HIGHMEM: Reductionist returns a lot of response")
# TODO this test gobbles large amounts of memory - it shouldn't - it should
# perform like a standard global min - return a single number
def test_no_axis_2():
    """
    Fails: it should pass: 'axis': (0, 1, 2, 3) default
    are fine! Just as no axis is defined - global stats returned.
    """
    active = build_active()
    result = active.min(axis=())[:]
    assert result == [[[[164.8125]]]]


@pytest.mark.skip(reason="HIGHMEM: Reductionist returns a lot of response")
# TODO test on a machine with lots of memory
def test_axis_0():
    active = build_active()
    result = active.min(axis=(0, ))[:]
    assert result == [[[[164.8125]]]]


def test_axis_0_1():
    """Passes fine."""
    active = build_active()
    result = active.min(axis=(0, 1))[:]
    assert result.shape == (1, 1, 324, 432)
    assert result[0, 0, 0, 0] == 173.39794921875
    assert result[0, 0, 0, 431] == 173.395263671875


@pytest.mark.skip(reason="HIGHMEM: Reductionist returns a lot of response")
# TODO test on a machine with lots of memory
def test_axis_1():
    active = build_active()
    result = active.min(axis=(1, ))[:]
    assert result == [[[[164.8125]]]]


def test_axis_0_1_2():
    """Passes fine."""
    active = build_active()
    result = active.min(axis=(0, 1, 2))[:]
    assert result[0][0][0][0] == 171.05126953125
