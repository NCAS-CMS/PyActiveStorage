import os
import numpy as np

from activestorage.active import Active


S3_BUCKET = "bnl"

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
    active = Active(test_file_uri, 'm01s30i111', storage_type="s3",  # 'm01s06i247_4', storage_type="s3",
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


def test_no_axis_2():
    """
    Fails: it should pass: 'axis': (0, 1, 2, 3) default
    are fine!

    activestorage.reductionist.ReductionistError: Reductionist error: HTTP 400: {"error": {"message": "request data is not valid", "caused_by": ["__all__: Validation error: Number of reduction axes must be less than length of shape - to reduce over all axes omit the axis field completely [{}]"]}}
    """
    active = build_active()
    result = active.min(axis=())[:]
    assert result == [[[[164.8125]]]]


def test_axis_0():
    """Fails: activestorage.reductionist.ReductionistError: Reductionist error: HTTP 502: -"""
    active = build_active()
    result = active.min(axis=(0, ))[:]
    assert result == [[[[164.8125]]]]


def test_axis_0_1():
    """Fails: activestorage.reductionist.ReductionistError: Reductionist error: HTTP 502: -"""
    active = build_active()
    result = active.min(axis=(0, 1))[:]
    assert result == [[[[164.8125]]]]


def test_axis_1():
    """Fails: activestorage.reductionist.ReductionistError: Reductionist error: HTTP 502: -"""
    active = build_active()
    result = active.min(axis=(1, ))[:]
    assert result == [[[[164.8125]]]]


def test_axis_0_1_2():
    """Passes fine."""
    active = build_active()
    result = active.min(axis=(0, 1, 2))[:]
    assert result[0][0][0][0] == 171.05126953125
