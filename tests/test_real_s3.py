import os
import numpy as np

from activestorage.active import Active
from activestorage.active import load_from_s3

S3_BUCKET = "bnl"

# TODO Remove after full testing and right before deployment
def test_s3_dataset():
    """Run somewhat as the 'gold' test."""
    storage_options = {
        'key': "f2d55c6dcfc7618b2c34e00b58df3cef",
        'secret': "$/'#M{0{/4rVhp%n^(XeX$q@y#&(NM3W1->~N.Q6VP.5[@bLpi='nt]AfH)>78pT",
        'client_kwargs': {'endpoint_url': "https://uor-aces-o.s3-ext.jc.rl.ac.uk"},  # old proxy
        # 'client_kwargs': {'endpoint_url': "https://uor-aces-o.ext.proxy.jc.rl.ac.uk"},  # new proxy
    }
    active_storage_url = "https://192.171.169.113:8080"
    # bigger_file = "ch330a.pc19790301-bnl.nc"  # 18GB 3400 HDF5 chunks
    bigger_file = "ch330a.pc19790301-def.nc"  # 17GB 64 HDF5 chunks
    # bigger_file = "da193a_25_day__198808-198808.nc"  # 3GB 30 HDF5 chunks

    test_file_uri = os.path.join(
        S3_BUCKET,
        bigger_file
    )
    print("S3 Test file path:", test_file_uri)
    dataset = load_from_s3(test_file_uri, storage_options=storage_options)
    av = dataset['UM_m01s16i202_vn1106']

    # big file bnl: 18GB/3400 HDF5 chunks; def: 17GB/64 HDF5 chunks
    active = Active(av, storage_type="s3",
                    storage_options=storage_options,
                    active_storage_url=active_storage_url)
    active._version = 2
    active._method = "min"

    # result = active[:]
    result = active[0:3, 4:6, 7:9]  # standardized slice

    print("Result is", result)
    assert result == 5098.625
