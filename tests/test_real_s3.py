import os
import numpy as np
import pytest

from activestorage.active import Active
from activestorage.active import load_from_s3


S3_BUCKET = "bnl"

# this could be a slow test on GHA depending on network load
# also Githb machines are very far from Oxford
@pytest.mark.slow
def test_s3_dataset():
    """Run somewhat as the 'gold' test."""
    # NOTE: "https://uor-aces-o.s3-ext.jc.rl.ac.uk" is the stable S3 JASMIN
    # proxy that is now migrated to the new proxy (1 April 2025)
    storage_options = {
        'key': "f2d55c6dcfc7618b2c34e00b58df3cef",
        'secret': "$/'#M{0{/4rVhp%n^(XeX$q@y#&(NM3W1->~N.Q6VP.5[@bLpi='nt]AfH)>78pT",
        'client_kwargs': {'endpoint_url': "https://uor-aces-o.s3-ext.jc.rl.ac.uk"},
    }
    # active_storage_url = "https://192.171.169.113:8080"  # Bryan VM
    active_storage_url = "https://reductionist.jasmin.ac.uk/"  # Wacasoft
    # bigger_file = "ch330a.pc19790301-bnl.nc"  # 18GB 3400 HDF5 chunks
    bigger_file = "ch330a.pc19790301-def.nc"  # 17GB 64 HDF5 chunks
    # bigger_file = "da193a_25_day__198808-198808.nc"  # 3GB 30 HDF5 chunks

    test_file_uri = os.path.join(
        S3_BUCKET,
        bigger_file
    )
    print("S3 Test file path:", test_file_uri)

    # file: explicit storage_type
    active = Active(test_file_uri, 'UM_m01s16i202_vn1106',
                    storage_type="s3",
                    storage_options=storage_options,
                    active_storage_url=active_storage_url)
    active._version = 2
    result = active.min[0:3, 4:6, 7:9]  # standardized slice
    print("Result is", result)
    assert result == 5098.625

    # file: implicit storage_type
    active = Active(test_file_uri, 'UM_m01s16i202_vn1106',
                    storage_options=storage_options,
                    active_storage_url=active_storage_url)
    active._version = 2
    result = active.min[0:3, 4:6, 7:9]  # standardized slice
    print("Result is", result)
    assert result == 5098.625

    # load dataset
    dataset = load_from_s3(test_file_uri, storage_options=storage_options)
    av = dataset['UM_m01s16i202_vn1106']

    # dataset: explicit storage_type
    active = Active(av, storage_type="s3",
                    storage_options=storage_options,
                    active_storage_url=active_storage_url)
    active._version = 2
    result = active.min[0:3, 4:6, 7:9]  # standardized slice
    print("Result is", result)
    assert result == 5098.625

    # dataset: implicit storage_type
    active = Active(av,
                    storage_options=storage_options,
                    active_storage_url=active_storage_url)
    active._version = 2
    result = active.min[0:3, 4:6, 7:9]  # standardized slice
    print("Result is", result)
    assert result == 5098.625
