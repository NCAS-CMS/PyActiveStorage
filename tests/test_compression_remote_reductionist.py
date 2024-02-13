import os
import numpy as np
import pytest

from netCDF4 import Dataset
from pathlib import Path

from activestorage.active import Active, load_from_s3
from activestorage.config import *
from activestorage.dummy_data import make_compressed_ncdata

import utils


# test two ways: local or Bryan's S3 machine + Bryan's reductionist
STORAGE_OPTIONS_Bryan = {
    'anon': True,
    'client_kwargs': {'endpoint_url': "https://uor-aces-o.s3-ext.jc.rl.ac.uk"},
}
S3_ACTIVE_URL_Bryan = "https://192.171.169.248:8080"
# TODO include all supported configuration types
storage_options_paramlist = [
    (None, None),
    (STORAGE_OPTIONS_Bryan, S3_ACTIVE_URL_Bryan)
]



@pytest.mark.parametrize("storage_options, active_storage_url", storage_options_paramlist)
def test_compression_and_filters_cmip6_data(storage_options, active_storage_url):
    """
    Test use of datasets with compression and filters applied for a real
    CMIP6 dataset (CMIP6-test.nc).
    """
    test_file = str(Path(__file__).resolve().parent / 'test_data' / 'CMIP6_IPSL-CM6A-LR_tas.nc')
    with Dataset(test_file) as nc_data:
        nc_min = np.min(nc_data["tas"][0:2,4:6,7:9])
    print(f"Numpy min from compressed file {nc_min}")

    active = Active(test_file, 'tas', utils.get_storage_type(),
                    storage_options=storage_options,
                    active_storage_url=active_storage_url)
    active._version = 1
    active._method = "min"
    result = active[0:2,4:6,7:9]
    assert nc_min == result
    assert result == 239.25946044921875
