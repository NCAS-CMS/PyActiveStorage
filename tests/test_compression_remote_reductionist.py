import os
import numpy as np
import pytest

from netCDF4 import Dataset
from pathlib import Path

from activestorage.active import Active, load_from_s3
from activestorage.config import *
from activestorage.dummy_data import make_compressed_ncdata
from activestorage.reductionist import ReductionistError as RedErr

import utils


# Bryan's S3 machine + Bryan's reductionist
STORAGE_OPTIONS_Bryan = {
    'anon': True,
    'client_kwargs': {'endpoint_url': "https://uor-aces-o.s3-ext.jc.rl.ac.uk"},
}
S3_ACTIVE_URL_Bryan = "https://192.171.169.248:8080"
# TODO include all supported configuration types
storage_options_paramlist = [
    (STORAGE_OPTIONS_Bryan, S3_ACTIVE_URL_Bryan)
]
# bucket needed too for this test only
# otherwise, bucket is extracted automatically from full file uri
S3_BUCKET = "bnl"


@pytest.mark.parametrize("storage_options, active_storage_url", storage_options_paramlist)
def test_compression_and_filters_cmip6_data(storage_options, active_storage_url):
    """
    Test use of datasets with compression and filters applied for a real
    CMIP6 dataset (CMIP6-test.nc) - an IPSL file.

    This test will always pass when USE_S3 = False; equally, it will always
    fail if USE_S3 = True until Reductionist supports anon=True S3 buckets. See following
    test below with a forced storage_type="s3" that mimicks locally the fail, and catches it.
    """
    test_file = str(Path(__file__).resolve().parent / 'test_data' / 'CMIP6-test.nc')
    with Dataset(test_file) as nc_data:
        nc_min = np.min(nc_data["tas"][0:2,4:6,7:9])
    print(f"Numpy min from compressed file {nc_min}")

    # TODO remember that the special case for "anon=True" buckets is that
    # the actual file uri = "bucket/filename"
    if USE_S3:
        ofile = os.path.basename(test_file)
        test_file_uri = os.path.join(S3_BUCKET, ofile)
    else:
        test_file_uri = test_file
    active = Active(test_file_uri, 'tas', utils.get_storage_type(),
                    storage_options=storage_options,
                    active_storage_url=active_storage_url)
    active._version = 1
    active._method = "min"
    result = active[0:2,4:6,7:9]
    assert nc_min == result
    assert result == 239.25946044921875


@pytest.mark.parametrize("storage_options, active_storage_url", storage_options_paramlist)
def test_compression_and_filters_cmip6_forced_s3(storage_options, active_storage_url):
    """
    Test use of datasets with compression and filters applied for a real
    CMIP6 dataset (CMIP6-test.nc) - an IPSL file.

    This is for a special anon=True bucket ONLY.
    """
    test_file = str(Path(__file__).resolve().parent / 'test_data' / 'CMIP6-test.nc')
    with Dataset(test_file) as nc_data:
        nc_min = np.min(nc_data["tas"][0:2,4:6,7:9])
    print(f"Numpy min from compressed file {nc_min}")

    # TODO remember that the special case for "anon=True" buckets is that
    # the actual file uri = "bucket/filename"
    ofile = os.path.basename(test_file)
    test_file_uri = os.path.join(S3_BUCKET, ofile)
    active = Active(test_file_uri, 'tas', storage_type="s3",
                    storage_options=storage_options,
                    active_storage_url=active_storage_url)

    active._version = 1
    active._method = "min"

    # for now anon=True S3 buckets are not supported by Reductionist
    with pytest.raises(RedErr) as rederr:
        result = active[0:2,4:6,7:9]
    access_denied_err = 'code: \\"AccessDenied\\"'
    assert access_denied_err in str(rederr.value)
    # assert nc_min == result
    # assert result == 239.25946044921875
