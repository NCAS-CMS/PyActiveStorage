import os
import pytest

from netCDF4 import Dataset
from pathlib import Path

from activestorage.active import Active
from activestorage.config import *
from activestorage.dummy_data import make_compressed_ncdata

import utils


def check_dataset_filters(temp_file: str, ncvar: str, compression: str, shuffle: bool):
    # Sanity check that test data is compressed and filtered as expected.
    with Dataset(temp_file) as test_data:
        test_data_filters = test_data.variables[ncvar].filters()
        print(test_data_filters)
        assert test_data_filters[compression]
        assert test_data_filters['shuffle'] == shuffle


def create_compressed_dataset(tmp_path: str, compression: str, shuffle: bool):
    """
    Make a vanilla test dataset which is compressed and optionally shuffled.
    """
    temp_file = str(tmp_path / "test_compression.nc")
    test_data = make_compressed_ncdata(filename=temp_file, compression=compression, shuffle=shuffle)

    check_dataset_filters(temp_file, "data", compression, shuffle)

    test_file = utils.write_to_storage(temp_file)
    if USE_S3:
        os.remove(temp_file)
    return test_file


@pytest.mark.skipif(USE_S3, reason="Compression and filtering not supported in S3 yet")
@pytest.mark.parametrize('compression', ['zlib'])
@pytest.mark.parametrize('shuffle', [False, True])
def test_compression_and_filters(tmp_path: str, compression: str, shuffle: bool):
    """
    Test use of datasets with compression and filters applied.
    """
    test_file = create_compressed_dataset(tmp_path, compression, shuffle)

    active = Active(test_file, 'data', utils.get_storage_type())
    active._version = 1
    active._method = "min"
    result = active[0:2,4:6,7:9]
    assert result == 740.0


@pytest.mark.skipif(USE_S3, reason="Compression and filtering not supported in S3 yet")
def test_compression_and_filters_cmip6_data():
    """
    Test use of datasets with compression and filters applied for a real
    CMIP6 dataset (CMIP6_IPSL-CM6A-LR_tas).
    """
    test_file = str(Path(__file__).resolve().parent / 'test_data' / 'CMIP6_IPSL-CM6A-LR_tas.nc')

    check_dataset_filters(test_file, "tas", "zlib", False)

    active = Active(test_file, 'tas', utils.get_storage_type())
    active._version = 1
    active._method = "min"
    result = active[0:2,4:6,7:9]
    assert result == 239.25946044921875


@pytest.mark.skipif(USE_S3, reason="Compression and filtering not supported in S3 yet")
def test_compression_and_filters_obs4mips_data():
    """
    Test use of datasets with compression and filters applied for a real
    obs4mips dataset (obs4MIPS_CERES-EBAF_L3B_Ed2-8_rlut.nc) at CMIP5 MIP standard
    but with CMIP6-standard file packaging.
    """
    test_file = str(Path(__file__).resolve().parent / 'test_data' / 'obs4MIPS_CERES-EBAF_L3B_Ed2-8_rlut.nc')

    check_dataset_filters(test_file, "rlut", "zlib", False)

    active = Active(test_file, 'rlut', utils.get_storage_type())
    active._version = 1
    active._method = "min"
    result = active[0:2,4:6,7:9]
    assert result == 124.0
