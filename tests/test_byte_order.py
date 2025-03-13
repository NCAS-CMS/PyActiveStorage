import os
import pytest

from netCDF4 import Dataset

from activestorage.active import Active
from activestorage.config import *
from activestorage.dummy_data import make_byte_order_ncdata

import utils


def check_dataset_byte_order(temp_file: str, ncvar: str, byte_order: str):
    # Sanity check that test data has expected byte order (endianness).
    with Dataset(temp_file) as test_data:
        assert test_data.variables[ncvar].endian() == byte_order


def create_byte_order_dataset(tmp_path: str, byte_order: str):
    """
    Make a vanilla test dataset which has the specified byte order (endianness).
    """
    temp_file = str(tmp_path / "test_byte_order.nc")
    test_data = make_byte_order_ncdata(filename=temp_file, byte_order=byte_order)

    check_dataset_byte_order(temp_file, "data", byte_order)

    test_file = utils.write_to_storage(temp_file)
    if USE_S3:
        os.remove(temp_file)
    return test_file


@pytest.mark.parametrize('byte_order', ['big', 'little'])
def test_byte_order(tmp_path: str, byte_order: str):
    """
    Test use of datasets with different byte orders (endianness).
    """
    test_file = create_byte_order_dataset(tmp_path, byte_order)

    active = Active(test_file, 'data', storage_type=utils.get_storage_type())
    active._version = 1
    active._method = "min"
    result = active[0:2,4:6,7:9]
    assert result == 740.0
