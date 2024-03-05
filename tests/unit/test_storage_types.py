# NOTE: Since these are unit tests, we can't assume that an S3 object store or
# active storage server is available. Therefore, we mock out the remote service
# interaction and replace with local file operations.

import botocore
import contextlib
import os
import h5netcdf
import numpy as np
import pytest
import requests.exceptions
from unittest import mock


import activestorage.active
from activestorage.active import Active
from activestorage.config import *
from activestorage.dummy_data import make_vanilla_ncdata
import activestorage.reductionist
import activestorage.storage




@mock.patch.object(activestorage.active, "load_from_s3")
@mock.patch.object(activestorage.active.reductionist, "reduce_chunk")
def test_s3(mock_reduce, mock_nz, mock_load, tmp_path):
    """Test stack when call to Active contains storage_type == s3."""

    # Since this is a unit test, we can't assume that an S3 object store or
    # active storage server is available. Therefore, we mock out the remote
    # service interaction and replace with local file operations.

    @contextlib.contextmanager
    def load_from_s3(uri):
        yield h5netcdf.File(test_file, 'r', invalid_netcdf=True)

    def reduce_chunk(
        session,
        server,
        source,
        bucket,
        object,
        offset,
        size,
        compressor,
        filters,
        missing,
        dtype,
        shape,
        order,
        chunk_selection,
        operation,
    ):
        return activestorage.storage.reduce_chunk(
            test_file,
            offset,
            size,
            compressor,
            filters,
            missing,
            dtype,
            shape,
            order,
            chunk_selection,
            np.max,
        )

    mock_load.side_effect = load_from_s3
    mock_nz.side_effect = load_netcdf_zarr_generic
    mock_reduce.side_effect = reduce_chunk

    uri = "s3://fake-bucket/fake-object"
    test_file = str(tmp_path / "test.nc")
    make_vanilla_ncdata(test_file)

    active = Active(uri, "data", "s3")
    active._version = 1
    active._method = "max"

    result = active[::]

    assert result == 999.0

    # S3 loading is not done from Active anymore
    mock_load.assert_not_called()

    mock_nz.assert_called_once_with(uri, "data", "s3", None)
    # NOTE: This gets called multiple times with various arguments. Match on
    # the common ones.
    mock_reduce.assert_called_with(
        mock.ANY,
        S3_ACTIVE_STORAGE_URL,
        S3_URL,
        mock.ANY,
        mock.ANY,
        mock.ANY,
        mock.ANY,
        None,
        None,
        (None, None, None, None),
        np.dtype("float64"),
        mock.ANY,
        "C",
        mock.ANY,
        operation="max",
    )


@mock.patch.object(activestorage.active, "load_from_s3")
def test_reductionist_version_0(mock_load, tmp_path):
    """Test stack when call to Active contains storage_type == s3 using version 0."""

    @contextlib.contextmanager
    def load_from_s3(uri, storage_options=None):
        yield h5netcdf.File(test_file, 'r', invalid_netcdf=True)

    mock_load.side_effect = load_from_s3

    uri = "s3://fake-bucket/fake-object"
    test_file = str(tmp_path / "test.nc")
    make_vanilla_ncdata(test_file)

    active = Active(uri, "data", "s3")
    active._version = 0

    result = active[::]

    assert np.max(result) == 999.0


@pytest.mark.skip(reason="No more valid file load in Active")
@mock.patch.object(activestorage.active, "load_from_s3")
def test_s3_load_failure(mock_load):
    """Test when an S3 object doesn't exist."""
    uri = "s3://fake-bucket/fake-object"

    mock_load.side_effect = FileNotFoundError

    with pytest.raises(FileNotFoundError):
        Active(uri, "data", "s3")


@mock.patch.object(activestorage.active, "load_from_s3")
@mock.patch.object(activestorage.active.reductionist, "reduce_chunk")
def test_reductionist_connection(mock_reduce, mock_nz, mock_load, tmp_path):
    """Test stack when call to Active contains storage_type == s3."""

    @contextlib.contextmanager
    def load_from_s3(uri):
        yield h5netcdf.File(test_file, 'r', invalid_netcdf=True)

    mock_load.side_effect = load_from_s3
    mock_reduce.side_effect = requests.exceptions.ConnectTimeout()

    uri = "s3://fake-bucket/fake-object"
    test_file = str(tmp_path / "test.nc")
    make_vanilla_ncdata(test_file)

    active = Active(uri, "data", "s3")
    active._version = 1
    active._method = "max"

    with pytest.raises(requests.exceptions.ConnectTimeout):
        assert active[::]


@mock.patch.object(activestorage.active, "load_from_s3")
@mock.patch.object(activestorage.active.reductionist, "reduce_chunk")
def test_reductionist_bad_request(mock_reduce, mock_nz, mock_load, tmp_path):
    """Test stack when call to Active contains storage_type == s3."""

    @contextlib.contextmanager
    def load_from_s3(uri):
        yield h5netcdf.File(test_file, 'r', invalid_netcdf=True)

    mock_load.side_effect = load_from_s3
    mock_nz.side_effect = load_netcdf_zarr_generic
    mock_reduce.side_effect = activestorage.reductionist.ReductionistError(400, "Bad request")

    uri = "s3://fake-bucket/fake-object"
    test_file = str(tmp_path / "test.nc")
    make_vanilla_ncdata(test_file)

    active = Active(uri, "data", "s3")
    active._version = 1
    active._method = "max"

    with pytest.raises(activestorage.reductionist.ReductionistError):
        assert active[::]
