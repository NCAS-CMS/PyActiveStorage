import os
import numpy as np
import pytest
import requests
import tempfile
from unittest import mock

from activestorage.active import Active
from activestorage.dummy_data import make_vanilla_ncdata
from activestorage import s3


def make_response(content, status_code, dtype=None, shape=None):
    response = requests.Response()
    response._content = content
    response.status_code = status_code
    if dtype:
        response.headers["x-activestorage-dtype"] = dtype
    if shape:
        response.headers["x-activestorage-shape"] = shape
    return response


@mock.patch.object(s3, 'request')
def test_s3_reduce_chunk(mock_request):
    """Unit test for s3_reduce_chunk."""
    result = np.int32(134351386)
    response = make_response(result.tobytes(), 200, "int32", "[]")
    mock_request.return_value = response

    active_url = "https://s3.example.com"
    access_key = "fake-access"
    secret_key = "fake-secret"
    s3_url = "https://active.example.com"
    bucket = "fake-bucket"
    object = "fake-object"
    offset = 2
    size = 128
    compression = None
    filters = None
    missing = []
    dtype = np.dtype("int32")
    shape = (32, )
    order = "C"
    chunk_selection = [slice(0, 2, 1)]
    operation = "min"

    # no compression, filters, missing

    tmp, count = s3.reduce_chunk(active_url, access_key, secret_key, s3_url,
                                 bucket, object, offset, size, compression,
                                 filters, missing, dtype, shape, order,
                                 chunk_selection, operation)

    assert tmp == result
    # count is None; no missing data yet in S3
    assert count == None

    expected_url = f"{active_url}/v1/{operation}/"
    expected_data = {
        "source": s3_url,
        "bucket": bucket,
        "object": object,
        "dtype": "int32",
        "offset": offset,
        "size": size,
        "order": order,
        "shape": shape,
        "selection": [[chunk_selection[0].start,
                       chunk_selection[0].stop,
                       chunk_selection[0].step]],
    }
    mock_request.assert_called_once_with(expected_url, access_key, secret_key,
                                         expected_data)


@mock.patch.object(s3, 'request')
def test_s3_reduce_chunk_not_found(mock_request):
    """Unit test for s3_reduce_chunk testing 404 response."""
    result = b'"Not found"'
    response = make_response(result, 404)
    mock_request.return_value = response

    active_url = "https://s3.example.com"
    access_key = "fake-access"
    secret_key = "fake-secret"
    s3_url = "https://active.example.com"
    bucket = "fake-bucket"
    object = "fake-object"
    offset = 2
    size = 128
    compression = None
    filters = None
    missing = []
    dtype = np.dtype("int32")
    shape = (32, )
    order = "C"
    chunk_selection = [slice(0, 2, 1)]
    operation = "min"

    with pytest.raises(s3.S3ActiveStorageError) as exc:
        s3.reduce_chunk(active_url, access_key, secret_key, s3_url, bucket,
                object, offset, size, compression, filters, missing, dtype,
                shape, order, chunk_selection, operation)


    assert str(exc.value) == 'S3 Active Storage error: HTTP 404: "Not found"'


def test_s3_storage_execution():
    """Test stack when call to Active contains storage_type == s3."""
    temp_folder = tempfile.mkdtemp()
    s3_testfile = os.path.join(temp_folder,
                               's3_test_bizarre.nc')
    if not os.path.exists(s3_testfile):
        make_vanilla_ncdata(filename=s3_testfile)

    active = Active(s3_testfile_uri, "data", "s3")
