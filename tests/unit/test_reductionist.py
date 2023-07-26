import os
import numpy as np
import pytest
import requests
from unittest import mock

from activestorage import reductionist


def make_response(content, status_code, dtype=None, shape=None, count=None):
    response = requests.Response()
    response._content = content
    response.status_code = status_code
    if dtype:
        response.headers["x-activestorage-dtype"] = dtype
    if shape:
        response.headers["x-activestorage-shape"] = shape
    if count:
        response.headers["x-activestorage-count"] = count
    return response


@mock.patch.object(reductionist, 'request')
def test_reduce_chunk(mock_request):
    """Unit test for reduce_chunk."""
    result = np.int32(134351386)
    response = make_response(result.tobytes(), 200, "int32", "[]", "2")
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

    tmp, count = reductionist.reduce_chunk(active_url, access_key, secret_key, s3_url,
                                           bucket, object, offset, size,
                                           compression, filters, missing,
                                           dtype, shape, order,
                                           chunk_selection, operation)

    assert tmp == result
    assert count == 2

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


@mock.patch.object(reductionist, 'request')
def test_reduce_chunk_not_found(mock_request):
    """Unit test for reduce_chunk testing 404 response."""
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

    with pytest.raises(reductionist.ReductionistError) as exc:
        reductionist.reduce_chunk(active_url, access_key, secret_key, s3_url, bucket,
                                  object, offset, size, compression, filters,
                                  missing, dtype, shape, order,
                                  chunk_selection, operation)


    assert str(exc.value) == 'Reductionist error: HTTP 404: "Not found"'
