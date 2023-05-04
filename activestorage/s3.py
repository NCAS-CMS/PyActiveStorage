"""S3 active storage module."""

import http.client
import json
import requests
import numpy as np
import sys


def reduce_chunk(server, username, password, source, bucket, object, offset,
                 size, compression, filters, missing, dtype, shape, order,
                 chunk_selection, operation):
    """Perform a reduction on a chunk using S3 Active Storage.

    :param server: S3 active storage server URL
    :param username: S3 username / access key
    :param password: S3 password / secret key
    :param source: S3 URL
    :param bucket: S3 bucket
    :param object: S3 object
    :param offset: offset of data in object
    :param size: size of data in object
    :param compression: name of compression, unsupported
    :param filters: name of filters, unsupported
    :param missing: 4-tuple describing missing data, unsupported
    :param dtype: data type name
    :param shape: will be a tuple, something like (3,3,1), this is the
                  dimensionality of the chunk itself
    :param order: typically 'C' for c-type ordering
    :param chunk_selection: N-tuple where N is the length of `shape`, and each
                            item is an integer or slice.  e.g.  (slice(0, 2,
                            1), slice(1, 3, 1), slice(0, 1, 1))
                            this defines the part of the chunk which is to be
                            obtained or operated upon.
    :param operation: name of operation to perform
    :returns: the reduced data as a numpy array or scalar
    :raises S3ActiveStorageError: if the request to S3 active storage fails
    """

    if compression is not None:
        raise NotImplementedError("Compression is not yet supported!")
    if filters is not None:
        raise NotImplementedError("Filters are not yet supported!")

    request_data = build_request_data(source, bucket, object, offset, size, compression, filters, missing, dtype, shape, order, chunk_selection)
    api_operation = "sum" if operation == "mean" else operation or "select"
    url = f'{server}/v1/{api_operation}/'
    response = request(url, username, password, request_data)

    if response.ok:
        # FIXME: Return count from mean
        result = decode_result(response)
        if operation == "mean":
            count = reduce_chunk(server, username, password, source, bucket, object, offset, size, compression, filters, missing, dtype, shape, order, chunk_selection, "count")[0]
        else:
            count = None
        return result, count
    else:
        decode_and_raise_error(response)


def encode_selection(selection):
    """Encode a chunk selection in a JSON-compatible format."""
    def encode_slice(s):
        if isinstance(s, slice):
            return [s.start, s.stop, s.step]
        else:
            # Integer - select single value
            return [s, s + 1, 1]

    return [encode_slice(s) for s in selection]


def build_request_data(source: str, bucket: str, object: str, offset: int,
                       size: int, compression, filters, missing, dtype, shape,
                       order, selection) -> dict:
    """Build request data for S3 active storage API."""
    # TODO: compression, filters, missing
    request_data = {
        'source': source,
        'bucket': bucket,
        'object': object,
        'dtype': dtype.name,
        'offset': offset,
        'size': size,
        'order': order,
    }
    if shape:
        request_data["shape"] = shape
    if selection:
        request_data["selection"] = encode_selection(selection)
    return {k: v for k, v in request_data.items() if v is not None}


def request(url: str, username: str, password: str, request_data: dict):
    """Make a request to an S3 active storage API."""
    response = requests.post(
        url,
        json=request_data,
        auth=(username, password)
    )
    return response


def decode_result(response):
    """Decode a successful response, return as a numpy array or scalar."""
    dtype = response.headers['x-activestorage-dtype']
    shape = json.loads(response.headers['x-activestorage-shape'])
    result = np.frombuffer(response.content, dtype=dtype)
    result = result.reshape(shape)
    return result


class S3ActiveStorageError(Exception):
    """Exception for S3 Active Storage failures."""

    def __init__(self, status_code, error):
        super(S3ActiveStorageError, self).__init__(f"S3 Active Storage error: HTTP {status_code}: {error}")


def decode_and_raise_error(response):
    """Decode an error response and raise S3ActiveStorageError."""
    try:
        error = json.dumps(response.json())
        raise S3ActiveStorageError(response.status_code, error)
    except requests.exceptions.JSONDecodeError as exc:
        raise S3ActiveStorageError(response.status_code, "-") from exc
