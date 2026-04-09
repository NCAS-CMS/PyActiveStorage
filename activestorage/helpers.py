import time
import urllib

import aiohttp
import fsspec
import numpy as np
import pyfive
import s3fs

from activestorage.config import S3_ACCESS_KEY, S3_SECRET_KEY, S3_URL


def load_from_https(uri, storage_options=None):
    """
    Load a pyfive.high_level.Dataset from a netCDF4 file on an https server.
    Works for both http and https endpoints.
    """
    if storage_options is None:
        client_kwargs = {'auth': None}
        fs = fsspec.filesystem('http', **client_kwargs)
        http_file = fs.open(uri, 'rb')
    else:
        username = storage_options.get("username", None)
        password = storage_options.get("password", None)
        client_kwargs = {
            'auth': aiohttp.BasicAuth(username, password) if username and password else None
        }
        fs = fsspec.filesystem('http', **client_kwargs)
        http_file = fs.open(uri, 'rb')

    ds = pyfive.File(http_file)
    print(f"Dataset loaded from https with Pyfive: {uri}")
    return ds


def load_from_s3(uri, storage_options=None):
    """Load a pyfive file-like dataset from S3."""
    if storage_options is None:
        fs = s3fs.S3FileSystem(
            key=S3_ACCESS_KEY,
            secret=S3_SECRET_KEY,
            client_kwargs={"endpoint_url": S3_URL},
        )
    else:
        fs = s3fs.S3FileSystem(**storage_options)

    t1 = time.time()
    s3file = fs.open(uri, "rb")
    t2 = time.time()
    ds = pyfive.File(s3file)
    t3 = time.time()
    print(f"Dataset loaded from S3 with s3fs and Pyfive: {uri} ({t2-t1:.2},{t3-t2:.2})")
    return ds


def get_missing_attributes(ds):
    """Load missing-value related attributes from a dataset variable."""

    def _hfix(x):
        if x is None:
            return x
        if not np.isscalar(x) and len(x) == 1:
            return x[0]
        return x

    fill_value = _hfix(ds.attrs.get("_FillValue"))
    missing_value = ds.attrs.get("missing_value")
    valid_min = _hfix(ds.attrs.get("valid_min"))
    valid_max = _hfix(ds.attrs.get("valid_max"))
    valid_range = _hfix(ds.attrs.get("valid_range"))

    if valid_max is not None or valid_min is not None:
        if valid_range is not None:
            raise ValueError(
                "Invalid combination in the file of valid_min, "
                "valid_max, valid_range: "
                f"{valid_min}, {valid_max}, {valid_range}"
            )
    elif valid_range is not None:
        valid_min, valid_max = valid_range

    return fill_value, missing_value, valid_min, valid_max


def get_endpoint_url(storage_options, filename):
    """Return endpoint URL from storage options or infer from URI."""
    if not storage_options:
        return f"http://{urllib.parse.urlparse(filename).netloc}"

    endpoint_url = storage_options.get("endpoint_url")
    if endpoint_url is not None:
        return endpoint_url

    client_kwargs = storage_options.get("client_kwargs")
    if client_kwargs:
        endpoint_url = client_kwargs.get("endpoint_url")
        if endpoint_url is not None:
            return endpoint_url

    return f"http://{urllib.parse.urlparse(filename).netloc}"


def return_interface_type(uri):
    """Infer interface type from URI scheme."""
    scheme = urllib.parse.urlparse(str(uri)).scheme
    if scheme in ("s3", "https"):
        return scheme
    return None
