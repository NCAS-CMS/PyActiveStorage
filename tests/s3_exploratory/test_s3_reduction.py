import os
import numpy as np
import pytest
import s3fs

import activestorage.storage as st
from activestorage.s3 import reduce_chunk as s3_reduce_chunk

from config_minio import *


def upload_to_s3(server, username, password, bucket, object, rfile):
    """Upload a file to an S3 object store."""
    s3_fs = s3fs.S3FileSystem(key=username, secret=password, client_kwargs={'endpoint_url': server})
    # Make sure s3 bucket exists
    try:
        s3_fs.mkdir(bucket)
    except FileExistsError:
        pass

    s3_fs.put_file(rfile, os.path.join(bucket, object))


def test_s3_reduce_chunk():
    """Unit test for s3_reduce_chunk."""
    rfile = "tests/test_data/cesm2_native.nc"
    offset = 2
    size = 128
    object = os.path.basename(rfile)

    # create bucket and upload to Minio's S3 bucket
    upload_to_s3(S3_URL, S3_ACCESS_KEY, S3_SECRET_KEY,
                 S3_BUCKET, object, rfile)
    
    tmp, count = s3_reduce_chunk(S3_ACTIVE_STORAGE_URL, S3_ACCESS_KEY,
                                 S3_SECRET_KEY, S3_URL, S3_BUCKET,
                                 object, offset, size,
                                 None, None, [],
                                 np.dtype("int32"), (32, ),
                                 "C", [slice(0, 2, 1), ],
                                 "min")
    assert tmp == 134351386
    assert count == None
