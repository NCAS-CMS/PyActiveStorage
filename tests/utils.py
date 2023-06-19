"""Test utilities"""

import os

import s3fs

from activestorage.config import *

def get_storage_type():
    if USE_S3:
        return "s3"
    else:
        return None


def upload_to_s3(server, username, password, bucket, object, rfile):
    """Upload a file to an S3 object store."""
    s3_fs = s3fs.S3FileSystem(key=username, secret=password, client_kwargs={'endpoint_url': server})
    # Make sure s3 bucket exists
    try:
        s3_fs.mkdir(bucket)
    except FileExistsError:
        pass

    s3_fs.put_file(rfile, os.path.join(bucket, object))


def write_to_storage(ncfile):
    """Write a file to storage and return an appropriate URI or path to access it."""
    if USE_S3:
        object = os.path.basename(ncfile)
        upload_to_s3(S3_URL, S3_ACCESS_KEY, S3_SECRET_KEY, S3_BUCKET, object, ncfile)
        return os.path.join("s3://", S3_BUCKET, object)
    else:
        return ncfile
