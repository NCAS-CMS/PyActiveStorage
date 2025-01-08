import os
import s3fs
import pathlib
import boto3
import moto
import pyfive
import pytest
import h5netcdf

from tempfile import NamedTemporaryFile


def spoof_s3(bucket, file_name, file_path):
    # "put" file
    if os.path.exists(file_path):
        with open(file_path, "rb") as file_contents:
            conn = boto3.session.Session()
            s3 = conn.resource('s3')
            object = s3.Object(bucket, file_name)
            result = object.put(Body=file_contents)
            res = result.get('ResponseMetadata')
            if res.get('HTTPStatusCode') == 200:
                print('File Uploaded Successfully')
            else:
                print('File Not Uploaded Successfully')

    # "download" file
    s3 = boto3.resource('s3')
    # arg0: file in bucket; arg1: file to download to
    target_file = "test.nc"
    s3file = s3.Bucket(bucket).download_file(file_name, target_file)
    print(os.path.isfile(target_file))

    # "access" file "remotely" with s3fs
    fs = s3fs.S3FileSystem()
    with open('testobj.nc', 'wb') as ncdata:
        object.download_fileobj(ncdata)
    with open('testobj.nc', 'rb') as ncdata:
        ncfile = h5netcdf.File(ncdata, 'r', invalid_netcdf=True)
        print(ncfile)  # it works but...

    return res


@pytest.fixture(scope='session')
def aws_credentials():
    """Mocked AWS Credentials for moto."""
    os.environ['AWS_ACCESS_KEY_ID'] = 'testing'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'testing'
    os.environ['AWS_SECURITY_TOKEN'] = 'testing'
    os.environ['AWS_SESSION_TOKEN'] = 'testing'
    os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

    try:
        tmp = NamedTemporaryFile(delete=False)
        tmp.write(b"""[wild weasel]
aws_access_key_id = testing
aws_secret_access_key = testing""")
        tmp.close()
        os.environ['AWS_SHARED_CREDENTIALS_FILE'] = str(tmp.name)
        yield
    finally:
        os.unlink(tmp.name)


@pytest.fixture(scope='function')
def empty_bucket(aws_credentials):
    moto_fake = moto.mock_aws()
    try:
        moto_fake.start()
        conn = boto3.resource('s3')
        conn.create_bucket(Bucket="MY_BUCKET")
        yield conn
    finally:
        moto_fake.stop()


def test_s3file_spoofing(empty_bucket):
    ncfile = "./tests/test_data/daily_data.nc"
    file_path = pathlib.Path(ncfile)
    file_name = pathlib.Path(ncfile).name
    result = spoof_s3("MY_BUCKET", file_name, file_path)
    assert result.get('HTTPStatusCode') == 200
