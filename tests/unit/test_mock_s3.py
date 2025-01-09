import os
import s3fs
import pathlib
import boto3
import moto
import pyfive
import pytest
import h5netcdf

from tempfile import NamedTemporaryFile
from test_s3fs import s3 as tests3
from moto.moto_server.threaded_moto_server import ThreadedMotoServer

# some spoofy server parameters
port = 5555
endpoint_uri = "http://127.0.0.1:%s/" % port

@pytest.fixture(scope="module")
def s3_base():
    # writable local S3 system

    # This fixture is module-scoped, meaning that we can re-use the MotoServer across all tests
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=port)
    server.start()
    if "AWS_SECRET_ACCESS_KEY" not in os.environ:
        os.environ["AWS_SECRET_ACCESS_KEY"] = "foo"
    if "AWS_ACCESS_KEY_ID" not in os.environ:
        os.environ["AWS_ACCESS_KEY_ID"] = "foo"
    os.environ.pop("AWS_PROFILE", None)

    print("server up")
    yield
    print("moto done")
    server.stop()


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
    fs = s3fs.S3FileSystem(anon=True)
    with open('testobj.nc', 'wb') as ncdata:
        object.download_fileobj(ncdata)
    with open('testobj.nc', 'rb') as ncdata:
        ncfile = h5netcdf.File(ncdata, 'r', invalid_netcdf=True)
        print(ncfile)  # it works but...
    # correct coupling between boto3 and s3fs requires yielding
    # an s3fs Filesystem,
    # see https://stackoverflow.com/questions/75902766/how-to-access-my-own-fake-bucket-with-s3filesystem-pytest-and-moto

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

@pytest.mark.skip(reason="This test is now obsolete")
def test_s3file_spoofing(empty_bucket):
    ncfile = "./tests/test_data/daily_data.nc"
    file_path = pathlib.Path(ncfile)
    file_name = pathlib.Path(ncfile).name
    # partial spoofing with only boto3+moto
    result = spoof_s3("MY_BUCKET", file_name, file_path)

    with s3.open(os.path.join("MY_BUCKET", file_name), "rb") as f:
        ncfile = h5netcdf.File(f, 'r', invalid_netcdf=True)
    assert result.get('HTTPStatusCode') == 200


def test_s3file_spoofing_2(tests3):
    """
    This test spoofs a complete s3fs FileSystem,
    creates a mock bucket inside it, then puts a REAL netCDF4 file in it,
    then it loads it as if it was an S3 file. This is proper
    Wild Weasel stuff right here.
    """
    ncfile = "./tests/test_data/daily_data.nc"
    file_path = pathlib.Path(ncfile)
    file_name = pathlib.Path(ncfile).name

    # use s3fs proper
    bucket = "MY_BUCKET"
    tests3.mkdir(bucket)
    tests3.put(file_path, bucket)
    s3 = s3fs.S3FileSystem(
        anon=False, version_aware=True, client_kwargs={"endpoint_url": endpoint_uri}
    )
    with s3.open(os.path.join("MY_BUCKET", file_name), "rb") as f:
        ncfile = h5netcdf.File(f, 'r', invalid_netcdf=True)
        print("File loaded from spoof S3 with h5netcdf:", ncfile)
        print(x)
