import os
import s3fs
import pathlib
import pyfive
import pytest
import h5netcdf
import numpy as np

from tempfile import NamedTemporaryFile
from activestorage.active import load_from_s3, Active


# needed by the spoofed s3 filesystem
port = 5555
endpoint_uri = "http://127.0.0.1:%s/" % port


def test_s3fs_s3(s3fs_s3):
    """Test mock S3 filesystem constructor."""
    # this is an entire mock S3 FS
    mock_s3_filesystem = s3fs_s3

    # explore its attributes and methods
    print(dir(mock_s3_filesystem))

    assert not mock_s3_filesystem.anon
    assert not mock_s3_filesystem.version_aware
    assert mock_s3_filesystem.client_kwargs == {'endpoint_url': 'http://127.0.0.1:5555/'}


def spoof_boto3_s3(bucket, file_name, file_path):
    # this is a pure boto3 implementation
    # I am leaving it here just in case we'll ever need it in the future
    # NOTE: we are NOT including boto3 as dependency yet, until we ever need it

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
        print(ncfile)

    return res


@pytest.fixture(scope='session')
def aws_credentials():
    """
    Mocked AWS Credentials for moto.
    NOTE: Used ONLY by the pure boto3 test method spoof_boto3_s3.
    """
    # NOTE: Used ONLY by the pure boto3 test method spoof_boto3_s3
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
    """Create an empty bucket."""
    # NOTE: Used ONLY by the pure boto3 test method spoof_boto3_s3
    moto_fake = moto.mock_aws()
    try:
        moto_fake.start()
        conn = boto3.resource('s3')
        conn.create_bucket(Bucket="MY_BUCKET")
        yield conn
    finally:
        moto_fake.stop()


@pytest.mark.skip(reason="This test uses the pure boto3 implement which we don't need at the moment.")
def test_s3file_with_pure_boto3(empty_bucket):
    ncfile = "./tests/test_data/daily_data.nc"
    file_path = pathlib.Path(ncfile)
    file_name = pathlib.Path(ncfile).name
    # partial spoofing with only boto3+moto
    result = spoof_s3("MY_BUCKET", file_name, file_path)
    with s3.open(os.path.join("MY_BUCKET", file_name), "rb") as f:
        ncfile = h5netcdf.File(f, 'r', invalid_netcdf=True)
    assert result.get('HTTPStatusCode') == 200


def test_s3file_with_s3fs(s3fs_s3):
    """
    This test spoofs a complete s3fs FileSystem via s3fs_s3,
    creates a mock bucket inside it, then puts a REAL netCDF4 file in it,
    then it loads it as if it was an S3 file. This is proper
    Wild Weasel stuff right here.
    """
    # set up physical file and Path properties
    ncfile = "./tests/test_data/daily_data.nc"
    file_path = pathlib.Path(ncfile)
    file_name = pathlib.Path(ncfile).name

    # use mocked s3fs
    bucket = "MY_BUCKET"
    s3fs_s3.mkdir(bucket)
    s3fs_s3.put(file_path, bucket)
    s3 = s3fs.S3FileSystem(
        anon=False, version_aware=True, client_kwargs={"endpoint_url": endpoint_uri}
    )

    # test load by standard h5netcdf
    with s3.open(os.path.join("MY_BUCKET", file_name), "rb") as f:
        print("File path", f.path)
        ncfile = h5netcdf.File(f, 'r', invalid_netcdf=True)
        print("File loaded from spoof S3 with h5netcdf:", ncfile)
        print(ncfile["ta"])
    assert "ta" in ncfile

    # test active.load_from_s3
    storage_options = dict(anon=False, version_aware=True,
                           client_kwargs={"endpoint_url": endpoint_uri})
    with load_from_s3(os.path.join("MY_BUCKET", file_name), storage_options) as ac_file:
        print(ac_file)
        assert "ta" in ac_file

    # test loading with Pyfive and passing the Dataset to Active
    with s3.open(os.path.join("MY_BUCKET", file_name), "rb") as f:
        print("File path", f.path)
        pie_ds = pyfive.File(f, 'r')
        print("File loaded from spoof S3 with Pyfive:", pie_ds)
        print("Pyfive dataset:", pie_ds["ta"])
        av = Active(pie_ds["ta"])
        av._method = "min"
        assert av.method([3,444]) == 3
        av_slice_min = av[3:5]
        assert av_slice_min == np.array(249.6583, dtype="float32")
