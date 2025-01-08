import os
import s3fs
import pathlib
from tempfile import NamedTemporaryFile
import logging
import boto3
import moto
import pyfive
import pytest
from botocore.exceptions import ClientError


def file_upload(upload_file_bucket, file_name, file_path):
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                xml = f.read()
        else:
            logging.error("File '%s' does not exist." % file_path)
            tools.exit_gracefully(botocore.log)
        try:
            conn = boto3.session.Session()
            s3 = conn.resource('s3')
            object = s3.Object(upload_file_bucket, file_name)
            result = object.put(Body=xml)
            res = result.get('ResponseMetadata')
            if res.get('HTTPStatusCode') == 200:
                logging.info('File Uploaded Successfully')
            else:
                logging.info('File Not Uploaded Successfully')
            return res
        except ClientError as e:
            logging.error(e)


def file_load(bucket, file_name):
    conn = boto3.session.Session()
    s3 = conn.resource('s3')
    object = s3.Object(bucket, file_name)
    result = object.get(Range="0=2")
    print("S3 Test mock file:", result)

    ds = pyfive.File(result)
    return ds


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
        # you many need to change 'aws_prof_dev_qa' to be your profile name
        tmp.write(b"""[aws_prof_dev_qa]
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


def test_file_upload(empty_bucket):
    with NamedTemporaryFile() as tmp:
        tmp.write(b'Hi')
        file_name = pathlib.Path(tmp.name).name

        result = file_upload("MY_BUCKET", file_name, tmp.name)
        result2 = file_load("MY_BUCKET", file_name)

        assert result.get('HTTPStatusCode') == 200
