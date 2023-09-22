import s3fs

from kerchunk.hdf import SingleHdf5ToZarr


def test_s3_SingleHdf5ToZarr(s3_file):
    """Check Kerchunk's SingleHdf5ToZarr when S3."""
    s3_file = 
    fs = s3fs.S3FileSystem(key=S3_ACCESS_KEY,
                           secret=S3_SECRET_KEY,
                           client_kwargs={'endpoint_url': S3_URL},
                           default_fill_cache=False,
                           default_cache_type="none"
    )
    with fs.open(s3_file, 'rb') as s3file:
        h5chunks = SingleHdf5ToZarr(s3file, s3_file,
                                    inline_threshold=0)


def test_local_SingleHdf5ToZarr():
    """Check Kerchunk's SingleHdf5ToZarr when NO S3."""
    local_file = 
    fs = fsspec.filesystem('')
    with fs.open(local_file, 'rb') as localfile:
        h5chunks = SingleHdf5ToZarr(localfile, local_file,
                                    inline_threshold=0)
