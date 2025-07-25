import json

import numpy as np
import pyfive
from test_bigger_data import save_cl_file_with_a

from activestorage import reductionist
from activestorage.active import Active, get_missing_attributes, load_from_s3
from activestorage.config import *
from activestorage.hdf2numcodec import decode_filters


class MockActive:

    def __init__(self, f, v):
        if USE_S3:
            self.f = load_from_s3(f)
        else:
            self.f = pyfive.File(f)
        ds = self.f[v]
        self.dtype = np.dtype(ds.dtype)
        self.array = pyfive.indexing.ZarrArrayStub(ds.shape, ds.chunks
                                                   or ds.shape)
        self.missing = get_missing_attributes(ds)
        ds = ds.id
        self.ds = ds

    def __getitem__(self, args):
        if self.ds.filter_pipeline is None:
            compressor, filters = None, None
        else:
            compressor, filters = decode_filters(self.ds.filter_pipeline,
                                                 self.dtype.itemsize, 'a')
        if self.ds.chunks is not None:
            self.ds._get_chunk_addresses()

        indexer = pyfive.indexing.OrthogonalIndexer(args, self.array)
        for chunk_coords, chunk_selection, out_selection in indexer:
            storeinfo = self.ds.get_chunk_info_from_chunk_coord(chunk_coords)
            offset, size = storeinfo.byte_offset, storeinfo.size
            jd = reductionist.build_request_data(
                'a', 'b', 'c', offset, size, compressor, filters, self.missing,
                self.dtype, self.array._chunks, self.ds._order,
                chunk_selection, tuple(range(len(self.array._chunks))))
            js = json.dumps(jd)
        return None


def test_build_request(tmp_path):

    ncfile, v = save_cl_file_with_a(tmp_path), 'cl'
    A = MockActive(ncfile, v)
    x = A[4:5, 1:2]
    # not interested in what is returned, checking that the request builds ok
