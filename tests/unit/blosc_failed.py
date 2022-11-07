import os
import numpy as np
import pytest
import zarr

from activestorage import active_tools as at
from numcodecs import Blosc


def assemble_zarr():
    """Create a test zarr object."""
    compressor = Blosc(cname='zstd', clevel=1, shuffle=Blosc.BITSHUFFLE)
    z = zarr.create((10000, 10000), chunks=(1000, 1000), dtype='i1', order='C',
                    compressor=compressor)

    return z


@pytest.mark.xfail(reason='Issue with BLOSC decompression.')
def test_process_chunk_compressed():
    """Test for processing chunk engine for uncompressed data"""
    z = assemble_zarr()
    z = at.make_an_array_instance_active(z)
    out = np.ones((1, 8))
    cdata = np.ones((1, 2))
    chunk_selection = slice(0, 1, 1)
    out_selection = np.array((0))
    ch = at.as_process_chunk(z,
                             out,
                             cdata,
                             chunk_selection,
                             drop_axes=False,
                             out_is_ndarray=True,
                             fields=None,
                             out_selection=out_selection,
                             partial_read_decode=False)
