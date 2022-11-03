"""All zarr imports are done at test instance level."""
import os
import numpy as np
import pytest

from activestorage import active_tools as at


def assemble_zarr():
    """Create a test zarr object."""
    import zarr
    from numcodecs import Blosc
    compressor = Blosc(cname='zstd', clevel=1, shuffle=Blosc.BITSHUFFLE)
    z = zarr.create((10000, 10000), chunks=(1000, 1000), dtype='i1', order='C',
                    compressor=compressor)

    return z


def test_as_get_orthogonal_selection():
    """Test Zarr Orthogonal selection."""
    z = assemble_zarr()
    z._cache_metadata = None
    selection = (slice(0, 2, 1), slice(4, 6, 1))
    sel = at.as_get_orthogonal_selection(z, selection=selection, out=None,
                                         fields=None)
    expected_shape = (2, 2)
    np.testing.assert_array_equal(sel.shape, expected_shape)
    expected_elem = [0, 0]
    np.testing.assert_array_equal(sel[0], expected_elem)


def test_as_get_selection():
    """Test chunk iterator."""
    from zarr.indexing import OrthogonalIndexer
    z = assemble_zarr()
    selection = (slice(0, 2, 1), slice(4, 6, 1))
    indexer = OrthogonalIndexer(selection, z)
    ch = at.as_get_selection(z, indexer, out=None,
                             fields=None)
    np.testing.assert_array_equal(ch[0][0], [0, 0])
    np.testing.assert_array_equal(ch[1][0], None)
    np.testing.assert_array_equal(ch[2][0], (0, 0))


def test_as_chunk_getitem():
    """Test chunk get item."""
    z = assemble_zarr()
    z = at.make_an_array_instance_active(z)
    chunk_coords = (0, 3)
    chunk_selection = [slice(0, 2, 1)]
    out = np.array((2, 2, 2))
    out_selection = slice(0, 2, 1)
    ch = at.as_chunk_getitem(z, chunk_coords, chunk_selection, out, out_selection,
                             drop_axes=None, fields=None)
    np.testing.assert_array_equal(ch[0], (1000, 1000))
    np.testing.assert_array_equal(ch[1], [slice(0, 2, 1)])
    assert list(ch[2]) == [(0, 2000, (slice(0, 2, 1),))]
