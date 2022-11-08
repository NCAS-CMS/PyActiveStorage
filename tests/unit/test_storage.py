import os
import numpy as np
import pytest

import activestorage.storage as st


def test_reduce_chunk():
    """Test reduce chunk entirely."""
    rfile = "tests/test_data/cesm2_native.nc"
    offset = 2
    size = 128

    # no compression
    rc = st.reduce_chunk(rfile, offset, size,
                         compression=None, filters=None,
                         missing=[None, 2050, None, None],
                         dtype="i2", shape=(8, 8),
                         order="C", chunk_selection=slice(0, 2, 1),
                         method=np.min)
    assert rc[0] == -1
    assert rc[1] == 15

    # with compression
    with pytest.raises(NotImplementedError) as exc:
        rc = st.reduce_chunk(rfile, offset, size,
                             compression="Blosc", filters=None,
                             missing=[None, 2050, None, None],
                             dtype="i2", shape=(8, 8),
                             order="C", chunk_selection=slice(0, 2, 1),
                             method=np.max)
    assert str(exc.value) == "Compression is not yet supported!"

    # with filters
    with pytest.raises(NotImplementedError) as exc:
        rc = st.reduce_chunk(rfile, offset, size,
                             compression=None, filters="filters",
                             missing=[None, 2050, None, None],
                             dtype="i2", shape=(8, 8),
                             order="C", chunk_selection=slice(0, 2, 1),
                             method=np.max)
    assert str(exc.value) == "Filters are not yet supported!"


def test_reduced_chunk_masked_data():
    """Test method with masked data."""
    rfile = "tests/test_data/daily_data_masked.nc"
    offset = 2
    size = 128

    # no compression
    # FIXME TODO this test doesn't work since the implementation is faulty
    rc = st.reduce_chunk(rfile, offset, size,
                         compression=None, filters=None,
                         missing=[999., 999., None, None],
                         dtype="i2", shape=(8, 8),
                         order="C", chunk_selection=slice(0, 2, 1),
                         method=np.mean)
    assert rc[0] == -1
    assert rc[1] == 15
