import os
import numpy as np
import pytest

import activestorage.storage as st


def test_reduce_chunk():
    """Test reduce chunk entirely."""
    rfile = "tests/test_data/cesm2_native.nc"
    offset = 2
    size = 128
    rc = st.reduce_chunk(rfile, offset, size,
                         compression=None, filters=None,
                         missing=[None, 2050, None, None],
                         dtype="i2", shape=(8, 8),
                         order="C", chunk_selection=slice(0, 2, 1),
                         method=np.min)
