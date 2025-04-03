import itertools
import netCDF4
import numpy as np
import pytest

from activestorage.active import Active

def axis_combinations(ndim):
    """Create axes permutations"""
    return [None] + [
        axes
        for n in range(1, ndim + 1)
        for axes in itertools.permutations(range(ndim), n)
    ]

rfile = "tests/test_data/test1.nc"
ncvar ='tas'
ref = netCDF4.Dataset(rfile)[ncvar][...]

@pytest.mark.parametrize(
    "index",
    (
        Ellipsis,
        (slice(6, 7), slice(None), slice(None)),
        (slice(None), slice(0, 64, 3), slice(None)),
        (slice(None), slice(None), slice(0, 128, 4)),
        (slice(6, 7), slice(0, 64, 3), slice(0, 128, 4)),
        (slice(1,11, 2), slice(0, 64, 3), slice(0, 128, 4)),
        (slice(None), [0, 1, 5, 7, 30, 31], slice(None)),
        (slice(None), [0, 1, 5, 7, 30, 31, 50, 51, 53], slice(None)),
    )
)
def test_active_axis_reduction(index):
    """Unit test for class:Active axis combinations."""
    for axis in axis_combinations(ref.ndim):
        for method, numpy_func in zip(
                ("mean", "sum", "min", "max"),
                (np.ma.mean, np.ma.sum, np.ma.min, np.ma.max)
        ):
            print (axis, index, method)

            r = numpy_func(ref[index], axis=axis, keepdims=True)

            active = Active(rfile, ncvar, axis=axis)
            active.method = method
            x  = active[index]

            assert x.shape == r.shape
            assert (x.mask == r.mask).all()
            assert np.ma.allclose(x, r)

            # Test dictionary components output
            # re-add method
            active.components = True
            active.method = method

            rn = np.ma.count(ref[index], axis=axis, keepdims=True)

            x = active[index]

            xn = x["n"]
            assert xn.shape == rn.shape
            assert (xn == rn).all()

            if method == "mean":
                method = "sum"
                r = np.ma.sum(ref[index], axis=axis, keepdims=True)

            x = x[method]
            assert x.shape == r.shape
            assert (x.mask == r.mask).all()
            assert np.ma.allclose(x, r)


def test_active_axis_format_1():
    """Unit test for class:Active axis format."""
    active1 = Active(rfile, ncvar, axis=[0, 2])
    active2 = Active(rfile, ncvar, axis=(-1, -3))

    x1 = active2.mean[...]
    x2 = active2.mean[...]

    assert x1.shape == x2.shape
    assert (x1.mask == x2.mask).all()
    assert np.ma.allclose(x1, x2)


def test_active_axis_format_2():
    """Unit test for class:Active axis format."""
    # Disallow out-of-range axes
    active = Active(rfile, ncvar, axis=(0, 3))
    active.method = "mean"

    with pytest.raises(ValueError):
        active[...]


def test_active_axis_index():
    """Unit test for class:Active axis format."""
    # Disallow reductions when the index drops an axis (i.e. index
    # contains an integer)
    active = Active(rfile, ncvar)
    active.method = "mean"

    with pytest.raises(IndexError):
        active[0]
