from copy import deepcopy

import numpy as np
import netCDF4

from activestorage import Active


class NetCDFArray:
    """An array stored in a netCDF file.

    Supports active storage operations.

    This object has been derived from `cfdm.NetCDFArray`. In order to
    make the code easier to understand, it has been simplified from
    its source, e.g. it doesn't support netCDF groups and doesn't
    properly deal with string data types. This functionality would all
    be reinstated in an operational version.

    """

    def __init__(self, filename, ncvar):
        """Initialisation.

        :Parameters:
        
            filename: `str`
                The URI of the dataset.

            ncvar: `str`
                The name of a variable in the dataset.

        """
        self.filename = filename
        self.ncvar = ncvar

        nc = netCDF4.Dataset(self.filename, "r")
        v = nc.variables[self.ncvar]
        
        self.shape = v.shape
        self.dtype = v.dtype
        self.ndim = v.ndim
        self.size = v.size

        nc.close()

    def __getitem__(self, indices):
        if self.active_storage_op:
            # Active storage read. Returns a dictionary.
            active = Active(self.filename, self.ncvar)
            active.method = self.active_storage_op
            active.components = True

            return active[indices]

        # Normal read by local client. Returns a numpy array.
        #
        # In production code groups, masks, string types, etc. will
        # need to be accounted for here.
        try:
            nc = netCDF4.Dataset(self.filename, "r")
        except RuntimeError as error:
            raise RuntimeError(f"{error}: {self.filename}")

        data = nc.variables[self.ncvar][indices]
        nc.close()

        return data

    def __repr__(self):
        return f"<{self.__class__.__name__}{self.shape}: {self}>"

    def __str__(self):
        return f"file={self.filename} {self.ncvar}"

    def _active_chunk_functions(self):
        return {
            "min": self.active_min,
            "max": self.active_max,
            "mean": self.active_mean,
        }

    @property
    def active_storage_op(self):
        return getattr(self, "_active_storage_op", None)

    @active_storage_op.setter
    def active_storage_op(self, value):
        self._active_storage_op = value

    @property
    def op_axis(self):
        return getattr(self, "_op_axis", None)

    @op_axis.setter
    def op_axis(self, value):
        self._op_axis = value

    @staticmethod
    def active_min(a, **kwargs):
        """Chunk calculations for the minimum.

        Assumes that the calculations have already been done,
        i.e. that *a* is already the minimum.

        This function is intended to be passed in to
        `dask.array.reduction()` as the ``chunk`` parameter. Its
        return signature must be the same as the non-active chunks
        function that it is replacing.

        :Parameters:

            a: `dict`

        :Returns:

            `numpy.ndarray`
                Currently set up to replace `dask.array.chunk.min`.

        """
        return a["min"]

    @staticmethod
    def active_max(a, **kwargs):
        """Chunk calculations for the maximum.

        Assumes that the calculations have already been done,
        i.e. that *a* is already the maximum.

        This function is intended to be passed in to
        `dask.array.reduction()` as the ``chunk`` parameter. Its
        return signature must be consistent with that expected by the
        functions of the ``aggregate`` and ``combine`` parameters.

        :Parameters:

            a: `dict`

        :Returns:

            `numpy.ndarray`
                Currently set up to replace `dask.array.chunk.max`.

        """
        return a["max"]

    @staticmethod
    def active_mean(a, **kwargs):
        """Chunk calculations for the mean.

        Assumes that the calculations have already been done,
        i.e. that *a* is already the mean.

        This function is intended to be passed in to
        `dask.array.reduction()` as the ``chunk`` parameter. Its
        return signature must be the same as the non-active chunks
        function that it is replacing.

        :Parameters:

            a: `dict`

        :Returns:

            `dict`
                Currently set up to replace
                `dask.array.reductions.mean_chunk`

        """
        return {"n": a["n"], "total": a["sum"]}

    def set_active_storage_op(self, op, axis=None):
        if op not in self._active_chunk_functions():
            raise ValueError(f"Invalid active storage operation: {op!r}")

        a = self.copy()
        a.active_storage_op = op
        a.op_axis = axis
        return a

    def get_active_chunk_function(self):
        try:
            return self._active_chunk_functions()[self.active_storage_op]
        except KeyError:
            raise ValueError("no active storage operation has been set")

    def copy(self):
        return deepcopy(self)


if __name__ == "__main__":
    import os

    import dask
    import dask.array as da

    try:
        # Check that we're using the modified dask.
        #
        # Note that it is not necessary for the 'actify` unction to be
        # inside the dask library. It can be moved to the client code
        # that calls dask. This has not been done in this client code
        # for clarity of demonstration of the client-side approach.
        dask.array.reductions.actify
    except AttributeError:
        raise AttributeError(
            "No 'dask.array.reductions.actify' function.\n"
            f"dask path: {dask.__file__}\n"
            f"PYTHONPATH={os.environ.get('PYTHONPATH', '')}"
        )

    # ----------------------------------------------------------------
    # Get the data as a lazy array with active capabilities
    # ----------------------------------------------------------------
    f = NetCDFArray(filename="file.nc", ncvar="q")

    # ----------------------------------------------------------------
    # Get the same data as an in-memory numpy array (with no active
    # capabilities)
    # ----------------------------------------------------------------
    nc = netCDF4.Dataset("file.nc", "r")
    x = nc.variables["q"][...]
    nc.close()

    # ----------------------------------------------------------------
    # Instantiate dask arrays from 'f' and 'x', each with the the same
    # arbitrary distribution of dask chunks.
    # ----------------------------------------------------------------
    dask_chunks = (3, 4)
    df = da.from_array(f, chunks=dask_chunks)
    dx = da.from_array(x, chunks=dask_chunks)

    # ----------------------------------------------------------------
    # Compare the results of some active operations with their normal
    # counterparts
    # ----------------------------------------------------------------
    g = da.max(df)
    y = da.max(dx)
    print("\nActive max(a) =", g.compute())
    print("Normal max(a) =", y.compute())
    assert g.compute() == y.compute()
    g.visualize(filename="active_max.png")
    y.visualize(filename="normal_max.png")

    g = da.mean(df)
    y = da.mean(dx)
    print("\nActive mean(a) =", g.compute())
    print("Normal mean(a) =", y.compute())
    assert g.compute() == y.compute()
    g.visualize(filename="active_mean.png")
    y.visualize(filename="normal_mean.png")

    # da.sum has been "actified", but NetCDFArray does not support
    # "sum" as an active operation
    g = da.sum(df)
    y = da.sum(dx)
    print("\nNon-active sum(a) =", g.compute())
    print("    Normal sum(a) =", y.compute())
    assert g.compute() == y.compute()
    g.visualize(filename="non_active_sum.png")
    y.visualize(filename="normal_sum.png")

    g = da.max(df) + df
    y = da.max(dx) + dx
    print("\nActive max(a) + a =", g.compute())
    print("Normal max(a) + a =", y.compute())
    assert (g.compute() == y.compute()).all()
    g.visualize(filename="active_max+a.png")
    y.visualize(filename="normal_max+a.png")

    g = da.sum(da.max(df) + df)
    y = da.sum(da.max(dx) + dx)
    print("\nActive sum(max(a) + a) =", g.compute())
    print("Normal sum(max(a) + a) =", y.compute())
    assert g.compute() == y.compute()
    g.visualize(filename="active_sum_max+a.png")
    y.visualize(filename="normal_sum_max+a.png")
