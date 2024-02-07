import concurrent.futures
import contextlib
import os
import numpy as np
import pathlib
import urllib

import h5netcdf
import s3fs

#FIXME: Consider using h5py throughout, for more generality
from netCDF4 import Dataset
from zarr.indexing import (
    OrthogonalIndexer,
)
from activestorage.config import *
from activestorage import reductionist
from activestorage.storage import reduce_chunk
from activestorage import netcdf_to_zarr as nz


@contextlib.contextmanager
def load_from_s3(uri, storage_options):
    """
    Load a netCDF4-like object from S3.

    First, set up an S3 filesystem with s3fs.S3FileSystem.
    Then open the uri with this FS -> s3file
    s3file is a File-like object: a memory view but wih all the metadata
    gubbins inside it (no data!)
    calling >> ds = netCDF4.Dataset(s3file) <<
    will throw a FileNotFoundError because the netCDF4 library is always looking for
    a local file, resulting in [Errno 2] No such file or directory:
    '<File-like object S3FileSystem, pyactivestorage/s3_test_bizarre.nc>'
    instead, we use h5netcdf: https://github.com/h5netcdf/h5netcdf
    a Python binder straight to HDF5-netCDF4 interface, that doesn't need a "local" file
    """
    fs = s3fs.S3FileSystem(**storage_options)
    with fs.open(uri, 'rb') as s3file:
        ds = h5netcdf.File(s3file, 'r')
        print(f"Dataset loaded from S3 via h5netcdf: {ds}")
        yield ds


class Active:
    """ 
    Instantiates an interface to active storage which contains either zarr files
    or HDF5 (NetCDF4) files.

    This is Verson 1 which simply provides support for standard read operations, but done via
    explicit reads within this class rather than within the underlying format libraries.
    
    Version 2 will add methods for actual active storage.

    """
    def __new__(cls, *args, **kwargs):
        """Store reduction methods."""
        instance = super().__new__(cls)
        instance._methods = {
            "min": np.min,
            "max": np.max,
            "sum": np.sum,
            # For the unweighted mean we calulate the sum and divide
            # by the number of non-missing elements
            "mean": np.sum,
        }
        return instance

    def __init__(self, uri, ncvar, missing_value=None, _FillValue=None, valid_min=None, valid_max=None, max_threads=100, storage_options=None):
        """
        Instantiate with a NetCDF4 dataset and the variable of interest within that file.
        (We need the variable, because we need variable specific metadata from within that
        file, however, if that information is available at instantiation, it can be provided
        using keywords and avoid a metadata read.)
        """
        # Assume NetCDF4 for now
        self.uri = uri
        if self.uri is None:
            raise ValueError(f"Must use a valid file for uri. Got {self.uri}")
        storage_type = urllib.parse.urlparse(uri).scheme
        self.storage_type = storage_type
        if not os.path.isfile(self.uri) and not self.storage_type:
            raise ValueError(f"Must use existing file for uri. {self.uri} not found")
        if storage_options is None:
            stroage_options = {}

        self.storage_options = storage_options
        
        self.ncvar = ncvar
        if self.ncvar is None:
            raise ValueError("Must set a netCDF variable name to slice")
        self.zds = None

        self._version = 1
        self._components = False
        self._method = None
        self._lock = False
        self._max_threads = max_threads
                   
        # obtain metadata, using netcdf4_python for now
        # FIXME: There is an outstanding issue with ._FilLValue to be handled.
        # If the user actually wrote the data with no fill value, or the
        # default fill value is in play, then this might go wrong.
        if storage_type == "s3":
            with load_from_s3(uri, storage_options) as _ds:
                ds = _ds
        else:
            ds = Dataset(uri)
            
        try:
            ds_var = ds[ncvar]
        except IndexError as exc:
            print(f"Dataset {ds} does not contain ncvar {ncvar!r}.")
            raise exc

        # FIXME: We do not get the correct byte order on the Zarr Array's dtype
        # when using S3, so capture it here.
        self._dtype = ds_var.dtype

        if (missing_value, _FillValue, valid_min, valid_max) == (None, None, None, None):
            if isinstance(ds, Dataset):
                self._missing = getattr(ds_var, 'missing_value', None)
                self._fillvalue = getattr(ds_var, '_FillValue', None)
                # could be fill_value set as netCDF4 attr
                if self._fillvalue is None:
                    self._fillvalue = getattr(ds_var, 'fill_value', None)
                valid_min = getattr(ds_var, 'valid_min', None)
                valid_max = getattr(ds_var, 'valid_max', None)
                valid_range = getattr(ds_var, 'valid_range', None)
                add_offset =  getattr(ds_var, 'add_offset', None)
                scale_factor =  getattr(ds_var, 'scale_factor', None)
            elif storage_type == "s3":
                self._missing = ds_var.attrs.get('missing_value')
                self._fillvalue = ds_var.attrs.get('_FillValue')
                # could be fill_value set as netCDF4 attr
                if self._fillvalue is None:
                    self._fillvalue = ds_var.attrs.get('fill_value')
                valid_min = ds_var.attrs.get('valid_min')
                valid_max = ds_var.attrs.get('valid_max')
                valid_range = ds_var.attrs.get('valid_range')
                add_offset =  ds_var.attrs.get('add_offset')
                scale_factor =  ds_var.attrs.get('scale_factor')

            if valid_max is not None or valid_min is not None:
                if valid_range is not None:
                    raise ValueError(
                        "Invalid combination in the file of valid_min, "
                        "valid_max, valid_range: "
                        f"{valid_min}, {valid_max}, {valid_range}"
                    )                
                valid_range = (valid_min, valid_max)
            elif valid_range is None:
                valid_range = (None, None)
                
            self._valid_min, self._valid_max = valid_range

        else:
            self._missing = missing_value
            self._fillvalue = _FillValue
            self._valid_min = valid_min
            self._valid_max = valid_max

        ds.close()

    def __getitem__(self, index):
        """ 
        Provides support for a standard get item.
        """
        # In version one this is done by explicitly looping over each chunk in the file
        # and returning the requested slice ourselves. In version 2, we can pass this
        # through to the default method.
        ncvar = self.ncvar

        if self.method is None and self._version == 0:
            # No active operation
            lock = self.lock
            if lock:
                lock.acquire()
                
           
            if self.storage_type == "s3":
                with load_from_s3(self.uri, storage_options) as nc:
                    data = nc[ncvar][index]
                    # h5netcdf doesn't return masked arrays.
                    if self._fillvalue:
                        data = np.ma.masked_equal(data, self._fillvalue)
                    if self._missing:
                        data = np.ma.masked_equal(data, self._missing)
                    if self._valid_max:
                        data = np.ma.masked_greater(data, self._valid_max)
                    if self._valid_min:
                        data = np.ma.masked_less(data, self._valid_min)
            else:
                nc = Dataset(self.uri)
                data = nc[ncvar][index]
                nc.close()
                
            if lock:
                lock.release()

            return data
        elif self._version == 1:
            return self._via_kerchunk(index)
        elif self._version  == 2:
            # No active operation either
            lock = self.lock
            if lock:
                lock.acquire()

            data = self._via_kerchunk(index)

            if lock:
                lock.release()

            return data
        else:
            raise ValueError(f'Version {self._version} not supported')

    @property
    def components(self):
        """Return or set the components flag.

        If True and `method` is not `None` then return the processed
        result in a dictionary that includes a processed value and the
        sample size, from which the final result can be calculated.

        """
        return self._components

    @components.setter
    def components(self, value):
        self._components = bool(value)

    @property
    def method(self):
        """Return or set the active method.

        The active method to apply when retrieving a subspace of the
        data. By default the data is returned unprocessed. Valid
        methods are:

        ==========  ==================================================
        *method*    Description
        ==========  ==================================================
        ``'min'``   The minumum

        ``'max'``   The maximum

        ``'mean'``  The unweighted mean

        ``'sum'``   The unweighted sum
        ==========  ==================================================

        """
        return self._methods.get(self._method)

    @method.setter
    def method(self, value):
        if value is not None and value not in self._methods:
            raise ValueError(f"Bad 'method': {value}. Choose from min/max/mean/sum.")

        self._method = value

    @property
    def ncvar(self):
        """Return or set the netCDF variable name."""
        return self._ncvar

    @ncvar.setter
    def ncvar(self, value):
        self._ncvar = value

    @property
    def lock(self):
        """Return or set a lock that prevents concurrent file reads when accessing the data locally.

        The lock is either a `threading.Lock` instance, an object with
        same API and functionality (such as
        `dask.utils.SerializableLock`), or is `False` if no lock is
        required.

        To be effective, the same lock instance must be used across
        all process threads.

        """
        return self._lock

    @lock.setter
    def lock(self, value):
        if not value:
            value = False
            
        self._lock = value

    def _get_active(self, method, *args):
        """ 
        *args defines a slice of data. This method loops over each of the chunks
        necessary to extract the parts of the slice, and asks the active storage 
        to apply the method to each part. It then applies the method to 
        the partial results and returns a value is if  method had been applied to
        an array returned via getitem.
        """
        raise NotImplementedError

    def _via_kerchunk(self, index):
        """ 
        The objective is to use kerchunk to read the slices ourselves. 
        """
        # FIXME: Order of calls is hardcoded'
        if self.zds is None:
            print(f"Kerchunking file {self.uri} with variable "
                  f"{self.ncvar} for storage type {self.storage_type}")
            ds = nz.load_netcdf_zarr_generic(self.uri,
                                             self.ncvar,
                                             self.storage_type,
                                             self.storage_options)
            # The following is a hangove from exploration
            # and is needed if using the original doing it ourselves
            # self.zds = make_an_array_instance_active(ds)
            self.zds = ds

        return self._get_selection(index)

    def _get_selection(self, *args):
        """ 
        First we need to convert the selection into chunk coordinates,
        steps etc, via the Zarr machinery, then we get everything else we can
        from zarr and friends and use simple dictionaries and tuples, then
        we can go to the storage layer with no zarr.
        """
        compressor = self.zds._compressor
        filters = self.zds._filters

        missing = self._fillvalue, self._missing, self._valid_min, self._valid_max

        indexer = OrthogonalIndexer(*args, self.zds)
        out_shape = indexer.shape
        out_dtype = self.zds._dtype
        stripped_indexer = [(a, b, c) for a,b,c in indexer]
        drop_axes = indexer.drop_axes  # not sure what this does and why, yet.

        # yes this next line is bordering on voodoo ...
        # this returns a nested dictionary with the full file FS reference
        # ie all the gubbins: chunks, data structure, types, etc
        # if using zarr<=2.13.3 call with _mutable_mapping ie
        # fsref = self.zds.chunk_store._mutable_mapping.fs.references 
        fsref = self.zds.chunk_store.fs.references

        return self._from_storage(stripped_indexer, drop_axes, out_shape,
                                  out_dtype, compressor, filters, missing, fsref)

    def _from_storage(self, stripped_indexer, drop_axes, out_shape, out_dtype,
                      compressor, filters, missing, fsref):
        method = self.method
        if method is not None:
            out = []
            counts = []
        else:
            out = np.empty(out_shape, dtype=out_dtype, order=self.zds._order)
            counts = None  # should never get touched with no method!

        # Create a shared session object.
        if self.storage_type == "s3":
            session = reductionist.get_session(S3_ACCESS_KEY, S3_SECRET_KEY,
                                               S3_ACTIVE_STORAGE_CACERT)
        else:
            session = None

        # Process storage chunks using a thread pool.
        with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_threads) as executor:
            futures = []
            # Submit chunks for processing.
            for chunk_coords, chunk_selection, out_selection in stripped_indexer:
                future = executor.submit(
                    self._process_chunk,
                    session, fsref, chunk_coords, chunk_selection,
                    counts, out_selection,
                    compressor, filters, missing,
                    drop_axes=drop_axes)
                futures.append(future)
            # Wait for completion.
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                except Exception as exc:
                    raise
                else:
                    if method is not None:
                        result, count = result
                        out.append(result)
                        counts.append(count)
                    else:
                        # store selected data in output
                        result, selection = result
                        out[selection] = result

        if method is not None:
            # Apply the method (again) to aggregate the result
            out = method(out)
            shape1 = (1,) * len(out_shape)
                
            if self._components:
                # Return a dictionary of components containing the
                # reduced data and the sample size ('n'). (Rationale:
                # cf-python needs the sample size for all reductions;
                # see the 'mtol' parameter of cf.Field.collapse.)
                #
                # Note that in all components must always have the
                # same number of dimensions as the original array,
                # i.e. 'drop_axes' is always considered False,
                # regardless of its setting. (Rationale: dask
                # reductions require the per-dask-chunk partial
                # reductions to retain these dimensions so that
                # partial results can be concatenated correctly.)
                out = out.reshape(shape1)

                n = np.sum(counts).reshape(shape1)
                if self._method == "mean":
                    # For the average, the returned component is
                    # "sum", not "mean"
                    out = {"sum": out, "n": n}
                else:
                    out = {self._method: out, "n": n}
            else:
                # Return the reduced data as a numpy array. For most
                # methods the data is already in this form.
                if self._method == "mean":
                    # For the average, it is actually the sum that has
                    # been created, so we need to divide by the sample
                    # size.
                    out = out / np.sum(counts).reshape(shape1)

        return out

    def _process_chunk(self, session, fsref, chunk_coords, chunk_selection, counts,
                       out_selection, compressor, filters, missing, 
                       drop_axes=None):
        """
        Obtain part or whole of a chunk.

        This is done by taking binary data from storage and filling
        the output array.

        Note the need to use counts for some methods

        """
        coord = '.'.join([str(c) for c in chunk_coords])
        key = f"{self.ncvar}/{coord}"
        rfile, offset, size = tuple(fsref[key])

        if self.storage_type == "s3":
            parsed_url = urllib.parse.urlparse(rfile)
            bucket = parsed_url.netloc
            object = parsed_url.path
            # FIXME: We do not get the correct byte order on the Zarr Array's dtype
            # when using S3, so use the value captured earlier.
            dtype = self._dtype
            tmp, count = reductionist.reduce_chunk(session, S3_ACTIVE_STORAGE_URL,
                                                   S3_URL,
                                                   bucket, object, offset,
                                                   size, compressor, filters,
                                                   missing, dtype,
                                                   self.zds._chunks,
                                                   self.zds._order,
                                                   chunk_selection,
                                                   operation=self._method)
        else:
            # note there is an ongoing discussion about this interface, and what it returns
            # see https://github.com/valeriupredoi/PyActiveStorage/issues/33
            # so neither the returned data or the interface should be considered stable
            # although we will version changes.
            tmp, count = reduce_chunk(rfile, offset, size, compressor, filters,
                                      missing, self.zds._dtype,
                                      self.zds._chunks, self.zds._order,
                                      chunk_selection, method=self.method)

        if self.method is not None:
            return tmp, count
        else:
            if drop_axes:
                tmp = np.squeeze(tmp, axis=drop_axes)
            return tmp, out_selection
