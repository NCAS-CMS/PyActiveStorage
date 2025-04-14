import concurrent.futures
import os
import fsspec
import numpy as np
import pathlib
import urllib
import pyfive
import requests
import s3fs
import time

from pathlib import Path
from pyfive.h5d import StoreInfo
from typing import Optional

from activestorage.config import *
from activestorage import reductionist
from activestorage.storage import reduce_chunk, reduce_opens3_chunk
from activestorage.hdf2numcodec import decode_filters


def return_storage_type(uri):
    """
    Extract the gateway-protocol to infer what type of storage
    """
    try:
        resp = requests.head(uri)
    except requests.exceptions.MissingSchema:  # eg local file
        return
    except requests.exceptions.InvalidSchema:  # eg Minio file s3://pyactivestorage/common_cl_a.nc
        if not uri.startswith("s3:"):
            return
        else:
            return "s3"
    except requests.exceptions.ConnectionError as exc:  # eg invalid link or offline
        print(exc)
        return
    response = resp.headers

    # https files on NGINX don't have "gateway-protocol" key
    if "gateway-protocol" in response:
        if response["gateway-protocol"] == "s3":
            print("Gateway protocol indicates S3 storage.")
            return "s3"
    else:
        return "https"


def load_from_s3(uri, storage_options=None):
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

    storage_options: kwarg dict containing S3 credentials passed straight to Active
    """
    if storage_options is None:  # use pre-configured S3 credentials
        fs = s3fs.S3FileSystem(key=S3_ACCESS_KEY,  # eg "minioadmin" for Minio
                               secret=S3_SECRET_KEY,  # eg "minioadmin" for Minio
                               client_kwargs={'endpoint_url': S3_URL})  # eg "http://localhost:9000" for Minio
    else:
        fs = s3fs.S3FileSystem(**storage_options)  # use passed-in dictionary
   
    t1=time.time()   
    s3file = fs.open(uri, 'rb')
    t2=time.time()
    ds = pyfive.File(s3file)
    t3=time.time()
    print(f"Dataset loaded from S3 with s3fs and Pyfive: {uri} ({t2-t1:.2},{t3-t2:.2})")
    return ds


def load_from_https(uri):
    """
    Load a pyfive.high_level.Dataset from a
    netCDF4 file on an https server (NGINX).
    """
    #TODO need to test if NGINX server behind https://
    fs = fsspec.filesystem('http')
    http_file = fs.open(uri, 'rb')
    ds = pyfive.File(http_file)
    print(f"Dataset loaded from https with Pyfive: {uri}")
    return ds


def get_missing_attributes(ds):
    """" 
    Load all the missing attributes we need from a netcdf file
    """

    def hfix(x):
        '''
        return item if single element list/array
        see https://github.com/h5netcdf/h5netcdf/issues/116
        '''
        if x is None:
            return x
        if not np.isscalar(x) and len(x) == 1:
            return x[0]
        return x

    _FillValue = hfix(ds.attrs.get('_FillValue'))
    missing_value = ds.attrs.get('missing_value')
    valid_min = hfix(ds.attrs.get('valid_min'))
    valid_max = hfix(ds.attrs.get('valid_max'))
    valid_range = hfix(ds.attrs.get('valid_range'))
    if valid_max is not None or valid_min is not None:
        if valid_range is not None:
            raise ValueError(
                "Invalid combination in the file of valid_min, "
                "valid_max, valid_range: "
                f"{valid_min}, {valid_max}, {valid_range}"
            )
    elif valid_range is not None:            
        valid_min, valid_max = valid_range
    
    return _FillValue, missing_value, valid_min, valid_max

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
            "min": np.ma.min,
            "max": np.ma.max,
            "sum": np.ma.sum,
            # For the unweighted mean we calulate the sum and divide
            # by the number of non-missing elements
            "mean": np.ma.sum,
        }
        return instance

    def __init__(
        self,
        dataset: Optional[str | Path | object] ,
        ncvar: str = None,
        axis: tuple = None,
        storage_type: str = None,
        max_threads: int = 100,
        storage_options: dict = None,
        active_storage_url: str = None
    ) -> None:
        """
        Instantiate with a NetCDF4 dataset URI and the variable of interest within that file.
        (We need the variable, because we need variable specific metadata from within that
        file, however, if that information is available at instantiation, it can be provided
        using keywords and avoid a metadata read.)

        :param storage_options: s3fs.S3FileSystem options
        :param active_storage_url: Reductionist server URL
        """
        self.ds = None
        input_variable = False
        if dataset is None:
            raise ValueError(f"Must use a valid file name or variable object for dataset. Got {dataset!r}")
        if isinstance(dataset, Path) and not dataset.exists():
            raise ValueError(f"Path to input file {dataset!r} does not exist.")
        if not isinstance(dataset, Path) and not isinstance(dataset, str):
            print(f"Treating input {dataset} as variable object.")
            if not type(dataset) is pyfive.high_level.Dataset:
                raise TypeError(f"Variable object dataset can only be pyfive.high_level.Dataset. Got {dataset!r}")
            input_variable = True
            self.ds = dataset
        self.uri = dataset

        # determine the storage_type
        # based on what we have available
        if not storage_type:
            if not input_variable:
                check_uri = self.uri
            else:
                check_uri = self.ds.id._filename

            # "special" case when we have to deal
            # with storage_options['client_kwargs']["endpoint_url"]
            if storage_options is not None and 'client_kwargs' in storage_options:
                if "endpoint_url" in storage_options['client_kwargs']:
                    base_url = storage_options['client_kwargs']["endpoint_url"]
                    if not input_variable:
                        check_uri = os.path.join(base_url, self.uri)
                    else:
                        check_uri = os.path.join(base_url, self.ds.id._filename)
            storage_type = return_storage_type(check_uri)

        # still allow for a passable storage_type
        # for special cases eg "special-POSIX" ie DDN
        if not storage_type and storage_options is not None:
            storage_type = urllib.parse.urlparse(dataset).scheme
        self.storage_type = storage_type

        # set correct filename attr
        if input_variable and not self.storage_type:
            self.filename = self.ds
        elif input_variable and self.storage_type == "s3":
            self.filename = self.ds.id._filename
        elif input_variable and self.storage_type == "https":
            self.filename = self.ds

        # get storage_options
        self.storage_options = storage_options
        self.active_storage_url = active_storage_url

        # basic check on file
        if not input_variable:
            if not os.path.isfile(self.uri) and not self.storage_type:
                raise ValueError(f"Must use existing file for uri. {self.uri} not found")

        self.ncvar = ncvar
        if self.ncvar is None and not input_variable:
            raise ValueError("Must set a netCDF variable name to slice")

        # Parse axis (note, if axis is None then we'll work out how
        # many dimensions there are at the time of an active
        # __getitem__ call).
        if axis is not None:
            if isinstance(axis, int):
                axis = (axis,)
            else:
                axis = tuple(axis)

        self._axis = axis
        self._version = 1
        self._components = False
        self._method = None
        self._max_threads = max_threads
        self.missing = None
        self.data_read = 0

    def __load_nc_file(self):
        """
        Get the netcdf file and its b-tree.

        This private method is used only if the input to Active
        is not a pyfive.high_level.Dataset object. In that case,
        any file opening is skipped, and ncvar is not used. The
        Dataset object will have already contained the b-tree,
        and `_filename` attribute.
        """
        ncvar = self.ncvar
        if self.storage_type is None:
            nc = pyfive.File(self.uri)
        elif self.storage_type == "s3":
            nc = load_from_s3(self.uri, self.storage_options)
        elif self.storage_type == "https":
            nc = load_from_https(self.uri)
        self.filename = self.uri
        self.ds = nc[ncvar]

    def __get_missing_attributes(self):
        if self.ds is None:
            self.__load_nc_file()
        return get_missing_attributes(self.ds)

    def __getitem__(self, index):
        """ 
        Provides support for a standard get item.
        #FIXME-BNL: Why is the argument index?
        """
        if self.ds is None:
            self.__load_nc_file()
        
        self.missing = self.__get_missing_attributes()

        self.data_read = 0
        
        if self.method is None and self._version == 0:
        
            # No active operation
            return self._get_vanilla(index)

        elif self._version == 1:

            #FIXME: is the difference between version 1 and 2 still honoured?
            return self._get_selection(index)
        
        elif self._version  == 2:

            return self._get_selection(index)
          
        else:
            raise ValueError(f'Version {self._version} not supported')

    def _get_vanilla(self, index):
        """ 
        Get the data without any active operation
        """
        data = self.ds[index]    
        data = self._mask_data(data)
        return data

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
    def mean(self):
        self._method =  "mean"
        return self

    @property
    def min(self):
        self._method =  "min"
        return self

    @property
    def max(self):
        self._method =  "max"
        return self

    @property
    def ncvar(self):
        """Return or set the netCDF variable name."""
        return self._ncvar

    @ncvar.setter
    def ncvar(self, value):
        self._ncvar = value

    def _get_active(self, method, *args):
        """ 
        *args defines a slice of data. This method loops over each of the chunks
        necessary to extract the parts of the slice, and asks the active storage 
        to apply the method to each part. It then applies the method to 
        the partial results and returns a value is if  method had been applied to
        an array returned via getitem.
        """
        raise NotImplementedError

    def _get_selection(self, *args):
        """ 
        At this point we have a Dataset object, but all the important information about
        how to use it is in the attribute DataoobjectDataset class. Here we gather 
        metadata from the dataset instance and then continue with the dataobjects instance.
        """

        # stick this here for later, to discuss with David
        keepdims = True

        name = self.ds.name
        dtype = np.dtype(self.ds.dtype)
        array = pyfive.indexing.ZarrArrayStub(self.ds.shape, self.ds.chunks)
        ds = self.ds.id
        
        if self._axis is None:
            # 'axis' is None, so work out how many dimensions there
            # are from the variable.
            self._axis = tuple(range(len(ds.shape)))

        if ds.filter_pipeline is None:
            compressor, filters = None, None
        else:
            compressor, filters = decode_filters(ds.filter_pipeline , dtype.itemsize, name)
        
        indexer = pyfive.indexing.OrthogonalIndexer(*args, array)
        out_shape = indexer.shape
        #stripped_indexer = [(a, b, c) for a,b,c in indexer]
        drop_axes = indexer.drop_axes and keepdims

        # we use array._chunks rather than ds.chunks, as the latter is
        # none in the case of unchunked data, and we need to tell the
        # storage the array dimensions in this case.
        return self._from_storage(ds, indexer, array._chunks, out_shape, dtype, compressor, filters, drop_axes, self._axis)

    def _from_storage(self, ds, indexer, chunks, out_shape, out_dtype, compressor, filters, drop_axes, axis):
        method = self.method

        # Whether or not we need to store reduction counts
        need_counts = self.components or self._method == "mean"
        # but never when we don't have a statistical method
        if self.components and self._method is None:
            raise ValueError("Setting components to True for None statistical method.")

        if method is not None:
            # Get the number of chunks per axis
            nchunks = []
            dim_indexers = indexer.dim_indexers
            for i, d in enumerate(dim_indexers):
                try:
                    nchunks.append(d.nchunks)
                except AttributeError:
                    # If 'd' doesn't have an 'nchunks' attribute then
                    # it must be for an integer index that results in
                    # a droped axis.
                    raise IndexError(
                        "Can't do an active reduction when the index for "
                        f"axis {i!r} drops the axis."
                    )

            # Replace the size of each reduced axis with the total
            # number of chunks along that axis
            out_shape = list(out_shape)
            for i in axis:
                try:
                    out_shape[i] = nchunks[i]
                except IndexError:
                    raise ValueError(
                        "Can't do an active reduction for an "
                        f"out-of-range axis: {i!r}"
                    )

            out = np.ma.empty(out_shape, dtype=out_dtype, order=ds._order)
            out.mask = True
            if need_counts:
                counts = np.ma.empty(out_shape, dtype='int64', order=ds._order)
                counts.mask = True
        else:
            out = np.ma.empty(out_shape, dtype=out_dtype, order=ds._order)

        # Create a shared session object.
        if self.storage_type == "s3" and self._version==2:
            if self.storage_options is not None:
                key, secret = None, None
                if "key" in self.storage_options:
                    key = self.storage_options["key"]
                if "secret" in self.storage_options:
                    secret = self.storage_options["secret"]
                if key and secret:
                    session = reductionist.get_session(key, secret,
                                                       S3_ACTIVE_STORAGE_CACERT)
                else:
                    session = reductionist.get_session(S3_ACCESS_KEY, S3_SECRET_KEY,
                                                       S3_ACTIVE_STORAGE_CACERT)
            else:
                session = reductionist.get_session(S3_ACCESS_KEY, S3_SECRET_KEY,
                                                   S3_ACTIVE_STORAGE_CACERT)
        else:
            session = None

        # Process storage chunks using a thread pool.
        chunk_count = 0
        t1 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_threads) as executor:
            futures = []
            # Submit chunks for processing.
            for chunk_coords, chunk_selection, out_selection in indexer:
                future = executor.submit(
                    self._process_chunk,
                    session,  ds, chunks, chunk_coords, chunk_selection,
                    out_selection, compressor, filters, drop_axes=drop_axes)
                futures.append(future)

            # Wait for completion.
            for future in concurrent.futures.as_completed(futures):
                try:
                    result, count, out_selection = future.result()
                except Exception as exc:
                    raise

                chunk_count += 1

                # Store the selected data
                out[out_selection] = result

                # Store the counts for the selected data
                if need_counts:
                    counts[out_selection] = count

        if method is not None:
            # Apply the method (again) to aggregate the result along
            # the reduction axes
            out = method(out, axis=axis, keepdims=True)

            # Aggregate the counts along the reduction axes
            if need_counts:
                n = np.ma.sum(counts, axis=axis, keepdims=True)

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
                    #
                    # Note: It's OK if an element of 'n' is zero,
                    #       because it will, by definition, correspond
                    #       to a masked value in 'out'.
                    out = out / n

        # reset the method to start from a clean property
        self._method = None

        return out

    def _get_endpoint_url(self):
        """Return the endpoint_url of an S3 object store, or `None`"""
        endpoint_url = self.storage_options.get('endpoint_url')
        if endpoint_url is not None:
            return endpoint_url

        client_kwargs = self.storage_options.get('client_kwargs')
        if client_kwargs:
            endpoint_url = client_kwargs.get('endpoint_url')
            if endpoint_url is not None:
                return endpoint_url

        return f"http://{urllib.parse.urlparse(self.filename).netloc}"

    def _process_chunk(self, session, ds, chunks, chunk_coords, chunk_selection,
                       out_selection, compressor, filters, drop_axes=None):
        """
        Obtain part or whole of a chunk.

        This is done by taking binary data from storage and filling
        the output array.

        """

        # retrieve coordinates from chunk index
        storeinfo = ds.get_chunk_info_from_chunk_coord(chunk_coords)
        offset, size = storeinfo.byte_offset, storeinfo.size
        self.data_read += size

        # Axes over which to apply a reduction
        axis = self._axis

        if self.storage_type == 's3' and self._version == 1:

            tmp, count = reduce_opens3_chunk(ds._fh, offset, size, compressor, filters,
                            self.missing, ds.dtype,
                            chunks, ds._order,
                            chunk_selection, axis=axis,
                            method=self.method
            )

        elif self.storage_type == "s3" and self._version == 2:
            # S3: pass in pre-configured storage options (credentials)
            # print("S3 rfile is:", self.filename)
            parsed_url = urllib.parse.urlparse(self.filename)

            bucket = parsed_url.netloc
            object = parsed_url.path
        
            # for certain S3 servers rfile needs to contain the bucket eg "bucket/filename"
            # as a result the parser above finds empty string bucket
            if bucket == "":
                bucket = os.path.dirname(object)
                object = os.path.basename(object)
            # print("S3 bucket:", bucket)
            # print("S3 file:", object)
            if self.storage_options is None:
              
                # for the moment we need to force ds.dtype to be a numpy type
                # Reductionist returns "count" as a list even for single elements
                tmp, count = reductionist.reduce_chunk(session,
                                                       S3_ACTIVE_STORAGE_URL,
                                                       S3_URL,
                                                       bucket, object, offset,
                                                       size, compressor, filters,
                                                       self.missing, np.dtype(ds.dtype),
                                                       chunks,
                                                       ds._order,
                                                       chunk_selection,
                                                       axis,
                                                       operation=self._method)
            else:
                # special case for "anon=True" buckets that work only with e.g.
                # fs = s3fs.S3FileSystem(anon=True, client_kwargs={'endpoint_url': S3_URL})
                # where file uri = bucketX/fileY.mc
                # print("S3 Storage options to Reductionist:", self.storage_options)
                if self.storage_options.get("anon", None) == True:
                    bucket = os.path.dirname(parsed_url.path)  # bucketX
                    object = os.path.basename(parsed_url.path)  # fileY
                    print("S3 anon=True Bucket and File:", bucket, object)

                # Reductionist returns "count" as a list even for single elements
                tmp, count = reductionist.reduce_chunk(session,
                                                       self.active_storage_url,
                                                       self._get_endpoint_url(),
                                                       bucket, object, offset,
                                                       size, compressor, filters,
                                                       self.missing, np.dtype(ds.dtype),
                                                       chunks,
                                                       ds._order,
                                                       chunk_selection,
                                                       axis,
                                                       operation=self._method)
        # this is for testing ONLY until Reductionist is able to handle https
        # located files; after that, we can pipe any regular https file through
        # to Reductionist, provided the https server is "closer" to Reductionist
        elif self.storage_type == "https" and self._version == 2:
            # build a simple session
            session = requests.Session()
            session.auth = (None, None)
            session.verify = False
            bucket = "https"  # really doesn't matter

            # note the extra "storage_type" kwarg
            # this currently makes Reductionist throw a wobbly
            # E           activestorage.reductionist.ReductionistError: Reductionist error: HTTP 400: {"error": {"message": "request data is not valid", "caused_by": ["Failed to deserialize the JSON body into the target type", "storage_type: unknown field `storage_type`, expected one of `source`, `bucket`, `object`, `dtype`, `byte_order`, `offset`, `size`, `shape`, `order`, `selection`, `compression`, `filters`, `missing` at line 1 column 550"]}}

            # Reductionist returns "count" as a list even for single elements
            tmp, count = reductionist.reduce_chunk(session,
                                                   "https://reductionist.jasmin.ac.uk/",  # Wacasoft
                                                   self.filename,
                                                   bucket, self.filename, offset,
                                                   size, compressor, filters,
                                                   self.missing, np.dtype(ds.dtype),
                                                   chunks,
                                                   ds._order,
                                                   chunk_selection,
                                                   axis,
                                                   operation=self._method,
                                                   storage_type="https")
        elif self.storage_type=='ActivePosix' and self.version==2:
            # This is where the DDN Fuse and Infinia wrappers go
            raise NotImplementedError
        else:
            # note there is an ongoing discussion about this interface, and what it returns
            # see https://github.com/valeriupredoi/PyActiveStorage/issues/33
            # so neither the returned data or the interface should be considered stable
            # although we will version changes.

            tmp, count = reduce_chunk(self.filename, offset, size, compressor, filters,
                                      self.missing, ds.dtype,
                                      chunks, ds._order,
                                      chunk_selection, axis, method=self.method)

        if self.method is not None:
            # For a reduced axis, replace the index in 'out_selection'
            # with the corresponding position of the chunk in
            # chunks-space.
            #
            # E.g. if 'out_selection' is (slice(0,12), slice(20,60)),
            #      'chunk_coord' is (1, 3), and 'axis' is (1,); then
            #      'out_selection' will become (slice(0,12),
            #      slice(3,4)). If 'axis' were instead (0, 1), then
            #      'out_selection' would become (slice(1,2),
            #      slice(3,4)).
            #
            # This makes sure that 'out_selection' puts 'tmp' in the
            # correct place of the numpy array defined by the method
            # (currently `_from_storage`) that collates the 'tmp'
            # arrays from each chunk.
            out_selection = list(out_selection)
            for i in axis:
                n = chunk_coords[i]
                out_selection[i] = slice(n, n + 1)

            return tmp, count, tuple(out_selection)
        else:
            if drop_axes:
                tmp = np.squeeze(tmp, axis=drop_axes)

            return tmp, None, out_selection

    def _mask_data(self, data):
        """ 
        Missing values obtained at initial getitem, and are used here to 
        mask data, if necessary
        """
        if self.missing is None:
            self.missing = self.__get_missing_attributes()
        _FillValue, missing_value, valid_min, valid_max = self.missing
        
        if _FillValue is not None:
            data = np.ma.masked_equal(data, _FillValue)

        if missing_value is not None:
            data = np.ma.masked_equal(data, missing_value)

        if valid_max is not None:
            data = np.ma.masked_greater(data, valid_max)

        if valid_min is not None:
            data = np.ma.masked_less(data, valid_min)

        return data
