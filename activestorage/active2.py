import concurrent.futures
import os
import urllib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import pyfive

from activestorage import reductionist
from activestorage.config import *  # noqa
from activestorage.hdf2numcodec import decode_filters
from activestorage.storage import reduce_chunk, reduce_opens3_chunk

# —————————————————————————

# Abstract base

# —————————————————————————

class StorageBackend(ABC):
    """
    Abstract base for all storage backends.

    Each backend encapsulates:
    - session lifecycle (get_session / close_session)
    - chunk reduction (reduce_chunk)
    - which methods it supports

    Subclasses must implement reduce_chunk(). get_session() and
    close_session() are optional (default to no-op / None).
    """

    # Backends that don't support every operation (e.g. a remote service)
    # can narrow this set. Active will raise early if the requested method
    # isn't in here, however, the minimum set (min, max, sum, mean) is 
    # expected to be supported by all backends.

    supported_methods: set = {"min", "max", "sum", "mean"}

    def __init__(self, active: "Active"):
        # Keep a reference to the owning Active instance so backends can
        # reach storage_options, filename, missing, etc. without duplicating state.
        self._active = active

    def get_session(self):
        """Return a session object (e.g. requests.Session) or None."""
        return None

    def close_session(self, session):
        """Clean up a session if necessary."""
        pass

    @abstractmethod
    def reduce_chunk(
        self,
        session,
        ds,
        chunks,
        chunk_coords,
        chunk_selection,
        compressor,
        filters,
        axis,
    ):
        """
        Reduce a single chunk.

        Must return (tmp, count) where:
        tmp   - reduced (or raw) numpy masked array for this chunk
        count - integer count of non-missing elements, or None
        """
        return NotImplementedError("Must implement reduce_chunk in subclass")


# —————————————————————————
# Concrete backends
# —————————————————————————

class LocalBackend(StorageBackend):
    """
    Reads directly from a local (POSIX) file.
    Used when interface_type is None / falsy and version is 1.
    """
    def reduce_chunk(self, session, ds, chunks, chunk_coords,
                    chunk_selection, compressor, filters, axis):
        active = self._active
        storeinfo = ds.get_chunk_info_from_chunk_coord(chunk_coords)
        offset, size = storeinfo.byte_offset, storeinfo.size
        active.data_read += size

        return reduce_chunk(
            active.filename,
            offset, size,
            compressor, filters,
            active.missing,
            ds.dtype, chunks, ds._order,
            chunk_selection, axis,
            method=active.method,
        )
    

class S3V1Backend(StorageBackend):
    """
    Reads from S3 directly (version 1 — no Reductionist).
    Chunk data is fetched and reduced locally via reduce_opens3_chunk.
    """
    def reduce_chunk(self, session, ds, chunks, chunk_coords,
                    chunk_selection, compressor, filters, axis):
        
        active = self._active
        storeinfo = ds.get_chunk_info_from_chunk_coord(chunk_coords)
        offset, size = storeinfo.byte_offset, storeinfo.size
        active.data_read += size

        return reduce_opens3_chunk(
            ds._fh,
            offset, size,
            compressor, filters,
            active.missing,
            ds.dtype, chunks, ds._order,
            chunk_selection,
            axis=axis,
            method=active.method,
        )


class S3V2Backend(StorageBackend):
    """
    Reductionist path for S3 (version 2).
    Delegates chunk reduction to the remote Reductionist service.
    """

    def get_session(self):
        active = self._active
        opts = active.storage_options

        if opts is not None:
            if opts.get("anon", None) is True:
                return reductionist.get_session(None, None, S3_ACTIVE_STORAGE_CACERT)
            key = opts.get("key")
            secret = opts.get("secret")
            if key and secret:
                return reductionist.get_session(key, secret, S3_ACTIVE_STORAGE_CACERT)

        return reductionist.get_session(
            S3_ACCESS_KEY, S3_SECRET_KEY, S3_ACTIVE_STORAGE_CACERT
        )

    def _resolve_bucket_object(self, filename, storage_options):
        """
        Parse bucket and object path from the S3 URI.
        """
        parsed = urllib.parse.urlparse(filename)
        bucket = parsed.netloc
        obj = parsed.path

        if bucket == "":
            bucket = os.path.dirname(obj)
            obj = os.path.basename(obj)

        if storage_options is not None and storage_options.get("anon", None) is True:
            bucket = os.path.dirname(parsed.path)
            obj = os.path.basename(parsed.path)

        return bucket, obj

    def reduce_chunk(self, session, ds, chunks, chunk_coords,
                    chunk_selection, compressor, filters, axis):
        active = self._active
        storeinfo = ds.get_chunk_info_from_chunk_coord(chunk_coords)
        offset, size = storeinfo.byte_offset, storeinfo.size
        active.data_read += size

        bucket, obj = self._resolve_bucket_object(
            active.filename, active.storage_options
        )

        if active.storage_options is None:
            url = f"{S3_URL}/{bucket}/{obj}"
            active_storage_url = S3_ACTIVE_STORAGE_URL
        else:
            url = f"{active._get_endpoint_url()}/{bucket}/{obj}"
            active_storage_url = active.active_storage_url

        return reductionist.reduce_chunk(
            session,
            active_storage_url,
            url,
            offset, size,
            compressor, filters,
            active.missing,
            np.dtype(ds.dtype),
            chunks, ds._order,
            chunk_selection, axis,
            operation=active._method,
        )

class HttpsV2Backend(StorageBackend):
    """
    Reductionist path for HTTPS (version 2).
    """

    def get_session(self):
        active = self._active
        opts = active.storage_options
        username = opts.get("username") if opts else None
        password = opts.get("password") if opts else None

        if username and password:
            return reductionist.get_session(username, password, None)
        return reductionist.get_session(None, None, None)

    def reduce_chunk(self, session, ds, chunks, chunk_coords,
                    chunk_selection, compressor, filters, axis):
        
        active = self._active
        storeinfo = ds.get_chunk_info_from_chunk_coord(chunk_coords)
        offset, size = storeinfo.byte_offset, storeinfo.size
        active.data_read += size

        return reductionist.reduce_chunk(
            session,
            active.active_storage_url,
            active.filename,
            offset, size,
            compressor, filters,
            active.missing,
            np.dtype(ds.dtype),
            chunks, ds._order,
            chunk_selection, axis,
            operation=active._method,
            interface_type="https",
        )

class P5RemBackend(StorageBackend):
    """
    Backend for p5rem.File instances.

    p5rem provides a remote HDF5-like interface. The reduction strategy
    here is intentionally left as a sketch — fill in reduce_chunk with
    the p5rem-specific API calls.

    If p5rem exposes its own server-side reduction, get_session() should
    return an authenticated client. If it doesn't, fall back to fetching
    raw bytes and reducing locally (same pattern as S3V1Backend).
    """

    # If p5rem's remote service doesn't support all operations yet,
    # narrow this. Active will raise ValueError before even attempting.
    supported_methods: set = {"min", "max", "sum", "mean"}

    def get_session(self):
        active = self._active
        opts = active.storage_options or {}
        # TODO: return an authenticated p5rem client/session
        # e.g. return p5rem.Client(**opts)
        return None

    def reduce_chunk(self, session, ds, chunks, chunk_coords,
                    chunk_selection, compressor, filters, axis):
        active = self._active
        storeinfo = ds.get_chunk_info_from_chunk_coord(chunk_coords)
        offset, size = storeinfo.byte_offset, storeinfo.size
        active.data_read += size

        # TODO: use session + p5rem API to fetch and reduce chunk
        # For now, delegate to local reduction as a safe fallback
        raise NotImplementedError(
            "P5RemBackend.reduce_chunk not yet implemented. "
            "Plug in p5rem chunk fetch + reduction here."
        )

# —————————————————————————
# Backend registry
# —————————————————————————

_BACKENDS: dict[tuple, type[StorageBackend]] = {
        (None, 1):      LocalBackend,
        (None, 2):      LocalBackend,
        ("s3", 1):      S3V1Backend,
        ("s3", 2):      S3V2Backend,
        ("https", 2):   HttpsV2Backend,
        ("p5rem", 1):   P5RemBackend,
        ("p5rem", 2):   P5RemBackend,
        }

def _select_backend(interface_type, version) -> type[StorageBackend]:
    key = (interface_type, version)
    backend_cls = _BACKENDS.get(key)
    if backend_cls is None:
        raise ValueError(
                f"No backend registered for interface_type={interface_type!r}, "
                f"version={version}. "
                f"Available: {list(_BACKENDS.keys())}"
            )
    return backend_cls

# —————————————————————————
# Active (core — orchestration only)
# —————————————————————————

class Active:
    """
    Instantiates an interface to active storage (zarr / HDF5 / NetCDF4).

    Storage-specific behaviour is delegated to StorageBackend subclasses.
    New backends can be added without modifying this class.
    """

    # ------------------------------------------------------------------
    # Immutable core reduction methods (always present on every instance)
    # ------------------------------------------------------------------

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance._methods = {
            "min":  np.ma.min,
            "max":  np.ma.max,
            "sum":  np.ma.sum,
            "mean": np.ma.sum,   # sum here; divided by n later
        }
        return instance

    def register_method(self, name: str, func):
        """
        Register a plugin reduction method.

        :param name: Method name (must not shadow a built-in).
        :param func: Callable with the same signature as np.ma.min etc.

        Methods must be available at the chosen backend, otherwise 
        things will go badly!

        """
        if name in self._methods:
            raise ValueError(
                f"Cannot override built-in method {name!r}. "
                f"Built-ins are: {list(self._methods)}"
            )
        if not callable(func):
            raise TypeError(f"func must be callable, got {type(func)}")
        self._methods[name] = func

    def __init__(
        self,
        dataset: Optional[str | Path | object],
        ncvar: str = None,
        axis: tuple = None,
        interface_type: str = None,
        max_threads: int = 100,
        storage_options: dict = None,
        active_storage_url: str = None,
        option_disable_chunk_cache: bool = False,
    ) -> None:
        self.ds = None
        self._version = 1
        self._components = False
        self._method = None
        self._max_threads = max_threads
        self.missing = None
        self.data_read = 0
        self.storage_options = storage_options
        self.active_storage_url = active_storage_url

        if dataset is None:
            raise ValueError(
                f"Must use a valid file name or variable object. Got {dataset!r}"
            )

        if isinstance(dataset, Path) and not dataset.exists():
            raise ValueError(f"Path {dataset!r} does not exist.")

        # --- Detect if dataset is a p5rem.File instance -----------------
        # Import lazily so p5rem remains an optional dependency
        try:
            import p5rem
            _p5rem_file_type = p5rem.File
        except ImportError:
            _p5rem_file_type = type(None)  # never matches

        if isinstance(dataset, _p5rem_file_type):
            # p5rem path: treat like a variable object but with its own backend
            self.ds = dataset
            self.uri = dataset
            interface_type = "p5rem"
        elif not isinstance(dataset, (Path, str)):
            # Existing pyfive.high_level.Dataset path
            if not isinstance(dataset, pyfive.high_level.Dataset):
                raise TypeError(
                    f"dataset must be a path, string, or pyfive Dataset. Got {dataset!r}"
                )
            self.ds = dataset
            self.uri = dataset
        else:
            self.uri = dataset

        self.interface_type = interface_type or self._detect_interface_type()

        if axis is not None:
            axis = (axis,) if isinstance(axis, int) else tuple(axis)
        self._axis = axis

        self.ncvar = ncvar
        if self.ncvar is None and self.ds is None:
            raise ValueError("Must set a netCDF variable name to slice")

        # --- Select and instantiate backend ----------------------------
        backend_cls = _select_backend(self.interface_type, self._version)
        self._backend = backend_cls(self)

        # Guard: check requested method is supported by this backend
        # (deferred to method.setter since method isn't set yet)

    def _detect_interface_type(self):
        """
        Infer interface_type from URI / storage_options.
        """
        # (Unchanged logic from original — abbreviated here for clarity)
        from activestorage.active import return_interface_type
        return return_interface_type(str(self.uri))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def components(self):
        return self._components

    @components.setter
    def components(self, value):
        self._components = bool(value)

    @property
    def method(self):
        return self._methods.get(self._method)

    @method.setter
    def method(self, value):
        if value is not None:
            if value not in self._methods:
                raise ValueError(
                    f"Unknown method {value!r}. "
                    f"Available: {list(self._methods)}"
                )
            if value not in self._backend.supported_methods:
                raise ValueError(
                    f"Method {value!r} is not supported by {type(self._backend).__name__}. "
                    f"Supported: {self._backend.supported_methods}"
                )
        self._method = value

    # ------------------------------------------------------------------
    # Public reduction shortcuts
    # ------------------------------------------------------------------

    def mean(self, axis=None):
        self._method = "mean"
        if axis is not None:
            self._axis = axis
        return self

    def min(self, axis=None):
        self._method = "min"
        if axis is not None:
            self._axis = axis
        return self

    def max(self, axis=None):
        self._method = "max"
        if axis is not None:
            self._axis = axis
        return self

    # ------------------------------------------------------------------
    # Core data retrieval
    # ------------------------------------------------------------------

    def __getitem__(self, index):
        if self.ds is None:
            self._load_nc_file()
        self.missing = self._get_missing_attributes()
        self.data_read = 0

        if self.method is None and self._version == 0:
            return self._get_vanilla(index)
        return self._get_selection(index)

    def _get_vanilla(self, index):
        data = self.ds[index]
        return self._mask_data(data)

    def _get_selection(self, *args):
        """Gather metadata then dispatch to _from_storage."""
        keepdims = True
        dtype = np.dtype(self.ds.dtype)
        array = pyfive.indexing.ZarrArrayStub(self.ds.shape, self.ds.chunks)
        ds = self.ds.id

        if self._axis is None:
            self._axis = tuple(range(len(ds.shape)))

        compressor, filters = (
            (None, None) if ds.filter_pipeline is None
            else decode_filters(ds.filter_pipeline, dtype.itemsize, self.ds.name)
        )

        indexer = pyfive.indexing.OrthogonalIndexer(*args, array)
        out_shape = indexer.shape
        drop_axes = indexer.drop_axes and keepdims

        return self._from_storage(
            ds, indexer, array._chunks, out_shape, dtype,
            compressor, filters, drop_axes, self._axis,
        )

    def _from_storage(self, ds, indexer, chunks, out_shape, out_dtype,
                    compressor, filters, drop_axes, axis):
        """
        Orchestrate threaded chunk processing.

        Session lifecycle and chunk reduction are fully delegated to
        self._backend — this method contains no backend-specific logic.
        """
        method = self.method
        need_counts = self.components or self._method == "mean"

        if self.components and self._method is None:
            raise ValueError("components=True requires a statistical method.")

        if method is not None:
            # Build output shape: one slot per chunk along reduced axes
            nchunks = []
            for i, d in enumerate(indexer.dim_indexers):
                try:
                    nchunks.append(d.nchunks)
                except AttributeError:
                    raise IndexError(
                        f"Can't do an active reduction when axis {i!r} drops the axis."
                    )

            out_shape = list(out_shape)
            for i in axis:
                try:
                    out_shape[i] = nchunks[i]
                except IndexError:
                    raise ValueError(f"Out-of-range axis for reduction: {i!r}")

        out = np.ma.empty(out_shape, dtype=out_dtype, order=ds._order)
        out.mask = True

        if need_counts:
            counts = np.ma.empty(out_shape, dtype="int64", order=ds._order)
            counts.mask = True

        # --- Session (backend handles auth / credentials) --------------
        session = self._backend.get_session()

        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self._max_threads
            ) as executor:
                futures = [
                    executor.submit(
                        self._process_chunk,
                        session, ds, chunks,
                        chunk_coords, chunk_selection, out_selection,
                        compressor, filters,
                        drop_axes=drop_axes,
                    )
                    for chunk_coords, chunk_selection, out_selection in indexer
                ]

                for future in concurrent.futures.as_completed(futures):
                    result, count, out_selection = future.result()
                    out[out_selection] = result
                    if need_counts:
                        counts[out_selection] = count
        finally:
            self._backend.close_session(session)

        # --- Aggregate across chunks -----------------------------------
        if method is not None:
            out = method(out, axis=axis, keepdims=True)
            if need_counts:
                n = np.ma.sum(counts, axis=axis, keepdims=True)
                if self._components:
                    key = "sum" if self._method == "mean" else self._method
                    out = {key: out, "n": n}
                elif self._method == "mean":
                    out = out / n

        self._method = None
        return out

    def _process_chunk(self, session, ds, chunks, chunk_coords,
                    chunk_selection, out_selection,
                    compressor, filters, drop_axes=None):
        """
        Reduce a single chunk via the backend, then place it in out_selection.
        No backend-specific logic lives here.
        """
        axis = self._axis

        tmp, count = self._backend.reduce_chunk(
            session, ds, chunks, chunk_coords,
            chunk_selection, compressor, filters, axis,
        )

        if self.method is not None:
            out_selection = list(out_selection)
            for i in axis:
                n = chunk_coords[i]
                out_selection[i] = slice(n, n + 1)
            return tmp, count, tuple(out_selection)
        else:
            if drop_axes:
                tmp = np.squeeze(tmp, axis=drop_axes)
            return tmp, None, out_selection

    # ------------------------------------------------------------------
    # Helpers (unchanged from original)
    # ------------------------------------------------------------------

    def _get_endpoint_url(self):
        from activestorage.active import get_endpoint_url
        endpoint_url = get_endpoint_url(self.storage_options)
        if endpoint_url is not None:
            return endpoint_url
        return f"http://{urllib.parse.urlparse(self.filename).netloc}"

    def _load_nc_file(self):
        from activestorage.active import load_from_s3, load_from_https
        ncvar = self.ncvar
        if self.interface_type is None:
            nc = pyfive.File(self.uri)
        elif self.interface_type == "s3":
            nc = load_from_s3(self.uri, self.storage_options)
        elif self.interface_type == "https":
            nc = load_from_https(self.uri, self.storage_options)
        self.filename = self.uri
        self.ds = nc[ncvar]

    def _get_missing_attributes(self):
        if self.ds is None:
            self._load_nc_file()
        from activestorage.active import get_missing_attributes
        return get_missing_attributes(self.ds)

    def _mask_data(self, data):
        if self.missing is None:
            self.missing = self._get_missing_attributes()
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

