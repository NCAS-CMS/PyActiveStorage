from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from activestorage.helpers import get_endpoint_url, get_missing_attributes, return_interface_type


@dataclass
class MissingAttributes:
    fill_value: Any = None
    missing_value: Any = None
    valid_min: Any = None
    valid_max: Any = None


@dataclass
class ChunkMetadata:
    dtype: Any
    shape: tuple
    chunks: tuple
    compressor: Any
    filters: list
    order: str
    filename: str


@dataclass
class ChunkRequest:
    uri: str
    offset: int
    size: int
    dtype: Any
    chunks: tuple
    order: str
    compressor: Any
    filters: list
    chunk_selection: tuple
    missing: MissingAttributes
    method: Optional[str]
    axis: Optional[tuple]


@dataclass
class SelectionRequest:
    uri: str
    variable: str
    selection: tuple
    method: Optional[str]
    axis: Optional[tuple]
    missing: MissingAttributes
    compressor: Any
    filters: list


@dataclass
class ChunkResult:
    data: Any
    count: Optional[int]
    out_selection: tuple


@dataclass
class SelectionResult:
    data: Any
    n: Optional[int]


class StorageFormat(ABC):
    @abstractmethod
    def open(self, uri, storage_options):
        raise NotImplementedError

    @abstractmethod
    def get_variable(self, ncvar):
        raise NotImplementedError

    @abstractmethod
    def get_missing_attributes(self) -> MissingAttributes:
        raise NotImplementedError

    @abstractmethod
    def get_indexer(self, selection):
        raise NotImplementedError

    @abstractmethod
    def get_chunk_metadata(self) -> ChunkMetadata:
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError


class StorageBackend(ABC):
    supported_methods = {"min", "max", "sum", "mean"}

    def __init__(self, active):
        self._active = active
        self.execution_strategy = None

    def get_session(self):
        return None

    def close_session(self, session):
        return None

    @abstractmethod
    def reduce_chunk(self, request: ChunkRequest) -> ChunkResult:
        raise NotImplementedError

    def reduce_selection(self, request: SelectionRequest) -> SelectionResult:
        raise NotImplementedError("Whole-array reduction not implemented for this backend")


class ExecutionStrategy(ABC):
    @abstractmethod
    def execute(
        self,
        backend,
        session,
        chunk_metadata,
        indexer,
        missing,
        method,
        need_counts,
        axis,
    ):
        raise NotImplementedError


def _select_backend(interface_type, version):
    from activestorage.backends import HttpsBackend, LocalBackend, P5RemBackend, S3Backend

    backends = {
        (None, 0): LocalBackend,
        (None, 1): LocalBackend,
        (None, 2): LocalBackend,
        ("s3", 0): S3Backend,
        ("s3", 1): S3Backend,
        ("s3", 2): S3Backend,
        ("https", 0): HttpsBackend,
        ("https", 1): LocalBackend,
        ("https", 2): HttpsBackend,
        ("p5rem", 0): P5RemBackend,
        ("p5rem", 1): P5RemBackend,
        ("p5rem", 2): P5RemBackend,
    }
    backend = backends.get((interface_type, version))
    if backend is None:
        raise ValueError(
            f"No backend registered for interface_type={interface_type!r}, "
            f"version={version}. Available: {list(backends)}"
        )
    return backend


def _select_format(dataset):
    from activestorage.formats import KerchunkFormat, P5RemFormat, PyfiveFormat, ZarrFormat

    lower = str(dataset).lower()
    if lower.endswith(".kerchunk"):
        return KerchunkFormat
    if lower.endswith(".zarr"):
        return ZarrFormat
    return PyfiveFormat


class Active:
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance._methods = {
            "min": np.ma.min,
            "max": np.ma.max,
            "sum": np.ma.sum,
            "mean": np.ma.sum,
        }
        return instance

    def __init__(
        self,
        uri,
        ncvar=None,
        storage_type=None,
        interface_type=None,
        max_threads=100,
        storage_options=None,
        active_storage_url=None,
        axis=None,
        option_disable_chunk_cache=False,
    ):
        self.uri = uri
        if self.uri is None:
            raise ValueError(f"Must use a valid file for uri. Got {uri}")

        # Keep source URI when a dataset/variable object is provided.
        is_pathlike = isinstance(uri, (str, bytes, os.PathLike))
        source_uri = uri
        if not is_pathlike:
            file_obj = getattr(uri, "file", None)
            fh = getattr(file_obj, "_fh", None)
            source_uri = (
                getattr(fh, "path", None)
                or getattr(fh, "url", None)
                or str(uri)
            )

        # interface_type is an alias for storage_type
        if interface_type is not None:
            storage_type = interface_type

        self.storage_type = storage_type or return_interface_type(source_uri)
        self.storage_options = storage_options or {}
        if self.storage_type is None:
            # Backward-compatible inference for bare S3 object paths like
            # "bucket/key.nc" when only storage_options indicate S3 access.
            # Only infer when the URI is not an existing local file.
            s3_hints = {"key", "secret", "anon", "client_kwargs", "endpoint_url"}
            if any(k in self.storage_options for k in s3_hints):
                if not (is_pathlike and os.path.isfile(str(uri))):
                    self.storage_type = "s3"
        self._option_disable_chunk_cache = bool(option_disable_chunk_cache)
        self.active_storage_url = active_storage_url

        # Allow passing dataset/variable objects directly (ncvar optional).
        is_file_object = not is_pathlike
        if is_pathlike and not os.path.isfile(self.uri) and not self.storage_type:
            raise ValueError(f"Must use existing file for uri. {self.uri} not found")

        # When uri is a dataset object, ncvar can be None (user will select variable via indexing)
        if ncvar is None and not is_file_object:
            raise ValueError("Must set a netCDF variable name to slice")

        self._ncvar = ncvar
        self._version = 1
        self._components = False
        self._method = None
        self._axis = (axis,) if isinstance(axis, int) else axis
        self._max_threads = max_threads
        self.metric_data = {}
        self.data_read = 0

        self._format = _select_format(source_uri)()
        self._format._storage_type = self.storage_type
        if is_file_object:
            # uri is already a pyfive.Group or similar
            self._format._dataset = uri
            self._format._uri = str(source_uri)
            self.ds = uri
        else:
            self._format.open(uri, self.storage_options)
            if ncvar is not None:
                self.ds = self._format.get_variable(ncvar)
            else:
                self.ds = None
        self.missing = None

        self._refresh_backend()

    @property
    def ncvar(self):
        return self._ncvar

    @ncvar.setter
    def ncvar(self, value):
        self._ncvar = value

    @property
    def interface_type(self):
        return self.storage_type

    def _refresh_backend(self):
        backend_cls = _select_backend(self.storage_type, self._version)
        self._backend = backend_cls(self)

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
        if value is not None and value not in self._methods:
            raise ValueError(f"Bad 'method': {value}. Choose from min/max/mean/sum.")
        self._method = value

    def register_method(self, name, func):
        self._methods[name] = func

    def min(self, axis=None):
        self._method = "min"
        self._axis = (axis,) if isinstance(axis, int) else axis
        return self

    def max(self, axis=None):
        self._method = "max"
        self._axis = (axis,) if isinstance(axis, int) else axis
        return self

    def mean(self, axis=None):
        self._method = "mean"
        self._axis = (axis,) if isinstance(axis, int) else axis
        return self

    def sum(self, axis=None):
        self._method = "sum"
        self._axis = (axis,) if isinstance(axis, int) else axis
        return self

    def __getitem__(self, index):
        self.metric_data = {}
        if self._version not in (0, 1, 2):
            raise ValueError(f"Version {self._version} not supported")

        self._refresh_backend()
        self.missing = self._format.get_missing_attributes()
        self.data_read = 0

        if self.method is None and self._version == 0:
            return self._get_vanilla(index)
        return self._get_selection(index)

    def _get_vanilla(self, index):
        data = self.ds[index]
        return self._mask_data(data)

    def _get_active(self, method, *args):
        raise NotImplementedError

    def _get_selection(self, selection):
        chunk_metadata = self._format.get_chunk_metadata()
        indexer = self._format.get_indexer(selection)
        ndim = len(chunk_metadata.shape)
        axis = self._axis
        if axis is None:
            axis = tuple(range(ndim))
        else:
            # Validate axis values; normalise negative indices for internal use.
            normalised = []
            for i in axis:
                if not (-ndim <= i < ndim):
                    raise ValueError(
                        f"axis {i} is out of bounds for array of dimension {ndim}"
                    )
                normalised.append(i % ndim)
            axis = tuple(normalised)

        session = self._backend.get_session()
        try:
            need_counts = self._components or self._method == "mean"
            return self._from_storage(
                session,
                chunk_metadata,
                indexer,
                self.missing,
                self._method,
                need_counts,
                axis,
            )
        finally:
            self._backend.close_session(session)

    def _from_storage(self, session, chunk_metadata, indexer, missing, method, need_counts, axis):
        return self._backend.execution_strategy.execute(
            self._backend,
            session,
            chunk_metadata,
            indexer,
            missing,
            method,
            need_counts,
            axis,
        )

    def _process_chunk(self, request: ChunkRequest) -> ChunkResult:
        return self._backend.reduce_chunk(request)

    def _get_endpoint_url(self):
        return get_endpoint_url(self.storage_options, self.uri)

    def _mask_data(self, data):
        if self.missing is None:
            self.missing = get_missing_attributes(self.ds)

        if isinstance(self.missing, MissingAttributes):
            fill_value = self.missing.fill_value
            missing_value = self.missing.missing_value
            valid_min = self.missing.valid_min
            valid_max = self.missing.valid_max
        else:
            fill_value, missing_value, valid_min, valid_max = self.missing

        def _as_scalar(value):
            if value is None:
                return None
            if not np.isscalar(value):
                try:
                    if len(value) == 1:
                        return value[0]
                except TypeError:
                    pass
            return value

        fill_value = _as_scalar(fill_value)
        missing_value = _as_scalar(missing_value)
        valid_min = _as_scalar(valid_min)
        valid_max = _as_scalar(valid_max)

        if fill_value is not None:
            data = np.ma.masked_equal(data, fill_value)
        if missing_value is not None:
            data = np.ma.masked_equal(data, missing_value)
        if valid_max is not None:
            data = np.ma.masked_greater(data, valid_max)
        if valid_min is not None:
            data = np.ma.masked_less(data, valid_min)
        return data
