from __future__ import annotations

import os
import urllib

import numpy as np

from activestorage import reductionist
from activestorage.config import (
    S3_ACCESS_KEY,
    S3_ACTIVE_STORAGE_CACERT,
    S3_ACTIVE_STORAGE_URL,
    S3_SECRET_KEY,
    S3_URL,
)
from activestorage.core import ChunkRequest, ChunkResult, SelectionRequest, SelectionResult, StorageBackend
from activestorage.helpers import get_endpoint_url
from activestorage.storage import reduce_chunk, reduce_opens3_chunk
from activestorage.strategies import ChunkedLocalStrategy, ChunkedRemoteStrategy, WholeArrayStrategy


class CacheAwareBackend(StorageBackend):
    def is_range_cached(self, fh, offset, size) -> bool:
        return False


class ReductionistBackend(CacheAwareBackend):
    def __init__(self, active):
        super().__init__(active)
        self.execution_strategy = ChunkedRemoteStrategy()

    def build_url(self, filename, storage_options):
        raise NotImplementedError

    def reduce_selection(self, request: SelectionRequest) -> SelectionResult:
        raise NotImplementedError("Whole-array reduction not implemented for ReductionistBackend")


class LocalBackend(StorageBackend):
    def __init__(self, active):
        super().__init__(active)
        self.execution_strategy = ChunkedLocalStrategy()

    def reduce_chunk(self, request: ChunkRequest) -> ChunkResult:
        method = self._active._methods.get(request.method) if request.method else None
        data, count = reduce_chunk(
            request.uri,
            request.offset,
            request.size,
            request.compressor,
            request.filters,
            (
                request.missing.fill_value,
                request.missing.missing_value,
                request.missing.valid_min,
                request.missing.valid_max,
            ),
            request.dtype,
            request.chunks,
            request.order,
            request.chunk_selection,
            method=method,
        )
        return ChunkResult(data=data, count=count, out_selection=())


class S3Backend(ReductionistBackend):
    def get_session(self):
        opts = self._active.storage_options or {}
        key = opts.get("key", S3_ACCESS_KEY)
        secret = opts.get("secret", S3_SECRET_KEY)
        return reductionist.get_session(key, secret, S3_ACTIVE_STORAGE_CACERT)

    def _resolve_bucket_object(self, filename, storage_options):
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

    def reduce_chunk(self, request: ChunkRequest) -> ChunkResult:
        if self._active._version == 1:
            fh = self._active._format.file_handle
            method = self._active._methods.get(request.method) if request.method else None
            data, count = reduce_opens3_chunk(
                fh,
                request.offset,
                request.size,
                request.compressor,
                request.filters,
                (
                    request.missing.fill_value,
                    request.missing.missing_value,
                    request.missing.valid_min,
                    request.missing.valid_max,
                ),
                request.dtype,
                request.chunks,
                request.order,
                request.chunk_selection,
                method=method,
            )
            return ChunkResult(data=data, count=count, out_selection=())

        bucket, obj = self._resolve_bucket_object(request.uri, self._active.storage_options)
        if self._active.storage_options is None:
            source = S3_URL
            server = S3_ACTIVE_STORAGE_URL
        else:
            source = get_endpoint_url(self._active.storage_options, request.uri)
            server = self._active.active_storage_url or S3_ACTIVE_STORAGE_URL

        session = self.get_session()
        data, count = reductionist.reduce_chunk(
            session,
            server,
            source,
            request.offset,
            request.size,
            request.compressor,
            request.filters,
            (
                request.missing.fill_value,
                request.missing.missing_value,
                request.missing.valid_min,
                request.missing.valid_max,
            ),
            np.dtype(request.dtype),
            request.chunks,
            request.order,
            request.chunk_selection,
            axis=None,
            operation=request.method,
            interface_type='s3',
        )
        self.close_session(session)
        return ChunkResult(data=data, count=count, out_selection=())


class HttpsBackend(ReductionistBackend):
    def get_session(self):
        opts = self._active.storage_options or {}
        username = opts.get("username")
        password = opts.get("password")
        return reductionist.get_session(username, password, None)

    def build_url(self, filename, storage_options):
        return filename

    def reduce_chunk(self, request: ChunkRequest) -> ChunkResult:
        session = self.get_session()
        data, count = reductionist.reduce_chunk(
            session,
            self._active.active_storage_url,
            request.uri,
            request.offset,
            request.size,
            request.compressor,
            request.filters,
            (
                request.missing.fill_value,
                request.missing.missing_value,
                request.missing.valid_min,
                request.missing.valid_max,
            ),
            np.dtype(request.dtype),
            request.chunks,
            request.order,
            request.chunk_selection,
            axis=None,
            operation=request.method,
            interface_type='https',
        )
        self.close_session(session)
        return ChunkResult(data=data, count=count, out_selection=())


class P5RemBackend(StorageBackend):
    def __init__(self, active):
        super().__init__(active)
        self.execution_strategy = WholeArrayStrategy()

    def get_session(self):
        return None

    def reduce_chunk(self, request: ChunkRequest) -> ChunkResult:
        raise NotImplementedError("Chunked p5rem path is not implemented")

    def reduce_selection(self, request: SelectionRequest) -> SelectionResult:
        data = self._active.ds[request.selection]
        method = self._active._methods.get(request.method) if request.method else None
        if method is None:
            return SelectionResult(data=data, n=None)
        reduced = method(data, axis=request.axis, keepdims=True)
        n = np.size(data)
        if request.method == "mean":
            return SelectionResult(data=reduced, n=n)
        return SelectionResult(data=reduced, n=n)
