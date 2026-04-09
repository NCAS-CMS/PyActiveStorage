import threading
import time
from types import SimpleNamespace

import numpy as np

from activestorage.core import ChunkMetadata, ChunkResult, MissingAttributes
from activestorage.strategies import ChunkedLocalStrategy, ChunkedRemoteStrategy


class FakeFormat:
    def get_chunk_offset_size(self, chunk_coords):
        return chunk_coords[0], 1


class FakeDimIndexer:
    def __init__(self, nchunks):
        self.nchunks = nchunks


class FakeIndexer:
    def __init__(self, nchunks):
        self.shape = (nchunks,)
        self.dim_indexers = [FakeDimIndexer(nchunks)]
        self._nchunks = nchunks

    def __iter__(self):
        for i in range(self._nchunks):
            yield (i,), (slice(0, 1, 1),), (slice(i, i + 1, 1),)


class FakeBackend:
    def __init__(self, max_threads):
        self._active = SimpleNamespace(
            _format=FakeFormat(),
            _max_threads=max_threads,
            method=np.ma.max,
            components=False,
        )
        self.thread_ids = set()

    def reduce_chunk(self, request):
        self.thread_ids.add(threading.get_ident())
        time.sleep(0.01)
        return ChunkResult(
            data=np.array([request.offset], dtype=np.float64),
            count=1,
            out_selection=(),
        )


def make_chunk_metadata():
    return ChunkMetadata(
        dtype=np.dtype("float64"),
        shape=(4,),
        chunks=(1,),
        compressor=None,
        filters=None,
        order="C",
        filename="fake.nc",
    )


def test_chunked_local_strategy_parallel_threads():
    strategy = ChunkedLocalStrategy()
    backend = FakeBackend(max_threads=4)

    result = strategy.execute(
        backend,
        session=None,
        chunk_metadata=make_chunk_metadata(),
        indexer=FakeIndexer(4),
        missing=MissingAttributes(),
        method="max",
        need_counts=False,
        axis=(0,),
    )

    assert result.shape == (1,)
    assert result[0] == 3.0
    assert len(backend.thread_ids) > 1


def test_chunked_remote_strategy_parallel_threads():
    strategy = ChunkedRemoteStrategy()
    backend = FakeBackend(max_threads=4)

    result = strategy.execute(
        backend,
        session=None,
        chunk_metadata=make_chunk_metadata(),
        indexer=FakeIndexer(4),
        missing=MissingAttributes(),
        method="max",
        need_counts=False,
        axis=(0,),
    )

    assert result.shape == (1,)
    assert result[0] == 3.0
    assert len(backend.thread_ids) > 1


def test_chunked_strategy_serial_when_single_thread():
    strategy = ChunkedLocalStrategy()
    backend = FakeBackend(max_threads=1)

    result = strategy.execute(
        backend,
        session=None,
        chunk_metadata=make_chunk_metadata(),
        indexer=FakeIndexer(4),
        missing=MissingAttributes(),
        method="max",
        need_counts=False,
        axis=(0,),
    )

    assert result.shape == (1,)
    assert result[0] == 3.0
    assert len(backend.thread_ids) == 1