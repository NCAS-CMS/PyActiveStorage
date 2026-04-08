from __future__ import annotations

import numpy as np

from activestorage.core import ChunkRequest, SelectionRequest


class ChunkedLocalStrategy:
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
        out_shape = list(indexer.shape)
        if method is not None:
            nchunks = []
            for dim in indexer.dim_indexers:
                nchunks.append(getattr(dim, "nchunks", 1))
            for i in axis:
                out_shape[i] = nchunks[i]

        out = np.ma.empty(out_shape, dtype=chunk_metadata.dtype, order=chunk_metadata.order)
        out.mask = True
        counts = None
        if need_counts:
            counts = np.ma.empty(out_shape, dtype="int64", order=chunk_metadata.order)
            counts.mask = True

        for chunk_coords, chunk_selection, out_selection in indexer:
            offset, size = backend._active._format.get_chunk_offset_size(chunk_coords)
            request = ChunkRequest(
                uri=chunk_metadata.filename,
                offset=offset,
                size=size,
                dtype=chunk_metadata.dtype,
                chunks=chunk_metadata.chunks,
                order=chunk_metadata.order,
                compressor=chunk_metadata.compressor,
                filters=chunk_metadata.filters,
                chunk_selection=chunk_selection,
                missing=missing,
                method=method,
                axis=axis,
            )
            result = backend.reduce_chunk(request)

            if method is not None:
                out_sel = list(out_selection)
                for i in axis:
                    n = chunk_coords[i]
                    out_sel[i] = slice(n, n + 1)
                out[tuple(out_sel)] = result.data
                if need_counts:
                    counts[tuple(out_sel)] = result.count
            else:
                out[out_selection] = result.data

        if method is None:
            return out

        reducer = backend._active.method
        reduced = reducer(out, axis=axis, keepdims=True)
        if not need_counts:
            return reduced

        n = np.ma.sum(counts, axis=axis, keepdims=True)
        if backend._active.components:
            key = "sum" if method == "mean" else method
            return {key: reduced, "n": n}
        if method == "mean":
            return reduced / n
        return reduced


class ChunkedRemoteStrategy(ChunkedLocalStrategy):
    pass


class WholeArrayStrategy:
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
        request = SelectionRequest(
            uri=chunk_metadata.filename,
            variable=backend._active.ncvar,
            selection=getattr(indexer, "selection", ()),
            method=method,
            axis=axis,
            missing=missing,
            compressor=chunk_metadata.compressor,
            filters=chunk_metadata.filters,
        )
        result = backend.reduce_selection(request)
        if backend._active.components and result.n is not None:
            key = "sum" if method == "mean" else method
            return {key: result.data, "n": result.n}
        if method == "mean" and result.n:
            return result.data / result.n
        return result.data
