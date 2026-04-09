from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from activestorage.core import ChunkRequest, SelectionRequest


class ChunkedLocalStrategy:
    def _iter_chunk_requests(self, backend, chunk_metadata, indexer, missing, method, axis):
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
            yield chunk_coords, out_selection, request

    def _store_chunk_result(self, backend, out, counts, axis, need_counts, method, chunk_coords, out_selection, result):
        if method is not None:
            out_sel = list(out_selection)
            for i in axis:
                n = chunk_coords[i]
                out_sel[i] = slice(n, n + 1)
            out[tuple(out_sel)] = result.data
            if need_counts:
                counts[tuple(out_sel)] = result.count
            return

        out[out_selection] = result.data

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

        max_threads = max(1, getattr(backend._active, "_max_threads", 1) or 1)

        if max_threads == 1:
            for chunk_coords, out_selection, request in self._iter_chunk_requests(
                backend, chunk_metadata, indexer, missing, method, axis
            ):
                result = backend.reduce_chunk(request)
                self._store_chunk_result(
                    backend,
                    out,
                    counts,
                    axis,
                    need_counts,
                    method,
                    chunk_coords,
                    out_selection,
                    result,
                )
        else:
            with ThreadPoolExecutor(max_workers=max_threads) as executor:
                futures = {
                    executor.submit(backend.reduce_chunk, request): (chunk_coords, out_selection)
                    for chunk_coords, out_selection, request in self._iter_chunk_requests(
                        backend, chunk_metadata, indexer, missing, method, axis
                    )
                }
                for future in as_completed(futures):
                    chunk_coords, out_selection = futures[future]
                    result = future.result()
                    self._store_chunk_result(
                        backend,
                        out,
                        counts,
                        axis,
                        need_counts,
                        method,
                        chunk_coords,
                        out_selection,
                        result,
                    )

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
