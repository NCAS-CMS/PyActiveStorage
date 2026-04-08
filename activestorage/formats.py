from __future__ import annotations

import numpy as np
import pyfive

from activestorage.core import ChunkMetadata, MissingAttributes, StorageFormat
from activestorage.hdf2numcodec import decode_filters
from activestorage.helpers import get_missing_attributes


class PyfiveFormat(StorageFormat):
    def __init__(self):
        self._dataset_file = None
        self._dataset = None
        self._uri = None

    def open(self, uri, storage_options):
        self._uri = str(uri)
        scheme = ""
        if "://" in self._uri:
            scheme = self._uri.split("://", 1)[0]
        if scheme == "s3":
            from activestorage import active as active_module

            self._dataset_file = active_module.load_from_s3(self._uri, storage_options)
        else:
            self._dataset_file = pyfive.File(self._uri)

    def get_variable(self, ncvar):
        self._dataset = self._dataset_file[ncvar]
        return self._dataset

    def get_missing_attributes(self) -> MissingAttributes:
        missing = get_missing_attributes(self._dataset)
        return MissingAttributes(*missing)

    def get_indexer(self, selection):
        if not isinstance(selection, tuple):
            selection = (selection,)
        array = pyfive.indexing.ZarrArrayStub(self._dataset.shape, self._dataset.chunks)
        return pyfive.indexing.OrthogonalIndexer(selection, array)

    def get_chunk_metadata(self) -> ChunkMetadata:
        dataset = self._dataset
        ds = dataset.id
        dtype = np.dtype(dataset.dtype)
        array = pyfive.indexing.ZarrArrayStub(dataset.shape, dataset.chunks)
        if ds.filter_pipeline is None:
            compressor, filters = None, None
        else:
            compressor, filters = decode_filters(ds.filter_pipeline, dtype.itemsize, dataset.name)

        return ChunkMetadata(
            dtype=dtype,
            shape=dataset.shape,
            chunks=array._chunks,
            compressor=compressor,
            filters=filters,
            order=ds._order,
            filename=self._uri,
        )

    def get_chunk_offset_size(self, chunk_coords):
        storeinfo = self._dataset.id.get_chunk_info_from_chunk_coord(chunk_coords)
        return storeinfo.byte_offset, storeinfo.size

    @property
    def file_handle(self):
        return self._dataset.id._fh

    def close(self):
        return None


class ZarrFormat(StorageFormat):
    def __init__(self):
        self._array = None

    def open(self, uri, storage_options):
        import zarr

        self._array = zarr.open(uri, mode="r")

    def get_variable(self, ncvar):
        return self._array

    def get_missing_attributes(self) -> MissingAttributes:
        attrs = self._array.attrs
        return MissingAttributes(
            attrs.get("_FillValue"),
            attrs.get("missing_value"),
            attrs.get("valid_min"),
            attrs.get("valid_max"),
        )

    def get_indexer(self, selection):
        if not isinstance(selection, tuple):
            selection = (selection,)
        return pyfive.indexing.OrthogonalIndexer(selection, self._array)

    def get_chunk_metadata(self) -> ChunkMetadata:
        return ChunkMetadata(
            dtype=np.dtype(self._array.dtype),
            shape=self._array.shape,
            chunks=self._array.chunks,
            compressor=getattr(self._array, "compressor", None),
            filters=getattr(self._array, "filters", None),
            order=getattr(self._array, "order", "C"),
            filename="",
        )

    def close(self):
        return None


class P5RemFormat(StorageFormat):
    def __init__(self):
        self._file = None
        self._dataset = None

    def open(self, uri, storage_options):
        self._file = uri

    def get_variable(self, ncvar):
        self._dataset = self._file[ncvar]
        return self._dataset

    def get_missing_attributes(self) -> MissingAttributes:
        attrs = self._dataset.attrs
        return MissingAttributes(
            attrs.get("_FillValue"),
            attrs.get("missing_value"),
            attrs.get("valid_min"),
            attrs.get("valid_max"),
        )

    def get_indexer(self, selection):
        if not isinstance(selection, tuple):
            selection = (selection,)
        array = pyfive.indexing.ZarrArrayStub(self._dataset.shape, self._dataset.chunks)
        return pyfive.indexing.OrthogonalIndexer(selection, array)

    def get_chunk_metadata(self) -> ChunkMetadata:
        return ChunkMetadata(
            dtype=np.dtype(self._dataset.dtype),
            shape=self._dataset.shape,
            chunks=self._dataset.chunks,
            compressor=None,
            filters=None,
            order="C",
            filename="",
        )

    def close(self):
        return None


class KerchunkFormat(ZarrFormat):
    pass
