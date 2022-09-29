import h5py
import itertools
import numpy as np
import os
import zarr

from io import BytesIO
from kerchunk.hdf import SingleHdf5ToZarr
from zarr.indexing import PartialChunkIterator


dsize = (60, 404, 802)
dchunks = (12, 80, 160)
dvalue = 42.

docstr = r"""
This code uses zarr.indexing.PartialChunkIterator:

Iterator to retrieve the specific coordinates of requested data
    from within a compressed chunk.

    Parameters
    ----------
    selection : tuple
        tuple of slice objects to take from the chunk
    arr_shape : shape of chunk to select data from

    Attributes
    -----------
    arr_shape
    selection

    Returns
    -------
    Tuple with 3 elements:

    start: int
        elements offset in the chunk to read from
    nitems: int
        number of elements to read in the chunk from start
    partial_out_selection: list of slices
        indices of a temporary empty array of size `Array._chunks` to assign
        the decompressed data to after the partial read.

    Notes
    -----
    An array is flattened when compressed with blosc, so this iterator takes
    the wanted selection of an array and determines the wanted coordinates
    of the flattened, compressed data to be read and then decompressed. The
    decompressed data is then placed in a temporary empty array of size
    `Array._chunks` at the indices yielded as partial_out_selection.
    Once all the slices yielded by this iterator have been read, decompressed
    and written to the temporary array, the wanted slice of the chunk can be
    indexed from the temporary array and written to the out_selection slice
    of the out array.
"""
print(docstr)      

def build_zarr_dataset():
    """Create a zarr array and save it."""
    store = zarr.DirectoryStore('data/array.zarr')
    z = zarr.zeros(dsize, chunks=dchunks, store=store, overwrite=True)
    z[...] = dvalue
    zarr.save_array("example.zarr", z, compressor=None)


def h5py_chunk_slice_info():
    """Use h5py and zarr utility to get chunk/slice info."""
    buf = BytesIO()
    with h5py.File(buf, 'w') as fout:
        fout.create_dataset('test', shape=dsize,
                            chunks=dchunks, dtype='f8')
        fout['test'][:] = dvalue
    buf.seek(0)

    with h5py.File(buf, 'r') as fin:
        ds = fin['test']
        ds_id = fin['test'].id
        num_chunks = ds_id.get_num_chunks()
        print("\n")
        print("H5Py stuffs")
        print("==================")
        print(f"Dataset number of chunks is {num_chunks}")
        print("\nAnalyzing 0th and 5th chunks")
        for j in [0, 5]:
            print(f"Chunk index {j}")
            si = ds_id.get_chunk_info(j)  # return a StoreInfo object
            # si has attrs: 'byte_offset', 'chunk_offset', 'count',
            # 'filter_mask', 'index', 'size'
            print("Chunk index offset", si.chunk_offset)
            print("Chunk byte offset", si.byte_offset)
            print("Chunk size", si.size)

        tot_size = ds.size * ds.dtype.itemsize
        print(f"\nTotal chunks size {tot_size}")

        # slice this cake
        print("\nNow looking at some slices:")
        data_slice = ds[0:2]  # get converted to an ndarray
        print(f"Slice Dataset[0:2] shape {data_slice.shape}")
        # use zarr.indexing.PartialChunkIterator
        PCI = PartialChunkIterator((slice(0, 2, 2), ), ds.shape)
        print("Slice offset and size:", list(PCI)[0][0], "and",
              list(PCI)[0][1] * ds.dtype.itemsize)
        print("\n")
        data_slice = ds[4:7]
        print(f"Slice Dataset[4:7] shape {data_slice.shape}")
        PCI = PartialChunkIterator((slice(4, 7, 1), ), ds.shape)
        print("Slice offset and size", list(PCI)[0][0], "and",
              list(PCI)[0][1] * ds.dtype.itemsize)
        print("\n")
        data_slice = ds[0:60]  # the whole cake
        print(f"Slice Dataset[0:60] shape {data_slice.shape}")
        PCI = PartialChunkIterator((slice(0, 60, 1), ), ds.shape)
        print("Slice offset and size", list(PCI)[0][0], "and",
              list(PCI)[0][1] * ds.dtype.itemsize)
        print("\n")

        # kerchunk it!
        ds = SingleHdf5ToZarr(buf).translate()
        print("\nKerchunk-IT stuffs")
        print("======================")
        no_chunks = len(ds["refs"].keys()) - 3
        print(f"Dataset number of chunks is {no_chunks}")
        print(f"(0, 0, 0) Chunk: offset and size:", ds["refs"]["test/0.0.0"][1], ds["refs"]["test/0.0.0"][2])
        print(f"(0, 0, 5) Chunk: offset and size:", ds["refs"]["test/0.0.5"][1], ds["refs"]["test/0.0.5"][2])
        # print(f"(0, 5, 0) Chunk: offset and size:", ds["refs"]["test/0.5.0"][1], ds["refs"]["test/0.5.0"][2])
        chunk_sizes = []
        for val in ds["refs"].values():
            if isinstance(val[2], int) or isinstance(val[2], float):
                chunk_sizes.append(val[2])
        print("Min chunk size", np.min(chunk_sizes))
        print("Max chunk size", np.max(chunk_sizes))
        print("Total size (sum of chunks), UNCOMPRESSED:", np.sum(chunk_sizes))
        print("\n")


def zarr_chunk_slice_info():
    """Use pure zarr insides to get chunk/slice info."""
    zarr_dir = "./example.zarr"
    if not os.path.isdir(zarr_dir):
        build_zarr_dataset()
    ds = zarr.open("./example.zarr")
    print("Zarr stuffs")
    print("==================")
    print(f"Data file loaded by Zarr\n: {ds}")
    print(f"Info of Data file loaded by Zarr\n: {ds.info}")
    # print(f"Data array loaded by Zarr\n: {ds[:]}")
    print(f"Data chunks: {ds.chunks}")

    # Zarr chunking information
    # from zarr.convenience._copy(); convenience module l.897
    # https://zarr.readthedocs.io/en/stable/api/convenience.html#zarr.convenience.copy
    shape = ds.shape
    chunks = ds.chunks
    chunk_offsets = [range(0, s, c) for s, c in zip(shape, chunks)]
    print("Chunk offsets", [tuple(k) for k in chunk_offsets])
    print("Zarr number of chunks", len(list(itertools.product(*chunk_offsets))))
    offsets = []  # index offsets
    sels = []  # indices
    ch_sizes = []  # chunk sizes
    for offset in itertools.product(*chunk_offsets):
        offsets.append(offset)
        sel = tuple(slice(o, min(s, o + c))
                    for o, s, c in zip(offset, shape, chunks))
        sels.append(sel)
        islice = ds[sel]
        slice_size = islice.size * islice.dtype.itemsize
        ch_sizes.append(slice_size)

    print("\nAnalyzing 0th and 5th chunks")
    for j in [0, 5]:
        print(f"Chunk index {j}")
        print("Chunk index offset:", offsets[j])
        print("Chunk position:", sels[j])
        print("Chunk size:", ch_sizes[j])

    print("\nChunks information")
    print("Min chunk size:", np.min(ch_sizes))
    print("Max chunk size:", np.max(ch_sizes))
    print("Total chunks size COMPRESSED:", np.sum(ch_sizes))

    tot_size = ds.size * ds.dtype.itemsize
    print(f"\nTotal size (sum of chunks), COMPRESSED: {tot_size}")

    # slice this cake
    print("\nNow looking at some slices:")
    data_slice = ds[0:2]  # zarr data slice
    print(f"Slice Dataset[0:2] shape {data_slice.shape}")
    PCI = PartialChunkIterator((slice(0, 2, 1), ), ds.shape)
    print("Slice offset and size", list(PCI)[0][0], "and",
          list(PCI)[0][1] * ds.dtype.itemsize)
    print("\n")
    data_slice = ds[4:7]  # zarr data slice
    print(f"Slice Dataset[4:7] shape {data_slice.shape}")
    PCI = PartialChunkIterator((slice(4, 7, 1), ), ds.shape)
    print("Slice offset and size", list(PCI)[0][0], "and",
          list(PCI)[0][1] * ds.dtype.itemsize)
    print("\n")
    data_slice = ds[0:60]  # the whole cake
    print(f"Slice Dataset[0:60] shape {data_slice.shape}")
    PCI = PartialChunkIterator((slice(0, 60, 1), ), ds.shape)
    print("Slice offset and size", list(PCI)[0][0], "and",
          list(PCI)[0][1] * ds.dtype.itemsize)
    print("\n")


def main():
    "Run the meat."""
    h5py_chunk_slice_info()
    zarr_chunk_slice_info()


if __name__ == '__main__':
    main()
