import os
import zarr

def build_zarr_dataset():
    """Create a zarr array and save it."""
    dsize = (60, 404, 802)
    dchunks = (12, 80, 160)
    dvalue = 42.
    store = zarr.DirectoryStore('data/array.zarr')
    z = zarr.zeros(dsize, chunks=dchunks, store=store, overwrite=True)
    z[...] = dvalue
    zarr.save_array("example.zarr", z, compressor=None)


def PCI_wrapper_standalone(chunk_selection):
    """Build a wrapper of PartialChunkIterator and run it as standalone."""
    from zarr import core
    zarr_dir = "./example.zarr"
    if not os.path.isdir(zarr_dir):
        build_zarr_dataset()
    ds = zarr.open("./example.zarr")
    # functional arguments example
    # chunk_selection: (slice(0, 2, 1), slice(0, 4, 1), slice(0, 2, 1))
    pc = core.Array._process_chunk_V(ds, chunk_selection)

    return pc


def PCI_wrapper(selection):
    """Use all the Zarr ins."""
    from zarr import core
    zarr_dir = "./example.zarr"
    if not os.path.isdir(zarr_dir):
        build_zarr_dataset()
    ds = zarr.open("./example.zarr")
    pc = core.Array.get_orthogonal_selection(ds, selection,
                                             out=None, fields=None)
    return pc


def main():
    """Extract the needed info straight from Zarr."""
    # pass a chunk_selection and run the standalone function
    chunk_selection = (slice(0, 2, 1), slice(0, 4, 1), slice(0, 2, 1))
    chunks, chunk_sel, PCI = PCI_wrapper_standalone(chunk_selection)
    print("Running the PCI wrapper as standalone:")
    print(f"Chunks {chunks}, chunk selection {chunk_sel}, PCI {list(PCI)}\n")

    # integrate it in the Zarr mechanism
    data_selection, chunk_info = PCI_wrapper(chunk_selection)
    print("Running the PCI wrapper integrated in Zarr.Array:")
    print(f"Data selection {data_selection}")
    chunks, chunk_sel, PCI = chunk_info[0]
    print(f"Chunks {chunks}, chunk selection {chunk_sel}, PCI {list(PCI)}\n")
        


if __name__ == '__main__':
    main()
