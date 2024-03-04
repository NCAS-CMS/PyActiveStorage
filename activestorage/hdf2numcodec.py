import numcodecs

def decode_filters(filter_pipeline, name):
    """
    
    Convert HDF5 filter and compression instructions into instructions understood
    by numcodecs. Input is a pyfive filter_pipeline object

    We can only support things that are supported by numcodecs, which one might 
    hope to be a superset of what native pyfive can support.

    See aslso zarr kerchunk _decode_filters. We may need to add much more 
    support for BLOSC than is currently supported by pyfive.

    """
    filters = []

    for filter in filter_pipeline:

        filter_id=filter['filter_id']
        properties = filter['client_data_values']

        if filter_id == GZIP_DEFLATE_FILTER:
            filters.append(numcodecs.Zlib(level=properties[0]))
        elif filter_id == SHUFFLE_FILTER:
            pass
            #FIXME: inherited the pass by inspection from Zarr, pretty sure that's wrong.
        elif filter_id == 32001:
            blosc_compressors = (
                "blosclz",
                "lz4",
                "lz4hc",
                "snappy",
                "zlib",
                "zstd",
            )
            (
                _1,
                _2,
                bytes_per_num,
                total_bytes,
                clevel,
                shuffle,
                compressor,
            ) = properties
            pars = dict(
                blocksize=total_bytes,
                clevel=clevel,
                shuffle=shuffle,
                cname=blosc_compressors[compressor],
            )
            filters.append(numcodecs.Blosc(**pars))
        elif filter_id == 32015:
            filters.append(numcodecs.Zstd(level=properties[0]))
        elif filter_id == 32004:
            raise RuntimeError(
                f"{name} uses lz4 compression - not supported as yet"
            )
        elif filter_id == 32008:
            raise RuntimeError(
                f"{name} uses bitshuffle compression - not supported as yet"
            )
        else:
            raise RuntimeError(
                f"{name} uses filter id {filter_id} with properties {properties},"
                f" not supported as yet"
            )
    return filters

# These are from pyfive's HDF5 filter definitions
# IV.A.2.l The Data Storage - Filter Pipeline message
RESERVED_FILTER = 0
GZIP_DEFLATE_FILTER = 1
SHUFFLE_FILTER = 2
FLETCH32_FILTER = 3
SZIP_FILTER = 4
NBIT_FILTER = 5
SCALEOFFSET_FILTER = 6
