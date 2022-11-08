"""Active storage module."""
import numpy as np

from numcodecs.compat import ensure_ndarray

def reduce_chunk(rfile, offset, size, compression, filters, missing, dtype, shape, order, chunk_selection, method=None):
    """ We do our own read of chunks and decoding etc 
    
    rfile - the actual file with the data 
    offset, size - where and what we want ...
    dtype - likely float32 in most cases. 
    shape - will be a tuple, something like (3,3,1), this is the dimensionality of the 
            chunk itself
    order - typically 'C' for c-type ordering
    chunk_selection - python slice tuples for each dimension, e.g.
                        (slice(0, 2, 1), slice(1, 3, 1), slice(0, 1, 1))
                        this defines the part of the chunk which is to be obtained
                        or operated upon.
    method - computation desired 
            (in this Python version it's an actual method, in 
            storage implementations we'll change to controlled vocabulary)
                    
    """
    
    if compression is not None:
        raise NotImplementedError("Compression is not yet supported!")
    if filters is not None:
        raise NotImplementedError("Filters are not yet supported!")

    #FIXME: for the moment, open the file every time ... we might want to do that, or not
    with open(rfile,'rb') as open_file:
        # get the data
        chunk = read_block(open_file, offset, size)
        # make it a numpy array of bytes
        chunk = ensure_ndarray(chunk)
        # convert to the appropriate data type
        chunk = chunk.view(dtype)
        # sort out ordering and convert to the parent hyperslab dimensions
        chunk = chunk.reshape(-1, order='A')
        chunk = chunk.reshape(shape, order=order)

    tmp = chunk[chunk_selection]
    if method:
        if missing != (None, None, None, None):
            tmp = remove_missing(tmp, missing)
            # we are using masked arrays here, so we have to undo that
            # a vanilla implementation of remove_missing which found
            # no valid data would have to do something like this too.
            result = method(tmp)
            if np.ma.is_masked(result):
                # FIXME dodgy
                prct = {False: 1.1, True: 0.9}
                if missing[0]:
                    return missing[0]
                if missing[1]:
                    return missing[1]
                #FIXME how do we avoid fail with over/under flow?
                if missing[2]:
                    norm_factor = prct[missing[2]>0]
                    return missing[2] * norm_factor
                if missing[3]:
                    norm_factor = prct[missing[3]<0]
                    return missing[3] * norm_factor
            else:
                return result, tmp.count()   
        else:
            return method(tmp), tmp.size
    else:
        return tmp, None

def remove_missing(data, missing):
    """ 
    As we are using numpy, we can use a masked array, storage implementations
    will have to do this by hand 
    """
    fill_value, missing_value, valid_min, valid_max = missing

    if fill_value:
        data = np.ma.masked_equal(data, fill_value)
    if missing_value:
        data = np.ma.masked_equal(data, missing_value)
    if valid_max:
        data = np.ma.masked_greater(data, valid_max)
    if valid_min:
        data = np.ma.masked_less(data, valid_min)

    return data

def read_block(open_file, offset, size):
    """ Read <size> bytes from <open_file> at <offset>"""
    place = open_file.tell()
    open_file.seek(offset)
    data = open_file.read(size)
    open_file.seek(place)
    return data
