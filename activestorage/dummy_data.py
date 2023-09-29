import numpy as np

from netCDF4 import Dataset


def _make_data(n=10):
    """ 
    Make the actual numpy arrays necessary to save to disk
    """
    data = np.ones((n, n, n))
    dd = np.arange(n)
    
    nsq = n * n
    for i in range(n):
        for j in range(n):
            for k in range(n):
                data[i, j, k] = i + j * n + k * nsq

    return dd, data

def make_partially_missing_ncdata(filename='test_partially_missing_data.nc',
                                  chunksize=(3,3,1), n=10):
    """ 
    Makes a test dataset based on the default vanilla dataset, but which includes
    half missing values
    """
    return make_ncdata(filename, chunksize, n, compression=None, missing=-999.,
                       partially_missing_data=True)


def make_missing_ncdata(filename='test_missing.nc', chunksize=(3, 3, 1), n=10):
    """ 
    Makes a test dataset based on the default vanilla dataset, but which includes
    some missing values
    """
    return make_ncdata(filename, chunksize, n, compression=None, missing=-999.)


def make_fillvalue_ncdata(filename='test_fillvalue.nc', chunksize=(3,3,1), n=10):
    """ 
    Makes a test dataset based on the default vanilla dataset, but which includes
    some non-default fillvalues.
    """
    ncdat = make_ncdata(filename, chunksize, n, compression=None, fillvalue=-999.)
    return ncdat

def make_validmin_ncdata(filename='test_validmin.nc', chunksize=(3,3,1), n=10, valid_min=200.):
    """ 
    Makes a test dataset based on the default vanilla dataset, but which includes
    missing values below min.
    """
    return make_ncdata(filename, chunksize, n, compression=None, valid_min=valid_min)

def make_validmax_ncdata(filename='test_validmax.nc', chunksize=(3, 3, 1), n=10, valid_max=1.2 * 10 ** 3):
    """ 
    Makes a test dataset based on the default vanilla dataset, but which includes
    missing values above max
    """
    return make_ncdata(filename, chunksize, n, compression=None, valid_max=valid_max)


def make_validrange_ncdata(filename='test_validrange.nc', chunksize=(3, 3, 1), n=10, valid_range=[-1.0, 1.2 * 10 **3]):
    """ 
    Makes a test dataset based on the default vanilla dataset, but which includes
    missing values outside range
    """
    return make_ncdata(filename, chunksize, n, compression=None, valid_range=valid_range)


def make_compressed_ncdata(filename='test_vanilla.nc', chunksize=(3, 3, 1), n=10, compression=None, shuffle=False):
    """
    Make a compressed and optionally shuffled vanilla test dataset which is
    three dimensional with indices and values that aid in testing data
    extraction.
    """
    return make_ncdata(filename, chunksize, n, compression=compression, shuffle=shuffle)


def make_byte_order_ncdata(filename='test_vanilla.nc', chunksize=(3, 3, 1), n=10, byte_order='native'):
    """
    Make a vanilla dataset with the specified byte order (endianness) which is
    three dimensional with indices and values that aid in testing data
    extraction.
    """
    return make_ncdata(filename, chunksize, n, byte_order=byte_order)


def make_vanilla_ncdata(filename='test_vanilla.nc', chunksize=(3, 3, 1), n=10):
    """
    Make a vanilla test dataset which is three dimensional with indices and values that
    aid in testing data extraction.
    """
    return make_ncdata(filename, chunksize, n, None, False)


def make_ncdata(filename, chunksize, n, compression=None, 
                missing=None, 
                fillvalue=None,
                valid_range=None,
                valid_min=None,
                valid_max=None,
                partially_missing_data=False,
                shuffle=False,
                byte_order='native'):
    """ 
    If compression is required, it can be passed in via keyword
    and is applied to all variables.

    Note that if compression is not None, or any of the valid
    data options (missing etc) are selected, then four values
    (for each option) are modified and made invalid. 

    For the purposes of test data, bounds (valid_min, range etc)
    need to be non-zero, although that wont hold in real life.

    partially_missing_data = True makes half the data missing so we can 
    ensure we find some chunks which are all missing ... Can 
    only be used in combination with a missing value.

    shuffle: if True, apply the HDF5 shuffle filter before compression.

    byte_order: Byte order (endianness) of the data. Must be 'big', 'little',
                or 'native'.
    """
    if partially_missing_data and not missing:
        raise ValueError(f'Missing data value keyword provided and set to {missing} '
                         'but partially_missing_data keyword missing from func call.')
    assert n > 4

    ds = Dataset(filename, 'w', format="NETCDF4")
    dd, data = _make_data(n)
    
    xdim = ds.createDimension("xdim", n)
    ydim = ds.createDimension("ydim", n)
    zdim = ds.createDimension("zdim", n)

    dtype_prefix = "<" if byte_order == "little" else ">" if byte_order == "big" else ""
    dim_dtype = dtype_prefix + "i4"
    var_dtype = dtype_prefix + "f8"

    x = ds.createVariable("x",dim_dtype,("xdim",), fill_value=fillvalue, compression=compression, shuffle=shuffle, endian=byte_order)
    y = ds.createVariable("y",dim_dtype,("ydim",), fill_value=fillvalue, compression=compression, shuffle=shuffle, endian=byte_order)
    z = ds.createVariable("z",dim_dtype,("zdim",), fill_value=fillvalue, compression=compression, shuffle=shuffle, endian=byte_order)

    for a,s in zip([x, y, z],[1, n, n * n]):
        a[:] = dd * s

    dvar = ds.createVariable("data",var_dtype, ("xdim","ydim","zdim"),
                             chunksizes=chunksize,
                             compression=compression,
                             shuffle=shuffle,
                             fill_value=fillvalue,
                             endian=byte_order)

    dvar[:] = data

    nm1, nm2 = n - 1, n - 2
    # we use a diffferent set of indices for all the values to be masked
    mindices, findices, vrindices, vm1indices, vm2indices = None, None, None, None, None
    if missing:
        # we use the deprecated missing_value option
        if partially_missing_data:
            dvar[::2,:,:] = missing
            setattr(dvar, "missing_value", missing)
        else:
            mindices = [(1,1,1),(n/2,1,1),(1,nm1,1),(nm1,1,n/2)]
            for ind in mindices:
                for tup in ind:
                    dvar[tup] = missing
            setattr(dvar, "missing_value", missing)

    if fillvalue:
        # note we use a different set of indices for 
        findices = [1, n/2, nm1]
        for ind in findices:
            dvar[ind] = fillvalue
        setattr(dvar, "fill_value", fillvalue)
        
    if valid_range and valid_min or valid_range and valid_max:
        raise ValueError("Can't mix and match validity options")

    if valid_min:
        if valid_min == 0.0:
            raise ValueError('Dummy data needs a non-zero valid min')
        vm1indices = [2, nm1/2, n/2, nm1]
        for ind in vm1indices:
            dvar[ind] = valid_min-abs(0.1*valid_min)
        setattr(dvar, "valid_min", valid_min)
    
    if valid_max:
        if valid_min == 0.0:
            raise ValueError('Dummy data needs a non-zero valid max')
        vm2indices = [2, n/2, nm2, nm1]
        for ind in vm2indices:
            dvar[ind] = valid_max*10
        setattr(dvar, "valid_max", valid_max)

    if valid_range:
        assert len(valid_range) == 2 and type(valid_range[0]) == float
        if valid_range[0] == 0.0 or valid_range[1] == 0.0:
            raise ValueError('Dummy data needs non-zero range bounds')
        vrindices = [(2,nm1,nm2),(2,nm2,nm1),(nm1,nm2,nm1),(n/2,n/2+1,n/2)]
        for i,j,k in vrindices[0:2]:
            dvar[i,j,k]= valid_range[0]-abs(0.1*valid_range[0])
        for i,j,k in vrindices[2:]:
            dvar[i,j,k] = valid_range[1]*10
        setattr(dvar, "valid_range", valid_range)

    var = ds.variables['data']
    print(f'\nCreated file "{filename}" with a variable called "data" with shape {var.shape} and chunking, compression {var.chunking()},{compression}\n')

    # all important close at the end!!
    ds.close()

    return mindices, findices, vrindices, vm1indices, vm2indices


if __name__=="__main__":
    make_vanilla_ncdata()
    make_validmin_ncdata()
    make_validmax_ncdata()
    make_missing_ncdata()
    make_fillvalue_ncdata()
    make_validrange_ncdata()

