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

def make_validmin_ncdata(filename='test_validmin.nc', chunksize=(3,3,1), n=10):
    """ 
    Makes a test dataset based on the default vanilla dataset, but which includes
    missing values below min.
    """
    return make_ncdata(filename, chunksize, n, compression=None, valid_min=-1.)

def make_validmax_ncdata(filename='test_validmax.nc', chunksize=(3, 3, 1), n=10):
    """ 
    Makes a test dataset based on the default vanilla dataset, but which includes
    missing values above max
    """
    return make_ncdata(filename, chunksize, n, compression=None, valid_max=1.2 * n ** 3)


def make_validrange_ncdata(filename='test_validrange.nc', chunksize=(3, 3, 1), n=10):
    """ 
    Makes a test dataset based on the default vanilla dataset, but which includes
    missing values outside range
    """
    return make_ncdata(filename, chunksize, n, compression=None, valid_range=[-1.0, 1.2 * n ** 3])


def make_vanilla_ncdata(filename='test_vanilla.nc', chunksize=(3, 3, 1), n=10):
    """
    Make a vanilla test dataset which is three dimensional with indices and values that
    aid in testing data extraction.
    """
    r = make_ncdata(filename, chunksize, n, None, False)
    return 


def make_ncdata(filename, chunksize, n, compression=None, 
                missing=None, 
                fillvalue=None,
                valid_range=None,
                valid_min=None,
                valid_max=None,
                partially_missing_data=False):
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
    """
    if partially_missing_data and not missing:
        raise ValueError(f'Missing data value keyword provided and set to {missing} '
                         'but partially_missing_data keyword missing from func call.')

    def make_holes(var, indices, attribute, value, dummy):
        if value is not None:
            assert type(value) == float
            setattr(var, attribute, value)
        for i, j, k in indices:
            var[i, j, k] = dummy

        return var

    assert n > 4

    ds = Dataset(filename, 'w', format="NETCDF4")
    dd, data = _make_data(n)
    
    xdim = ds.createDimension("xdim", n)
    ydim = ds.createDimension("ydim", n)
    zdim = ds.createDimension("zdim", n)
    
    x = ds.createVariable("x","i4",("xdim",), fill_value=fillvalue, compression=compression)
    y = ds.createVariable("y","i4",("ydim",), fill_value=fillvalue, compression=compression)
    z = ds.createVariable("z","i4",("zdim",), fill_value=fillvalue, compression=compression)

    for a,s in zip([x, y, z],[1, n, n * n]):
        a[:] = dd * s
    
    dvar = ds.createVariable("data","f8", ("xdim","ydim","zdim"),
                             chunksizes=chunksize, compression=compression)
    dvar[:] = data

    nm1, nm2 = n - 1, n - 2
    # we use a diffferent set of indices for all the values to be masked
    mindices, findices, vrindices, vm1indices, vm2indices = None, None, None, None, None
    if missing:
        # we use the deprecated missing_value option
        if partially_missing_data:
            dvar[::2,:,:] = missing
            dvar.missing_value = missing
        else:
            mindices = [(1,1,1),(n/2,1,1),(1,nm1,1),(nm1,1,n/2)]
            dvar = make_holes(dvar, mindices, 'missing_value', missing, missing)

    if fillvalue:
        # note we use a different set of indices for 
        findices = [(nm1,nm1,nm1),(n/2,n/2,1),(1,1,n/2),(nm1,nm1,n/2)]
        dvar = make_holes(dvar, findices, '_FillValue', None, fillvalue)
        
    if valid_range and valid_min or valid_range and valid_max:
        raise ValueError("Can't mix and match validity options")

    if valid_min:
        if valid_min == 0.0:
            raise ValueError('Dummy data needs a non-zero valid min')
        vm1indices = [(2,2,2),(n/2,2,2),(2,nm1,2),(nm1,2,nm1/2)]
        dvar = make_holes(dvar, vm1indices, 'valid_min', valid_min, valid_min-abs(0.1*valid_min))
    
    if valid_max:
        if valid_min == 0.0:
            raise ValueError('Dummy data needs a non-zero valid max')
        vm2indices = [(2,nm1,2),(2,2,nm1),(nm2,nm2,nm1),(nm1,nm2,n/2)]
        dvar = make_holes(dvar, vm2indices, 'valid_max', valid_max, valid_max*10)

    if valid_range:
        assert len(valid_range) == 2 and type(valid_range[0]) == float
        if valid_range[0] == 0.0 or valid_range[1] == 0.0:
            raise ValueError('Dummy data needs non-zero range bounds')
        vrindices = [(2,nm1,nm2),(2,nm2,nm1),(nm1,nm2,nm1),(n/2,n/2+1,n/2)]
        dvar.valid_range=valid_range
        for i,j,k in vrindices[0:2]:
            dvar[i,j,k]= valid_range[0]-abs(0.1*valid_range[0])
        for i,j,k in vrindices[2:]:
            dvar[i,j,k] = valid_range[1]*10

    ds.close()
    
    ds = Dataset(filename,'r')
    var = ds.variables['data']
    print(f'\nCreated file "{filename}" with a variable called "data" with shape {var.shape} and chunking, compression {var.chunking()},{compression}\n')

    return mindices, findices, vrindices, vm1indices, vm2indices 


if __name__=="__main__":
    make_vanilla_ncdata()
    make_validmin_ncdata()
    make_validmax_ncdata()
    make_missing_ncdata()
    make_fillvalue_ncdata()
    make_validrange_ncdata()

