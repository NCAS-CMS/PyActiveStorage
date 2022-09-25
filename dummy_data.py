from netCDF4 import Dataset
import numpy as np
import zarr

def _make_data(n=10):
    """ 
    Make the actual numpy arrays necessary to save to disk
    """
    data = np.ones((n,n,n))
    dd = np.arange(n)
    
    nsq = n*n
    for i in range(n):
        for j in range(n):
            for k in range(n):
                data[i,j,k] = i + j*n+k*nsq

    return dd, data


def make_test_ncdata(filename='test_bizarre.nc', chunksize=(3,3,1), compression=None, n=10):
    """ 
    Make a test dataset which is three dimensional with indices and values that
    aid in testing data extraction. If compression is required, it can be passed in via keyword
    and is applied to all variables
    """
    ds = Dataset(filename, 'w', format="NETCDF4")
    dd, data = _make_data(n)
    
    xdim = ds.createDimension("xdim",n)
    ydim = ds.createDimension("ydim",n)
    zdim = ds.createDimension("zdim",n)
    x = ds.createVariable("x","i4",("xdim",), compression=compression)
    y = ds.createVariable("y","i4",("ydim",), compression=compression)
    z = ds.createVariable("z","i4",("zdim",), compression=compression)

    for a,s in zip([x,y,z],[1,n,n*n]):
        a[:] = dd*s
    
    dvar = ds.createVariable("data","f8",("xdim","ydim","zdim"), chunksizes=chunksize, compression=compression)
    dvar[:] = data
    
    ds.close()
    
    ds = Dataset(filename,'r')
    var = ds.variables['data']
    print(f'\nCreated file "{filename}" with a variable called "data" with shape {var.shape} and chunking, compression {var.chunking()},{compression}\n')


def make_testzarr_variable_file(filename='test.zarr'):
    """ 
    Make a test variable and write to a zarr file.
    #FIXME: Not quite sure how to get the chunking right yet
    """
    dd, data = _make_data()
    zarr.save(filename, x=dd, y=dd*10, z=dd*100, data=data)


if __name__=="__main__":
    make_test_ncdata()

