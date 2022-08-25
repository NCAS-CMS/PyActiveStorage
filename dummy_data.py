from netCDF4 import Dataset
import numpy as np
import zarr

def _make_data():
    """ 
    Make the actual numpy arrays necessary to save to disk
    """
    data = np.ones((10,10,10))
    dd = np.arange(10)
    
    for i in range(10):
        for j in range(10):
            for k in range(10):
                data[i,j,k] = i + j*10+k*100

    return dd, data


def make_test_ncdata(filename='test_bizarre.nc', chunksize=(3,3,1)):
    """ 
    Make a test dataset which is three dimensional with indices and values that
    aid in testing data extraction. 
    """
    ds = Dataset(filename, 'w', format="NETCDF4")
    dd, data = _make_data()
    
    xdim = ds.createDimension("xdim",10)
    ydim = ds.createDimension("ydim",10)
    zdim = ds.createDimension("zdim",10)
    x = ds.createVariable("x","i4",("xdim",))
    y = ds.createVariable("y","i4",("ydim",))
    z = ds.createVariable("z","i4",("zdim",))

    for a,s in zip([x,y,z],[1,10,100]):
        a[:] = dd*s
    
    dvar = ds.createVariable("data","f8",("xdim","ydim","zdim"), chunksizes=chunksize)
    dvar[:] = data
    
    ds.close()
    
    ds = Dataset(filename,'r')
    var = ds.variables['data']
    print(f'\nCreated file "{filename}" with a variable called "data" with shape {var.shape} and chunking {var.chunking()}\n')


def make_testzarr_variable_file(filename='test.zarr'):
    """ 
    Make a test variable and write to a zarr file.
    #FIXME: Not quite sure how to get the chunking right yet
    """
    dd, data = _make_data()
    zarr.save(filename, x=dd, y=dd*10, z=dd*100, data=data)


if __name__=="__main__":
    make_test_ncdata()

