import os
import pytest

import numpy as np
from netCDF4 import Dataset
from pathlib import Path
import s3fs

from activestorage.active import Active
from activestorage.config import *


def mytest():
    """
    Test again a native model, this time around netCDF4 loadable with h5py
    Also, this has _FillValue and missing_value
    """
    ncfile, v = "cesm2_native.nc","TREFHT"
    ncfile, v  = "CMIP6-test.nc", 'tas'
    #ncfile, v = "chunked.hdf5", "dataset1"
    ncfile, v = 'daily_data.nc', 'ta'
    mypath = Path(__file__).parent
    uri = str(mypath/ncfile)
    active = Active(uri, v, None)
    active._version = 0
    if v == "dataset1":
        d = active[2,:]
    else:
        d = active[4:5, 1:2]
    mean_result = np.mean(d)
    active = Active(uri, v, None)
    active._version = 2
    active.method = "mean"
    active.components = True
    if  v == "dataset1":
        result2 = active[2,:]
    else:
        result2 = active[4:5, 1:2]
    print(result2, ncfile)
    # check for active
    np.testing.assert_allclose(mean_result, result2["sum"]/result2["n"], rtol=1e-6)


if __name__=="__main__":
    mytest()