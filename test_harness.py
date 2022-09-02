from ast import Not
import unittest
import os
from dummy_data import make_test_ncdata
from netCDF4 import Dataset
import numpy as np
import zarr

from zarr import core as zarrcore
from PCI_wrapper import load_netcdf_zarr_generic


class Active:
    """ 
    Instantiates an interface to active storage which contains either zarr files
    or HDF5 (NetCDF4) files.

    This is Verson 1 which simply provides support for standard read operations, but done via
    explicit reads within this class rather than within the underlying format libraries.
    
    Version 2 will add methods for actual active storage.

    """
    def __init__(self,uri,*args,**kwargs):
        """ Instantiate in the same way as normal """
        # Assume NetCDF4 for now
        self.file = Dataset(uri)
        self._version = 1 
        self.method = None

    def __getitem__(self, *args):
        """ 
        Provides support for a standard get item.
        """ 
        # In version one this is done by explicitly looping over each chunk in the file
        # and returning the requested slice ourselves. In version 2, we can pass this
        # through to the default method.
        if self.method is not None and self._version != 2:
            raise ValueError(f'Cannot apply active storage with this version ({self._version}) of the getitem method')
        if self._version == 0:
            return self.file.__getitem__(*args)
        elif self._version == 1:
            raise NotImplementedError
        elif self._version  == 2:
            raise NotImplementedError
        else:
            raise ValueError(f'Version {self._version} not supported')

    def method(self, method):
        """ Set the method for any future get items"""
        self.method = method

    def _get_active(self, method, *args):
        """ 
        *args defines a slice of data. This method loops over each of the chunks
        necessary to extract the parts of the slice, and asks the active storage 
        to apply the method to each part. It then applies the method to 
        the partial results and returns a value is if  method had been applied to
        an array returned via getitem.
        """
        raise NotImplementedError


    def close(self):
        self.file.close()

class TestActive(unittest.TestCase):
    """ 
    Test basic functionality
    """

    def setUp(self):
        """ 
        Ensure there is test data
        """
        self.testfile = 'test_bizarre.nc'
        if not os.path.exists(self.testfile):
            make_test_ncdata(filename=self.testfile)
        
    def testRead0(self):
        """ 
        Test a normal read slicing the data an interesting way, using version 0 (native interface)
        """
        active = Active(self.testfile)
        active._version = 0
        var = active['data']
        d = var[0:2,4:6,7:9]
        nda = np.ndarray.flatten(d.data)
        assert np.array_equal(nda,np.array([740.,840.,750.,850.,741.,841.,751.,851.]))
        active.close()

    def testRead1(self):
        """ 
        Test a normal read slicing the data an interesting way, uing version 1 (replicating native interface in our code)
        """
        active = Active(self.testfile)
        active._version = 1
        var = active['data']
        d = var[0:2,4:6,7:9]
        nda = np.ndarray.flatten(d.data)
        assert np.array_equal(nda,np.array([740.,840.,750.,850.,741.,841.,751.,851.]))
        active.close()

    def testActive(self):
        """ 
        Shows what we expect an active example test to achieve and provides "the right answer" 
        """
        active = Active(self.testfile)
        active._version = 0
        var = active['data']
        d = var[0:2,4:6,7:9]
        nda = np.ndarray.flatten(d.data)
        mean_result = np.mean(nda)
        active.close()

        active = Active(self.testfile)
        active._version = 2
        active.method='mean'
        result2 = var['data'][0:2,4:6,7:9]
        assert mean_result == result2

    def test_zarr_hijack(self):
        """ 
        Test the hijacking of Zarr. 
        """
        data_file = self.testfile
        varname = "test_bizarre"

        # load without data payload and apply a selection
        ds = load_netcdf_zarr_generic(data_file, varname)
        selection = (slice(0, 2, 1), slice(4, 6, 1), slice(7, 9, 1))
        data_selection, chunk_info = \
            zarrcore.Array.get_orthogonal_selection(ds, selection,
                                                    out=None, fields=None,
                                                    compute_data=False)

        # get slicing info
        _, _, PCI = chunk_info[0]
        PCI = list(PCI)
        offsets = []
        sizes = []
        for offset, size, _ in PCI:
            offsets.append(offset)
            sizes.append(size)

        # sanity checks
        assert len(offsets) == 4
        assert len(sizes) == len(offsets)
        assert offsets == [47, 57, 147, 157]
        assert sizes == [2, 2, 2, 2]

        # compute a mean
        nda = np.ndarray.flatten(ds[:][0])
        mean_result = np.mean(nda)
        assert np.isnan(mean_result)  # has to be; no data passed


if __name__=="__main__":
    unittest.main()
