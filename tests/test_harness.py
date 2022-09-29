import os
import numpy as np
import pytest
import unittest

from activestorage.active import Active
from activestorage.dummy_data import make_test_ncdata


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
        Test a normal read slicing the data an interesting way, using version 1 (replicating native interface in our code)
        """
        active = Active(self.testfile)
        active._version = 1
        var = active['data']
        d = var[0:2,4:6,7:9]
        nda = np.ndarray.flatten(d)
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
        mean_result = np.mean(d)
        active.close()

        active = Active(self.testfile)
        active._version = 2
        active.method=np.mean
        var = active['data']
        result2 = var[0:2,4:6,7:9]
        self.assertEqual(mean_result, result2)
