import os
import numpy as np
import shutil
import tempfile
import unittest

from activestorage.active import Active
from activestorage.dummy_data import make_vanilla_ncdata


class TestActive(unittest.TestCase):
    """ 
    Test basic functionality
    """

    def setUp(self):
        """ 
        Ensure there is test data
        """
        self.temp_folder = tempfile.mkdtemp()
        self.testfile = os.path.join(self.temp_folder,
                                     'test_bizarre.nc')
        print(f"Test file is {self.testfile}")
        if not os.path.exists(self.testfile):
            make_vanilla_ncdata(filename=self.testfile)

    def tearDown(self):
        """Remove temp folder."""
        shutil.rmtree(self.temp_folder)
        
    def testRead0(self):
        """ 
        Test a normal read slicing the data an interesting way, using version 0 (native interface)
        """
        active = Active(self.testfile, 'data')
        active._version = 0
        d = active[0:2,4:6,7:9]
        nda = np.ndarray.flatten(d.data)
        assert np.array_equal(nda,np.array([740.,840.,750.,850.,741.,841.,751.,851.]))

    def testRead1(self):
        """ 
        Test a normal read slicing the data an interesting way, using version 1 (replicating native interface in our code)
        """
        active = Active(self.testfile, 'data')
        active._version = 0
        d0 = active[0:2,4:6,7:9]
        
        active = Active(self.testfile, 'data')
        active._version = 1
        d1 = active[0:2,4:6,7:9]
        assert np.array_equal(d0,d1)

    def testActive(self):
        """ 
        Shows what we expect an active example test to achieve and provides "the right answer"
        """
        active = Active(self.testfile, 'data')
        active._version = 0
        d = active[0:2,4:6,7:9]
        mean_result = np.mean(d)

        active = Active(self.testfile, 'data')
        active.method = "mean"
        result2 = active[0:2,4:6,7:9]
        self.assertEqual(mean_result, result2)

    def testActiveComponents(self):
        """
        Shows what we expect an active example test to achieve and provides "the right answer" 
        """
        active = Active(self.testfile, "data")
        active._version = 0
        d = active[0:2, 4:6, 7:9]
        mean_result = np.mean(d)

        active = Active(self.testfile, "data")
        active._version = 2
        active.method = "mean"
        active.components = True
        result2 = active[0:2, 4:6, 7:9]
        print(result2)
        self.assertEqual(mean_result, result2["sum"]/result2["n"])

