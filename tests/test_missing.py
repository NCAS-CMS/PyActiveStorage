import os
from this import d
import numpy as np
import shutil
import tempfile
import unittest

from activestorage.active import Active
from activestorage.dummy_data import make_pathological_ncdata, make_vanilla_ncdata

class TestActive(unittest.TestCase):
    """ 
    Test basic functionality
    """

    def setUp(self):
        """ 
        Ensure there is test data
        """
        self.temp_folder = tempfile.mkdtemp()

    def tearDown(self):
        """Remove temp folder."""
        shutil.rmtree(self.temp_folder)


    def _doit(self, testfile):
        """ 
        Compare and contrast vanilla mean with actual means
        """
        active = Active(testfile, "data")
        active._version = 0
        d = active[0:2, 4:6, 7:9]
        mean_result = np.mean(d)

        active = Active(testfile, "data")
        active._version = 2
        active.method = "mean"
        active.components = True
        result2 = active[0:2, 4:6, 7:9]
        self.assertEqual(mean_result, result2["sum"]/result2["n"])


    def test_pathological_data(self):
        testfile = os.path.join(self.temp_folder, 'test_pathological_missing.nc')
        r = make_pathological_ncdata(testfile)
        self._doit(testfile)

    def test_missing(self):
        testfile = os.path.join(self.temp_folder, 'test_missing.nc')
        r = make_pathological_ncdata(testfile)
        self._doit(testfile)


