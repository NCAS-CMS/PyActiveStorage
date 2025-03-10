import os
import numpy as np

from activestorage.active import Active
from activestorage.active import load_from_https


def test_https_dataset():
    """Run a true test with a https dataset."""
    test_file_uri = "https://esgf.ceda.ac.uk/thredds/fileServer/esg_cmip6/CMIP6/AerChemMIP/MOHC/UKESM1-0-LL/ssp370SST-lowNTCF/r1i1p1f2/Amon/cl/gn/latest/cl_Amon_UKESM1-0-LL_ssp370SST-lowNTCF_r1i1p1f2_gn_205001-209912.nc"
    dataset = load_from_https(test_file_uri)
    av = dataset['cl']

    active = Active(av, storage_type="https")
    active._version = 2
    active._method = "min"
    result = active[0:3, 4:6, 7:9]
    print("Result is", result)
    assert result == 5098.625
