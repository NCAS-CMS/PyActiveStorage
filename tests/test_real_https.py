import os
import numpy as np
import activestorage
import pytest

from activestorage.active import Active
from activestorage.active import load_from_https


@pytest.mark.skip(reason="save time: test_https_implicit_storage is more general.")
def test_https():
    """Run a true test with a https FILE."""
    test_file_uri = "https://esgf.ceda.ac.uk/thredds/fileServer/esg_cmip6/CMIP6/AerChemMIP/MOHC/UKESM1-0-LL/ssp370SST-lowNTCF/r1i1p1f2/Amon/cl/gn/latest/cl_Amon_UKESM1-0-LL_ssp370SST-lowNTCF_r1i1p1f2_gn_205001-209912.nc"

    active = Active(test_file_uri, "cl", storage_type="https")
    active._version = 1
    result = active.min[0:3, 4:6, 7:9]
    print("Result is", result)
    assert result == np.array([0.6909787], dtype="float32")


@pytest.mark.skip(reason="save time: 2xdata = 2xtime compared to test_https.")
def test_https_100years():
    """Run a true test with a https FILE."""
    test_file_uri = "https://esgf.ceda.ac.uk/thredds/fileServer/esg_cmip6/CMIP6/CMIP/MOHC/UKESM1-1-LL/historical/r1i1p1f2/Amon/pr/gn/latest/pr_Amon_UKESM1-1-LL_historical_r1i1p1f2_gn_195001-201412.nc"
    active = Active(test_file_uri, "pr")
    active._version = 1
    result = active.min[0:3, 4:6, 7:9]
    print("Result is", result)
    assert result == np.array([5.4734613e-07], dtype="float32")


# this could be a slow test on GHA depending on network load
# also Githb machines are very far from Oxford
@pytest.mark.slow
def test_https_reductionist():
    """Run a true test with a https FILE."""
    test_file_uri = "https://esgf.ceda.ac.uk/thredds/fileServer/esg_cmip6/CMIP6/AerChemMIP/MOHC/UKESM1-0-LL/ssp370SST-lowNTCF/r1i1p1f2/Amon/cl/gn/latest/cl_Amon_UKESM1-0-LL_ssp370SST-lowNTCF_r1i1p1f2_gn_205001-209912.nc"

    # added storage_type in request_data dict; Reductionist not liking it
    # E           activestorage.reductionist.ReductionistError: Reductionist error: HTTP 400: {"error": {"message": "request data is not valid", "caused_by": ["Failed to deserialize the JSON body into the target type", "storage_type: unknown field `storage_type`, expected one of `source`, `bucket`, `object`, `dtype`, `byte_order`, `offset`, `size`, `shape`, `order`, `selection`, `compression`, `filters`, `missing` at line 1 column 550"]}}
    with pytest.raises(activestorage.reductionist.ReductionistError):
        active = Active(test_file_uri, "cl")
        active._version = 2
        result = active.min[0:3, 4:6, 7:9]
        print("Result is", result)
        assert result == np.array([0.6909787], dtype="float32")


# this could be a slow test on GHA depending on network load
# also Githb machines are very far from Oxford
@pytest.mark.slow
def test_https_implicit_storage():
    """Run a true test with a https FILE."""
    test_file_uri = "https://esgf.ceda.ac.uk/thredds/fileServer/esg_cmip6/CMIP6/AerChemMIP/MOHC/UKESM1-0-LL/ssp370SST-lowNTCF/r1i1p1f2/Amon/cl/gn/latest/cl_Amon_UKESM1-0-LL_ssp370SST-lowNTCF_r1i1p1f2_gn_205001-209912.nc"

    active = Active(test_file_uri, "cl")
    active._version = 1
    result = active.min[0:3, 4:6, 7:9]
    print("Result is", result)
    assert result == np.array([0.6909787], dtype="float32")


def test_https_implicit_storage_file_not_found():
    """
    Run a true test with a https FILE that is not found.
    Code raises a very descriptive exception via fsspec.
    Keep test to capture any changes in behaviour.
    """
    test_file_uri = "https://esgf.ceda.ac.uk/thredds/fileServer/esg_cmip6/CMIP6/AerChemMIP/MOHC/UKESM1-0-LL/ssp370SST-lowNTCF/r1i1p1f2/Amon/cl/gn/latest/cl_Amon_UKESM1-0-LL_ssp370SST-lowNTCF_r1i1p1f2_gn_205001-209912.ncx"

    with pytest.raises(FileNotFoundError):
        active = Active(test_file_uri, "cl")
        active._version = 1
        result = active.min[0:3, 4:6, 7:9]


def test_https_implicit_storage_wrong_url():
    """
    Run a true test with a bogus URL.
    """
    test_file_uri = "https://esgf.cedacow.ac.uk/thredds/fileServer/esg_cmip6/CMIP6/AerChemMIP/MOHC/UKESM1-0-LL/ssp370SST-lowNTCF/r1i1p1f2/Amon/cl/gn/latest/cl_Amon_UKESM1-0-LL_ssp370SST-lowNTCF_r1i1p1f2_gn_205001-209912.nc"

    with pytest.raises(ValueError):
        active = Active(test_file_uri, "cl")
        active._version = 1
        result = active.min[0:3, 4:6, 7:9]


@pytest.mark.skip(reason="save time: test_https_dataset_implicit_storage is more general.")
def test_https_dataset():
    """Run a true test with a https DATASET."""
    test_file_uri = "https://esgf.ceda.ac.uk/thredds/fileServer/esg_cmip6/CMIP6/AerChemMIP/MOHC/UKESM1-0-LL/ssp370SST-lowNTCF/r1i1p1f2/Amon/cl/gn/latest/cl_Amon_UKESM1-0-LL_ssp370SST-lowNTCF_r1i1p1f2_gn_205001-209912.nc"
    dataset = load_from_https(test_file_uri)
    av = dataset['cl']

    active = Active(av, storage_type="https")
    active._version = 1
    result = active.min[0:3, 4:6, 7:9]
    print("Result is", result)
    assert result == np.array([0.6909787], dtype="float32")


# this could be a slow test on GHA depending on network load
# also Githb machines are very far from Oxford
@pytest.mark.slow
def test_https_dataset_implicit_storage():
    """Run a true test with a https DATASET."""
    test_file_uri = "https://esgf.ceda.ac.uk/thredds/fileServer/esg_cmip6/CMIP6/AerChemMIP/MOHC/UKESM1-0-LL/ssp370SST-lowNTCF/r1i1p1f2/Amon/cl/gn/latest/cl_Amon_UKESM1-0-LL_ssp370SST-lowNTCF_r1i1p1f2_gn_205001-209912.nc"
    dataset = load_from_https(test_file_uri)
    av = dataset['cl']

    active = Active(av)
    active._version = 1
    result = active.min[0:3, 4:6, 7:9]
    print("Result is", result)
    assert result == np.array([0.6909787], dtype="float32")
