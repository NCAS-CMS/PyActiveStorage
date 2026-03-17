import os

import numpy as np
import pytest

import activestorage
from requests.exceptions import MissingSchema
from activestorage.active import Active, load_from_https


def test_https():
    """
    Run a https test with a small enough file for the test
    not to be marked as slow. We test all aspects here.
    File: https://esgf.ceda.ac.uk/thredds/fileServer/esg_cmip6/CMIP6/CMIP/MOHC/UKESM1-1-LL/piControl/r1i1p1f2/Amon/ta/gn/latest/ta_Amon_UKESM1-1-LL_piControl_r1i1p1f2_gn_274301-274912.nc
    Size: 75 MiB, variable: ta
    Entire test uses at most 400M RES memory.
    """
    test_file_uri = "https://esgf.ceda.ac.uk/thredds/fileServer/esg_cmip6/CMIP6/CMIP/MOHC/UKESM1-1-LL/piControl/r1i1p1f2/Amon/ta/gn/latest/ta_Amon_UKESM1-1-LL_piControl_r1i1p1f2_gn_274301-274912.nc"
    active_storage_url = "https://reductionist.jasmin.ac.uk/"  # Wacasoft new Reductionist

    # v1: all local
    active = Active(test_file_uri, "ta")
    active._version = 1
    result = active.min()[0:3, 4:6, 7:9]
    print("Result is", result)
    assert result == np.array([220.3180694580078], dtype="float32")

    # v2: declared storage type, no activa storage URL
    active = Active(test_file_uri, "ta",
                    interface_type="https", )
    active._version = 2
    with pytest.raises(MissingSchema):
        result = active.min()[0:3, 4:6, 7:9]

    # v2: declared storage type
    active = Active(test_file_uri, "ta",
                    interface_type="https",
                    active_storage_url=active_storage_url)
    active._version = 2
    result = active.min()[0:3, 4:6, 7:9]
    print("Result is", result)
    assert result == np.array([220.3180694580078], dtype="float32")

    # v2: inferred storage type
    active = Active(test_file_uri, "ta",
                    active_storage_url=active_storage_url)
    active._version = 2
    result = active.min()[0:3, 4:6, 7:9]
    print("Result is", result)
    assert result == np.array([220.3180694580078], dtype="float32")

    # set these as fixed floats
    f_1 = 176.882080078125
    f_2 = 190.227783203125

    # v2: inferred storage type, pop axis
    active = Active(test_file_uri, "ta",
                    interface_type="https",
                    active_storage_url=active_storage_url)
    active._version = 2
    result = active.min(axis=(0, 1))[:]
    print("Result is", result)
    print("Result shape is", result.shape)
    assert result.shape == (1, 1, 144, 192)
    assert result[0, 0, 0, 0] == f_1
    assert result[0, 0, 143, 191] == f_2

    # load dataset with Pyfive
    dataset = load_from_https(test_file_uri)
    av = dataset['ta']
    r_min = np.min(av[:], axis=(0, 1))
    # NOTE the difference in shapes:
    # - Reductionist: (1, 1, 144, 192)
    # - numpy: (144, 192)
    # Contents is identical though.
    print(r_min)
    assert r_min[0, 0] == f_1
    assert r_min[143, 191] == f_2

    # basic auth on; username and password
    # should work with both Active and Reductionist but we
    # don't have such an NGINX-auth-ed file yet
    active = Active(test_file_uri, "ta",
                    interface_type="https",
                    storage_options={"username": None, "password": None},
                    active_storage_url=active_storage_url)
    active._version = 2
    result = active.min(axis=(0, 1))[:]
    print("Result is", result)
    print("Result shape is", result.shape)
    assert result.shape == (1, 1, 144, 192)
    assert result[0, 0, 0, 0] == f_1
    assert result[0, 0, 143, 191] == f_2

    # run with pyfive.Dataset instead of File
    dataset = load_from_https(test_file_uri)
    av = dataset['ta']
    active = Active(av,
                    active_storage_url=active_storage_url)
    active._version = 2
    print("Interface type", active.interface_type)
    result = active.min(axis=(0, 1))[:]
    print("Result is", result)
    print("Result shape is", result.shape)
    assert result.shape == (1, 1, 144, 192)
    assert result[0, 0, 0, 0] == f_1
    assert result[0, 0, 143, 191] == f_2
    

@pytest.mark.skip(
    reason="save time: test_https_implicit_storage is more general.")
def test_https_v1():
    """Run a true test with a https FILE."""
    test_file_uri = "https://esgf.ceda.ac.uk/thredds/fileServer/esg_cmip6/CMIP6/AerChemMIP/MOHC/UKESM1-0-LL/ssp370SST-lowNTCF/r1i1p1f2/Amon/cl/gn/latest/cl_Amon_UKESM1-0-LL_ssp370SST-lowNTCF_r1i1p1f2_gn_205001-209912.nc"

    active = Active(test_file_uri, "cl", interface_type="https")
    active._version = 1
    result = active.min()[0:3, 4:6, 7:9]
    print("Result is", result)
    assert result == np.array([0.6909787], dtype="float32")


@pytest.mark.skip(reason="save time: 2xdata = 2xtime compared to test_https.")
def test_https_v1_100years_file():
    """Run a true test with a https FILE."""
    test_file_uri = "https://esgf.ceda.ac.uk/thredds/fileServer/esg_cmip6/CMIP6/CMIP/MOHC/UKESM1-1-LL/historical/r1i1p1f2/Amon/pr/gn/latest/pr_Amon_UKESM1-1-LL_historical_r1i1p1f2_gn_195001-201412.nc"
    active = Active(test_file_uri, "pr")
    active._version = 1
    result = active.min()[0:3, 4:6, 7:9]
    print("Result is", result)
    assert result == np.array([5.4734613e-07], dtype="float32")


@pytest.mark.slow
def test_https_bigger_file():
    """Run a true test with a https FILE."""
    test_file_uri = "https://esgf.ceda.ac.uk/thredds/fileServer/esg_cmip6/CMIP6/AerChemMIP/MOHC/UKESM1-0-LL/ssp370SST-lowNTCF/r1i1p1f2/Amon/cl/gn/latest/cl_Amon_UKESM1-0-LL_ssp370SST-lowNTCF_r1i1p1f2_gn_205001-209912.nc"
    active_storage_url = "https://reductionist.jasmin.ac.uk/"  # Wacasoft new Reductionist
    active = Active(test_file_uri, "cl", active_storage_url=active_storage_url)
    active._version = 2
    result = active.min()[0:3, 4:6, 7:9]
    print("Result is", result)
    assert result == np.array([0.6909787], dtype="float32")


@pytest.mark.slow
def test_https_implicit_storage():
    """Run a true test with a https FILE."""
    test_file_uri = "https://esgf.ceda.ac.uk/thredds/fileServer/esg_cmip6/CMIP6/AerChemMIP/MOHC/UKESM1-0-LL/ssp370SST-lowNTCF/r1i1p1f2/Amon/cl/gn/latest/cl_Amon_UKESM1-0-LL_ssp370SST-lowNTCF_r1i1p1f2_gn_205001-209912.nc"

    active = Active(test_file_uri, "cl")
    active._version = 1
    result = active.min()[0:3, 4:6, 7:9]
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
        result = active.min()[0:3, 4:6, 7:9]


def test_https_implicit_storage_wrong_url():
    """
    Run a true test with a bogus URL.
    """
    test_file_uri = "https://esgf.cedacow.ac.uk/thredds/fileServer/esg_cmip6/CMIP6/AerChemMIP/MOHC/UKESM1-0-LL/ssp370SST-lowNTCF/r1i1p1f2/Amon/cl/gn/latest/cl_Amon_UKESM1-0-LL_ssp370SST-lowNTCF_r1i1p1f2_gn_205001-209912.nc"

    with pytest.raises(ValueError):
        active = Active(test_file_uri, "cl")
        active._version = 1
        result = active.min[0:3, 4:6, 7:9]


@pytest.mark.skip(
    reason="save time: test_https_dataset_implicit_storage is more general.")
def test_https_dataset():
    """Run a true test with a https DATASET."""
    test_file_uri = "https://esgf.ceda.ac.uk/thredds/fileServer/esg_cmip6/CMIP6/AerChemMIP/MOHC/UKESM1-0-LL/ssp370SST-lowNTCF/r1i1p1f2/Amon/cl/gn/latest/cl_Amon_UKESM1-0-LL_ssp370SST-lowNTCF_r1i1p1f2_gn_205001-209912.nc"
    dataset = load_from_https(test_file_uri)
    av = dataset['cl']

    active = Active(av, interface_type="https")
    active._version = 1
    result = active.min()[0:3, 4:6, 7:9]
    print("Result is", result)
    assert result == np.array([0.6909787], dtype="float32")


@pytest.mark.slow
def test_https_dataset_implicit_storage():
    """Run a true test with a https DATASET."""
    test_file_uri = "https://esgf.ceda.ac.uk/thredds/fileServer/esg_cmip6/CMIP6/AerChemMIP/MOHC/UKESM1-0-LL/ssp370SST-lowNTCF/r1i1p1f2/Amon/cl/gn/latest/cl_Amon_UKESM1-0-LL_ssp370SST-lowNTCF_r1i1p1f2_gn_205001-209912.nc"
    dataset = load_from_https(test_file_uri)
    av = dataset['cl']

    active = Active(av)
    active._version = 1
    result = active.min()[0:3, 4:6, 7:9]
    print("Result is", result)
    assert result == np.array([0.6909787], dtype="float32")
