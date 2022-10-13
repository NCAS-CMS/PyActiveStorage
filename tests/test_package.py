import activestorage


# test version
def test_version():
    assert hasattr(activestorage, "__version__")
    assert activestorage.__version__ == "0.0.0"
    print(activestorage.__version__)

# check class
def test_class_attrs():
    assert hasattr(activestorage, "Active")
    assert hasattr(activestorage, "active")
    assert hasattr(activestorage, "storage")
    assert hasattr(activestorage, "netcdf_to_zarr")
