import botocore
import os
import numpy as np
import pytest

from activestorage.active import Active


def test_s3_active():
    """Test stack when call to Active contains storage_type == s3."""
    active_url = "https://s3.example.com"
    s3_testfile = "s3_test_bizarre.nc"

    with pytest.raises(botocore.exceptions.ParamValidationError):
        Active(os.path.join(active_url, s3_testfile), "data", "s3")
