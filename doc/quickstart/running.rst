.. _running:

*******
Running
*******

Example 1: S3 file
------------------

.. code-block:: python

    import os
    import numpy as np

    from activestorage.active import Active


    def s3_dataset():
        """Run a simple active storage instance test."""
        S3_BUCKET = "bnl"
        storage_options = {
            'key': "KEY",
            'secret': "SECRET",
            'client_kwargs': {'endpoint_url': "https://uor-aces-o.s3-ext.jc.rl.ac.uk"},
        }

        active_storage_url = "https://reductionist.jasmin.ac.uk/"
        a_file = "ch330a.pc19790301-def.nc"

        test_file_uri = os.path.join(
            S3_BUCKET,
            a_file
        )

        active = Active(test_file_uri, ncvar='UM_m01s16i202_vn1106',
                        storage_options=storage_options,
                        active_storage_url=active_storage_url)
        active._version = 2
        result = active.min[0:3, 4:6, 7:9]
