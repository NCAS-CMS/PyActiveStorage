.. _running:

*******
Running
*******

Example 1: S3 file
------------------

Here is a basic example of of obtaining a minumum on a slice ``[0:3, 4:6, 7:9]`` on the data for a variable
``UM_m01s16i202_vn1106`` in a file ``ch330a.pc19790301-def.nc`` hosted on JASMIN's S3 storage in a bucket called ``bnl``:

.. code-block:: python

    import os

    from activestorage.active import Active


    def s3_file():
        """Run a simple active storage instance test with S3 file."""
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

        return result  # 5098.625

Example 2: HTTPS file
---------------------

Same as above, only the file is stored on an HTTPS-facing server (NGINX-enabled):

.. code-block:: python

    from activestorage.active import Active


    def https_file():
        """Run a simple active storage instance test with https file."""
        test_file_uri = "https://esgf.ceda.ac.uk/thredds/fileServer/esg_cmip6/CMIP6/AerChemMIP/MOHC/UKESM1-0-LL/ssp370SST-lowNTCF/r1i1p1f2/Amon/cl/gn/latest/cl_Amon_UKESM1-0-LL_ssp370SST-lowNTCF_r1i1p1f2_gn_205001-209912.nc"

        active = Active(test_file_uri, ncvar="cl")
        active._version = 1
        result = active.min[0:3, 4:6, 7:9]

        return result  # numpy.array([0.6909787], dtype="float32")
