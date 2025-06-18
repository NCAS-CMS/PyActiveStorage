.. _configuration:

*************
Configuration
*************

Configuring a run with S3 data
------------------------------

As a standalone application, PyActiveStorage does not need any configuration, data being
supplied directly to the ``Active`` class; however, some configuration is still needed to be
able to access an S3 bucket.

For an example, suppose we are trying to access a file ``ch330a.pc19790301-def.nc`` on CEDA-JASMIN's S3 storage,
for which one needs to supply ``Active`` the ``storage_options`` dictionary:

.. code-block:: bash

    storage_options = {
        'key': "key.string",
        'secret': "sercret.string",
        'client_kwargs': {'endpoint_url': "https://uor-aces-o.s3-ext.jc.rl.ac.uk"},
    }

where ``key.string`` and ``secret.string`` are the key and secret strings needed to access the ``S3_BUCKET`` S3 bucket. Then,
the call to ``Active`` is:

.. code-block:: bash

    active = Active("S3_BUCKET/ch330a.pc19790301-def.nc", ncvar='UM_m01s16i202_vn1106',
                    storage_options=storage_options,
                    active_storage_url="https://reductionist.jasmin.ac.uk/")

Note that the `<https://reductionist.jasmin.ac.uk/>`_ specified here is the Reductionist server deployed
on CEDA-JASMIN.
