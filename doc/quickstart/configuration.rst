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

Configuring a Reductionist deployment
-------------------------------------

A few points on how to deploy `Reductionist <https://github.com/stackhpc/reductionist-rs>`_ on a Rocky9 cloud machine:

* 99% of the deployment documentation is found in `Reductionist's nice deployment instructions <https://stackhpc.github.io/reductionist-rs/deployment/>`_
* there are a few caveats specific to a pristine Rocky9 (and other distros) deployment though:
* (n00b step) always have a system ``pip`` by installing it with: ``python -m ensurepip --upgrade``
* system Python executable is ``python3`` - you can, of course, ``ln -s`` it to ``python``, or, better, run Ansible pointing it to the correct system Python3:

.. code-block:: bash

    ansible-playbook -i reductionist-rs/deployment/inventory reductionist-rs/deployment/site.yml -e 'ansible_python_interpreter=/usr/bin/python3'

* that call *may result* (as in our case) in an error:

.. code-block:: bash

    TASK [Ensure step RPM is installed] ****************************************************************************************************
    fatal: [localhost]: FAILED! => {"changed": false, "msg": "Failed to validate GPG signature for step-cli-0.24.4-1.x86_64: Package step-cli_0.24.4_amd643z16ickc.rpm is not signed"}

that's because, in our case, we missed the ``step-cli`` package, and a ``dfn`` install is not well liked by the system (it's not ``mamba`` business);
that gets sorted out via `Step's install docs <https://smallstep.com/docs/step-cli/installation>`_:

.. code-block:: bash

    wget https://dl.smallstep.com/cli/docs-cli-install/latest/step-cli_amd64.rpm
    sudo rpm -i step-cli_amd64.rpm
