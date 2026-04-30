Getting Started
===============

Installation
------------

p5rem is a pure-Python package.  Install it from source with pip::

   pip install .

Or in development mode::

   pip install -e .

The client-side dependencies (``paramiko``, ``pyfive``, ``cbor2``,
``diskcache``, ``numpy``) are pulled in automatically.  The remote server stub
(``p5rem/remote_server.py``) requires ``numpy``, ``cbor2``, and the backend
matching the file format you plan to open: ``pyfive`` for HDF5/NetCDF or
``ppfive`` for PP.

Requirements
~~~~~~~~~~~~

* Python ≥ 3.10 (client and server)
* SSH access to the remote host (standard user login, no special privileges)
* A Python environment on the remote host that contains ``numpy``, ``cbor2``,
  and the required file backend: ``pyfive >= 0.5.0`` for HDF5/NetCDF or
  ``ppfive`` for PP

Minimal remote env setup script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This repository includes a helper script to create the minimal remote
environment using mamba:

.. code-block:: bash

   ./examples/setup_remote_mamba_env.sh

By default this creates an environment named ``p5rem-remote`` with:
