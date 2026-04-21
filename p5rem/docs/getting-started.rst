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

* ``python >= 3.10``
* ``pyfive >= 0.5.0``
* ``cbor2``

You can choose a different environment name:

.. code-block:: bash

   ./examples/setup_remote_mamba_env.sh my-remote-env

If your server uses micromamba, override the executable:

.. code-block:: bash

   MAMBA_EXE=micromamba ./examples/setup_remote_mamba_env.sh my-remote-env

The script also accepts ``MAMBA_BIN`` for compatibility.

Then either:

* set ``remote_setup`` to activate the environment and keep
   ``remote_python="python"``, or
* set ``remote_python`` directly to ``conda run -n my-remote-env python``.

Example setup fragments:

* module-based host: ``remote_setup="module load jaspy"`` with
   ``remote_python="python"``
* conda activation host: ``remote_setup="source /path/to/conda.sh && conda activate myenv"``
   with ``remote_python="python"``

Quick start
-----------

The minimal usage pattern is:

1. Call :func:`p5rem.bootstrap_session` to open a session (this launches the
   remote server over SSH).
2. Call :meth:`~p5rem.session.p5remSession.open` to open a remote file.
3. Index into the returned :class:`~p5rem.proxy.rDataset` objects to
   materialise NumPy arrays locally.

.. code-block:: python

   from p5rem import bootstrap_session

   REMOTE_HOST   = "xfer1"            # SSH host alias from ~/.ssh/config
   REMOTE_SETUP  = "module load jaspy"
   REMOTE_PYTHON = "python"
   REMOTE_FILE   = "project/data/temperature.nc"

   with bootstrap_session(
       host=REMOTE_HOST,
       remote_setup=REMOTE_SETUP,
       remote_python=REMOTE_PYTHON,
       login_shell=True,      # needed if conda is only in .bashrc / .bash_profile
       use_cache=False,
   ) as session:
       with session.open(REMOTE_FILE) as f:
           tas = f["tas"]
           print(tas.shape, tas.dtype)
           data = tas[0, :, :]        # fetches only the required HDF5 chunks

``data`` is a local :class:`numpy.ndarray`.  All HDF5 metadata traversal and
chunk I/O happened on the remote host at local fast disk speed.

SSH configuration
-----------------

p5rem uses paramiko to launch the server.  The simplest way to give it
credentials is to configure the target host in ``~/.ssh/config``, 
as in this example::

   Host xfer1
       HostName xfer1.example.ac.uk
       User myusername
       IdentityFile ~/.ssh/id_myid_rsa

and pass ``host="xfer1"`` to :func:`~p5rem.bootstrap.bootstrap_session`.

If your remote Python environment is only activated in shell startup files
(``~/.bashrc`` etc.) you must pass ``login_shell=True``; p5rem will then
invoke the remote command via ``bash --login -c "…"`` so the startup scripts
run first.

Use ``remote_setup`` for environment preparation and ``remote_python`` for the
Python executable itself.  For example, use ``remote_setup="module load ..."``
with ``remote_python="python"`` on module-based systems, or use
``remote_setup="source /path/to/conda.sh && conda activate myenv"`` with
``remote_python="python"`` for explicit conda activation.  As an alternative,
you can skip ``remote_setup`` and set ``remote_python="conda run -n myenv python"``.

Example script
--------------

A runnable example is provided in the repository::

   python examples/read_remote_slice.py

Edit the constants at the top of the file to match your host, environment,
remote file path, variable name, and index selection.
