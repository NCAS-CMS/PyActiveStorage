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
(``p5rem/remote_server.py``) only requires ``pyfive`` and ``numpy``, which
keeps the per-environment HPC footprint small.

Requirements
~~~~~~~~~~~~

* Python ≥ 3.10 (client and server)
* SSH access to the remote host (standard user login, no special privileges)
* A Python environment on the remote host that contains ``pyfive ≥ 0.5.0``
  and ``numpy``

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
   REMOTE_PYTHON = "conda run -n myenv python"
   REMOTE_FILE   = "project/data/temperature.nc"

   with bootstrap_session(
       host=REMOTE_HOST,
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

Example script
--------------

A runnable example is provided in the repository::

   python examples/read_remote_slice.py

Edit the constants at the top of the file to match your host, environment,
remote file path, variable name, and index selection.
