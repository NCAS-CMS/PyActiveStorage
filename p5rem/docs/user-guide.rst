User Guide
==========

Sessions
--------

A *session* manages the SSH connection and the remote server process.  The
recommended way to create one is :func:`~p5rem.bootstrap.bootstrap_session`,
which handles server upload, launch, and teardown automatically:

.. code-block:: python

   from p5rem import bootstrap_session

   with bootstrap_session(
       host="hpc",
       remote_python="conda run -n myenv python",
       login_shell=True,
       use_cache=False,
   ) as session:
       ...   # use session inside the block

The session is closed (and the remote server process exited) when the ``with``
block exits.

.. rubric:: Key parameters

``host``
   SSH host alias or hostname.  Resolved via ``~/.ssh/config``.

``remote_python``
   Command string used to launch Python on the remote host.  Typical values:

   * ``"python"`` — plain Python on ``$PATH``
   * ``"conda run -n myenv python"`` — activate a conda environment
   * ``"/home/user/venv/bin/python"`` — absolute path to a virtualenv

   That remote Python environment must include the server-stub runtime
   dependencies: ``pyfive``, ``numpy``, and ``cbor2``.

``login_shell``
   Set ``True`` if the remote Python command is only available after shell
   startup scripts have run (common with conda or module-load environments).

``use_cache``
   Set ``False`` to disable the local metadata and chunk cache and always go
   directly to the remote host.  Useful for benchmarking or when data changes
   frequently.  See :ref:`caching` for details.

Opening remote files
--------------------

Once you have a session, open remote files with
:meth:`~p5rem.session.p5remSession.open`:

.. code-block:: python

   with session.open("/path/to/data.nc") as f:
       print(f.keys())          # top-level variable / group names
       print(f.attrs)           # global attributes dict

The returned object is an :class:`~p5rem.proxy.rFile`, which mirrors the
interface of :class:`pyfive.File`.

Accessing variables
-------------------

Index an :class:`~p5rem.proxy.rFile` by variable name to get an
:class:`~p5rem.proxy.rDataset`:

.. code-block:: python

   tas = f["tas"]
   print(tas.shape)   # e.g. (12, 64, 128)
   print(tas.dtype)   # e.g. float64

Fetching data
~~~~~~~~~~~~~

Index an :class:`~p5rem.proxy.rDataset` with any NumPy-compatible selection
to materialise a local :class:`numpy.ndarray`.  Only the HDF5 chunks that
overlap the requested selection are fetched:

.. code-block:: python

   first_timestep = tas[0, :, :]          # 2-D slice
   subregion      = tas[0:6, 20:50, :]    # 3-D subregion
   full_array     = tas[()]               # entire variable

.. caution::

   Fetching the entire variable at once (``tas[()]``) transfers all chunks
   over SSH.  For large variables, prefer narrow index selections.

Attributes and coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~~

Variable and file attributes are available as plain dicts:

.. code-block:: python

   print(tas.attrs["units"])        # e.g. 'K'
   print(f.attrs["Conventions"])    # e.g. 'CF-1.8'

Coordinate variables (latitude, longitude, time …) are accessed the same way
as data variables:

.. code-block:: python

   lat = f["lat"][()]
   lon = f["lon"][()]

.. _caching:

Local cache
-----------

By default p5rem uses a persistent disk cache (backed by
`diskcache <https://grantjenks.com/docs/diskcache/>`_) to avoid re-fetching
metadata and chunk data across sessions.  The cache lives in
``~/.cache/p5rem/`` and has a 10 GB size limit by default.

The high-level :func:`~p5rem.bootstrap.bootstrap_session` helper uses p5rem's
default persistent cache automatically.  If you want to avoid any local cache
usage for a session, disable it explicitly:

.. code-block:: python

   # Disable caching entirely
   with bootstrap_session(host="hpc", remote_python="…", use_cache=False) as session:
       ...

Server-side reductions
----------------------

For large arrays it is often more efficient to compute a reduction on the
remote host — fetching only the scalar result — rather than transferring all
chunk data and reducing locally.

p5rem exposes two reduction APIs on the session object.

.. rubric:: Whole-selection reduction

:meth:`~p5rem.session.p5remSession.reduce_selection` reads all chunks that
overlap a given index selection, computes the reduction in parallel on the
server (using a thread pool), and returns a scalar:

.. code-block:: python

   result = session.reduce_selection(
       "/path/to/data.nc",
       "tas",
       "mean",
       selection=[
           {"type": "slice", "start": 0, "stop": 6, "step": 1},   # time axis
           None,                                                    # full latitude
           None,                                                    # full longitude
       ],
       thread_count=8,
   )
   print(result["value"])   # scalar mean over the selection

Supported operations: ``"sum"``, ``"mean"``, ``"min"``, ``"max"``,
``"range"``, ``"count"``, ``"argmin"``, ``"argmax"``.

A ``selection`` of ``None`` covers the entire variable.

.. rubric:: Single-chunk reduction

:meth:`~p5rem.session.p5remSession.reduce_chunk` applies a reduction to one
raw HDF5 chunk, identified by its byte offset and size.  This is a lower-level
API used by clients that already have the chunk index from a previous
:meth:`~p5rem.session.p5remSession.var_open` call:

.. code-block:: python

   meta   = session.var_open("/path/to/data.nc", "tas")
   chunk0 = meta["index"][0]

   result = session.reduce_chunk(
       "/path/to/data.nc", "tas",
       chunk0["byte_offset"], chunk0["size"],
       "mean",
   )
   print(result["value"])

Reconnecting sessions
---------------------

For long-running processes (e.g. a GUI that may run for hours),
:class:`~p5rem.bootstrap.ReconnectingBootstrappedSession` wraps a session and
automatically re-bootstraps if the SSH connection drops:

.. code-block:: python

   from p5rem import bootstrap_reconnecting_session

   session = bootstrap_reconnecting_session(
       host="hpc",
       remote_python="conda run -n myenv python",
       login_shell=True,
   )
   # session transparently reconnects on the next operation after a failure

Conda environment discovery
---------------------------

To list the conda environments available on the remote host:

.. code-block:: python

   from p5rem import discover_remote_conda_envs

   envs = discover_remote_conda_envs(host="hpc")
   for name, python in envs.items():
       print(name, python)
   