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
          remote_setup="module load jaspy",
          remote_python="python",
       login_shell=True,
       use_cache=False,
   ) as session:
       ...   # use session inside the block

    # Conda activation is equally valid when you need it:
    # remote_setup="source /path/to/conda.sh && conda activate myenv"

The session is closed (and the remote server process exited) when the ``with``
block exits.

.. rubric:: Key parameters

``host``
   SSH host alias or hostname.  Resolved via ``~/.ssh/config``.

``remote_setup``
   Optional shell fragment run before launching remote Python.  Use this for
   module loads or explicit environment activation.  Typical values:

   * ``"module load python/3.11"``
   * ``"source /path/to/conda.sh && conda activate myenv"``
