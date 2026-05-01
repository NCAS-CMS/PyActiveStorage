p5rem Session Setup and Environment Discovery
=============================================

This page documents the **session/bootstrap side** of p5rem:

- SSH connection setup
- remote Python preflight checks
- remote server upload/launch
- session lifecycle and reconnect behavior
- remote conda environment discovery

For data-path protocol operations used by ``rFile`` / ``rDataset``, see
:doc:`wire_protocols_p5rem`.

Bootstrap Flow
--------------

Main entry points:

- ``activestorage.bootstrap.bootstrap_server``
- ``activestorage.bootstrap.bootstrap_session``
- ``activestorage.bootstrap.bootstrap_reconnecting_session``

High-level flow:

1. Resolve SSH host/user/port/key from explicit args and ``~/.ssh/config``
2. Connect over SSH (optionally prompting for password in interactive mode)
3. Run remote preflight checks against ``remote_python``
4. Upload remote server stub (``p5rem/remote_server.py``)
5. Launch server process and bind stdin/stdout streams
6. Construct ``p5remSession`` and verify startup via ``HEARTBEAT``

Preflight Checks
----------------

Preflight is handled by ``_probe_remote_python`` and validates:

- ``remote_setup`` executes successfully (if provided)
- ``remote_python`` can run a python command
- required dependencies are present in that runtime:

  - ``cbor2``
  - backend package: ``pyfive`` (HDF5/NetCDF) or ``ppfive`` (PP)

On failure, ``BootstrapError`` is raised with stage-specific diagnostics.

Session/Control Wire Operations
-------------------------------

These protocol messages are session/control oriented rather than rFile/rDataset
content operations.

.. list-table::
   :header-rows: 1
   :widths: 22 22 56

   * - Request Type
     - Response Type
     - Purpose
   * - ``LIST``
     - ``LIST_RESULT``
     - List remote filesystem entries for a path.
   * - ``STAT``
     - ``STAT_RESULT``
     - Return stat metadata for a remote path.
   * - ``HEARTBEAT``
     - ``HEARTBEAT``
     - Keepalive/health probe for an active session.

Remote Environment Discovery
----------------------------

``activestorage.bootstrap.discover_remote_conda_envs`` provides explicit remote
environment discovery independent of the p5rem data protocol.

Behavior:

- Connects via SSH using the same SSH resolution logic
- Executes ``conda env list`` remotely (optionally in login shell)
- Parses and returns ``{env_name: env_path}`` mapping

Example return value::

    {
        "base": "/path/to/miniforge3",
        "work26": "/path/to/miniforge3/envs/work26",
        "jas26": "/path/to/miniforge3/envs/jas26",
    }

Reconnect Semantics
-------------------

``ReconnectingBootstrappedSession`` wraps a bootstrapped session and retries by
re-running bootstrap when transport/session errors are detected.

- On first failure, reconnect is attempted
- Calls are retried once after reconnect
- Background heartbeat can trigger reconnect via callback
