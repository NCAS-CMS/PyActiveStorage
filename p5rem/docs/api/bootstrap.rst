Bootstrap — ``p5rem.bootstrap``
================================

These helpers handle server upload, SSH launch, and session lifecycle
management.  Most users need only :func:`bootstrap_session`.

.. autofunction:: p5rem.bootstrap.bootstrap_session

.. autofunction:: p5rem.bootstrap.bootstrap_server

.. autofunction:: p5rem.bootstrap.bootstrap_reconnecting_session

.. autofunction:: p5rem.bootstrap.discover_remote_conda_envs

.. autoclass:: p5rem.bootstrap.BootstrappedProcess
   :members:

.. autoclass:: p5rem.bootstrap.ReconnectingBootstrappedSession
   :members:

.. autoexception:: p5rem.bootstrap.BootstrapError
