Session — ``p5rem.session``
===========================

The session layer manages the wire protocol between client and server.
Normally you do not construct these directly — use
:func:`~p5rem.bootstrap.bootstrap_session` instead.

.. autoclass:: p5rem.session.p5remSession
   :members:

.. autoclass:: p5rem.session.Session
   :members:

.. autoexception:: p5rem.session.SessionError

.. autoexception:: p5rem.session.ResponseError

.. autoexception:: p5rem.session.UnexpectedResponseError
