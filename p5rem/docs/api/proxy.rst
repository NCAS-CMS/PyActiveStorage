Proxy objects — ``p5rem.proxy``
===============================

These are the primary objects a user interacts with.  They mirror the
:class:`pyfive.File` / :class:`pyfive.Dataset` interface so that existing
code can be adapted with minimal changes.

.. autoclass:: p5rem.proxy.rFile
   :members:
   :special-members: __getitem__, __enter__, __exit__

.. autoclass:: p5rem.proxy.rDataset
   :members:
   :special-members: __getitem__
