Cache — ``p5rem.cache``
=======================

p5rem can optionally cache remote metadata and chunk payloads in a local disk
store so that repeated access to the same data avoids redundant SSH round
trips.

.. autoclass:: p5rem.cache.P5RemCache
   :members:
