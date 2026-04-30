Design Overview
===============

This page gives a high-level description of how p5rem works internally.
Developers wishing to contribute or embed p5rem in a larger system will find
this a useful starting point.

Core concept
------------

p5rem is a **thin proxy over a remote pyfive instance**.  Rather than
emulating file handles or smuggling metadata across the wire, the server runs a
real :mod:`pyfive` instance directly on the HPC host — where HDF5 metadata
reads are local and fast — and the client proxies the results.

::

    Client (desktop)                     Server (HPC)

    rFile                                pyfive.File  (fast local I/O)
      │                                      │
      ├── file_open()  ─────────────────►  f = pyfive.File(path)
      │   ◄── keys, attrs, mtime ────────  return serialise(f.keys(), f.attrs)
      │                                      │
      ├── var_open()   ─────────────────►  ds = f[varname]
      │   ◄── shape, dtype, chunk index ─  return serialise(ds.shape, …)
      │                                      │
      └── get_chunk()  ─────────────────►  ds.id._get_raw_chunk(storeinfo)
          ◄── raw bytes ─────────────────  return chunk_bytes

The server does all HDF5 heavy lifting — superblock, object headers, B-tree
traversal — entirely at local disk speed.

Transport
---------

Communication happens over the stdin/stdout of the SSH process.  Messages are
CBOR-encoded and prefixed with a 4-byte big-endian length field.  No TCP port
forwarding is required; the design follows the same pattern as VSCode Remote.
