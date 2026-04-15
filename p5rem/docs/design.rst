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

The server process exits automatically when the SSH connection closes, so
there are no orphaned server processes to clean up.

Architecture diagram
~~~~~~~~~~~~~~~~~~~~

::

    Desktop (xconv2 / script)
      │
      └── paramiko SSH
              │  stdin/stdout (CBOR frames)
              │
              └── p5rem server stub (HPC)
                      ├── pyfive          (local HDF5 reading)
                      └── pyactivestorage (server-side reductions)

Chunk-worklist parallel reductions
-----------------------------------

When :meth:`~p5rem.session.p5remSession.reduce_selection` is called, the
server:

1. Uses pyfive's :class:`~pyfive.indexing.OrthogonalIndexer` to enumerate
   every HDF5 chunk that overlaps the requested selection.
2. Submits those chunks to a :class:`~concurrent.futures.ThreadPoolExecutor`
   for parallel I/O via :func:`os.pread`.
3. Decodes each chunk (including decompression) and computes a *partial*
   reduction (e.g. partial sum + count for ``"mean"``).
4. Merges the partial results into a final scalar.

This mirrors the approach used by
`pyactivestorage <https://github.com/NCAS-CMS/PyActiveStorage>`_ and avoids
ever materialising the full array in memory.

For *contiguous* (non-chunked) variables, the server streams the data in
4 MB blocks, again without full materialisation.

Key design decisions
--------------------

No persistent server
~~~~~~~~~~~~~~~~~~~~

The server is a single self-contained Python file (``remote_server.py``) that
is uploaded fresh each session.  This means:

* Upgrades take effect immediately on the next session.
* No HPC admin involvement — no daemons, no init scripts, no port
  registrations.
* The server vanishes when the SSH connection closes.

Self-contained server stub
~~~~~~~~~~~~~~~~~~~~~~~~~~

``remote_server.py`` deliberately does not import any other p5rem module.
All protocol constants and framing code are inlined so the file can be
dropped onto any Python ≥ 3.10 installation that has ``pyfive`` and
``numpy``.

Single persistent SSH connection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A single SSH channel is established at startup and reused for all operations
in the session.  This avoids the latency of repeated SSH handshakes (which
can be 200–500 ms on distant HPC nodes).

Reconnecting wrapper
~~~~~~~~~~~~~~~~~~~~

:class:`~p5rem.bootstrap.ReconnectingBootstrappedSession` wraps the session
and automatically re-bootstraps when the SSH connection is lost, making p5rem
suitable for long-running GUI processes that should survive network
interruptions.

Protocol versioning
-------------------

The current protocol is intentionally simple and unversioned.  Server and
client are always from the same release (the server stub is uploaded at
session start), so version skew cannot occur.  A version handshake may be
added in a future release if the protocol is stabilised for third-party
clients.
