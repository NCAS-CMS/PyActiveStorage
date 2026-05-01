p5rem Wire Protocol
===================

The p5rem protocol is a session-oriented request/response protocol used between
client and server over the stdin/stdout streams of an SSH-launched process.

This page documents the **data-path operations** used by
``activestorage.remote.rFile`` and ``activestorage.remote.rDataset``.

For bootstrap, preflight checks, SSH setup, and remote environment discovery,
see :doc:`wire_protocols_p5rem_session_setup`.

Transport
---------

- **Carrier**: stdin/stdout byte streams
- **Framing**: 4-byte big-endian length prefix + payload
- **Encoding**: CBOR (``cbor2``)
- **Connection model**: persistent session (multiple requests per process)

Core Request Types
------------------

The protocol is command-based. Current request/response message types are
defined in ``activestorage.p5rem.protocol`` and implemented by
``activestorage.p5rem.remote_server.ServerStub``.

The operations below are those used to interrogate or serve ``rFile`` /
``rDataset`` objects.

.. list-table::
   :header-rows: 1
   :widths: 22 22 56

   * - Request Type
     - Response Type
     - Purpose
   * - ``FILE_OPEN``
     - ``FILE_INFO``
     - Open remote file and return top-level keys, attrs, and mtime.
   * - ``VAR_OPEN``
     - ``VAR_INFO``
     - Open dataset and return shape/dtype/chunk/index/filter metadata.
   * - ``GET_CHUNK``
     - ``CHUNK_DATA``
     - Fetch one raw chunk by byte offset and size.
   * - ``GET_CHUNKS``
     - ``CHUNK_DATA`` (streamed) + ``CHUNKS_DONE``
     - Fetch multiple chunks in one round-trip.
   * - ``REDUCE``
     - ``REDUCTION_RESULT``
     - Reduce one chunk (``mode=chunk``) or a selection
       (``mode=selection``).
   * - ``FILE_CLOSE``
     - ``FILE_CLOSE``
     - Release remote file handle and cached dataset state.

If the server cannot satisfy a request it returns ``ERROR``.

Operation Details
-----------------

``FILE_OPEN``
~~~~~~~~~~~~~

Request fields:

- ``path`` (str)

Response highlights (``FILE_INFO``):

- ``path``
- ``keys``
- ``attrs``
- ``dim_id_to_name``
- ``mtime``

``VAR_OPEN``
~~~~~~~~~~~~

Request fields:

- ``path`` (str)
- ``varname`` (str)

Response highlights (``VAR_INFO``):

- ``shape``
- ``dtype``
- ``chunks``
- ``index``
- ``attrs``
- ``fillvalue``
- ``filter_pipeline``
- ``order``
- ``layout``
- ``fragmented``

``GET_CHUNK``
~~~~~~~~~~~~~

Request fields:

- ``path``
- ``varname``
- ``byte_offset``
- ``size``
- optional chunk-identifying fields (for example ``chunk_coord``)

Response highlights (``CHUNK_DATA``):

- ``byte_offset``
- ``size``
- ``filter_mask``
- ``data`` (bytes)

``GET_CHUNKS``
~~~~~~~~~~~~~~

Request fields:

- ``path``
- ``varname``
- ``chunks`` (list of chunk descriptors)
- ``thread_count``

Server response is a stream:

1. zero or more ``CHUNK_DATA`` messages
2. one ``CHUNKS_DONE`` terminator

``REDUCE``
~~~~~~~~~~

``REDUCE`` supports two modes:

- ``mode="chunk"``: reduction over one raw chunk (requires ``byte_offset`` and
  ``size``)
- ``mode="selection"``: reduction over a logical selection (or full dataset)

Common request fields:

- ``path``
- ``varname``
- ``operation``
- ``mode``

Extra request fields by mode:

- chunk mode: ``byte_offset``, ``size``
- selection mode: ``selection`` (optional), ``thread_count``

Response (``REDUCTION_RESULT``) includes:

- ``operation``
- ``mode``
- ``value``

``FILE_CLOSE``
~~~~~~~~~~~~~~

Request fields:

- ``path``

Response fields:

- ``type`` = ``FILE_CLOSE``
- ``path``
- ``closed`` (bool)

Response Model
--------------

Each request returns a structured CBOR response object.

- Success responses contain command-specific fields (e.g. ``data`` for chunk
  payloads, metadata dictionaries for open calls)
- Failure responses return structured error data interpreted by
  ``SessionError`` / ``ResponseError`` wrappers

``GET_CHUNKS`` is the only operation that returns multiple messages before a
terminator (``CHUNKS_DONE``).

Caching and Consistency
-----------------------

Client-side session cache can be enabled, and file modification timestamps
(``mtime``) are tracked to invalidate stale metadata/chunk state.

Related API Surface
-------------------

- ``activestorage.session.p5remSession`` for transport requests
- ``activestorage.remote.rFile`` and ``activestorage.remote.rDataset`` for
  pyfive-like proxies on top of the protocol
