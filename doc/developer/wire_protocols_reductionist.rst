Reductionist Wire Protocol
==========================

`Reductionist <https://github.com/stackhpc/reductionist-rs>`_ is an HTTP-based
active storage server. PyActiveStorage sends ``POST`` requests that describe a
chunk reduction operation and receives CBOR-encoded results.

Base URL
--------

All requests are sent to a versioned endpoint on the Reductionist server::

    POST {server}/v2/{operation}/

where ``{server}`` is ``active_storage_url`` and ``{operation}`` is one of the
supported operations listed below.

Transport
---------

- **Protocol**: HTTPS (HTTP also accepted for development/testing)
- **Authentication**: Optional HTTP Basic Auth on ``requests.Session``
- **TLS verification**: Configurable; defaults to off (``session.verify=False``)

Supported Operations
--------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Operation
     - Description
   * - ``min``
     - Minimum value over selected chunk region (and optionally axes)
   * - ``max``
     - Maximum value over selected chunk region
   * - ``sum``
     - Sum over selected chunk region; also used internally for ``mean``
   * - ``select``
     - Raw data extraction with no reduction (used when operation is ``None``)

.. note::

   ``mean`` is not a native Reductionist operation. The client sends ``sum``
   and computes the final mean itself.

Request Format
--------------

Request body is JSON (``Content-Type: application/json``).

.. list-table::
   :header-rows: 1
   :widths: 25 10 65

   * - Field
     - Required
     - Description
   * - ``interface_type``
     - yes
     - Storage interface, e.g. ``"s3"`` or ``"https"``
   * - ``url``
     - yes
     - Full storage object URL
   * - ``dtype``
     - yes
     - NumPy dtype name, e.g. ``"float32"``
   * - ``byte_order``
     - yes
     - Endianness: ``"little"`` or ``"big"``
   * - ``offset``
     - yes
     - Byte offset of the chunk
   * - ``size``
     - yes
     - Byte length of the chunk
   * - ``order``
     - yes
     - Memory order: ``"C"`` or ``"F"``
   * - ``shape``
     - no
     - Full chunk shape list
   * - ``selection``
     - no
     - Per-dimension ``[start, stop, step]`` triples
   * - ``compression``
     - no
     - Compression codec descriptor
   * - ``filters``
     - no
     - Filter descriptors (e.g. shuffle)
   * - ``missing``
     - no
     - Missing-data descriptor
   * - ``axis``
     - no
     - Axis tuple/list to reduce over
   * - ``option_disable_chunk_cache``
     - no
     - Disable server-side chunk cache when true

Response Format
---------------

Successful responses are CBOR objects with:

- ``bytes``: raw result bytes
- ``dtype``: dtype string
- ``shape``: optional result shape
- ``count``: contributing non-missing element counts

Client decodes these and masks where ``count == 0``.

Error Model
-----------

Non-2xx responses are raised as
``activestorage.reductionist.ReductionistError``.

- ``HTTP 500``: body is parsed for JSON error details when possible
- Other status codes: HTTP reason phrase is used

Authentication
--------------

Credentials are passed via HTTP Basic Auth on ``requests.Session`` (not in the
JSON payload). The server uses them to access backend object storage.
