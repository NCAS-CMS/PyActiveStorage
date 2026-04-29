Wire Protocols
==============

PyActiveStorage currently supports one wire protocol for communicating with a
remote active storage server: the **Reductionist** protocol.  A second protocol
is planned; this page will be updated when it is defined.

.. contents:: Contents
   :local:
   :depth: 2

----

Reductionist Protocol
---------------------

`Reductionist <https://github.com/stackhpc/reductionist-rs>`_ is an HTTP-based
active storage server.  PyActiveStorage communicates with it by sending
``POST`` requests that describe a chunk reduction operation and receiving CBOR-
encoded results.

Base URL
^^^^^^^^

All requests are sent to a versioned endpoint on the Reductionist server::

    POST {server}/v2/{operation}/

where ``{server}`` is the ``active_storage_url`` supplied to ``Active``, and
``{operation}`` is one of the supported operations listed below.

Transport
^^^^^^^^^

- **Protocol**: HTTPS (HTTP also accepted for development/testing)
- **Authentication**: Optional HTTP Basic Auth (username / password passed via
  a ``requests.Session``)
- **TLS verification**: Configurable; defaults to *off* (``session.verify =
  False``)

Supported Operations
^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Operation
     - Description
   * - ``min``
     - Minimum value over the selected chunk region (and optionally over axes)
   * - ``max``
     - Maximum value over the selected chunk region
   * - ``sum``
     - Sum of values over the selected chunk region.  Also used internally for
       ``mean`` (the client maps ``mean`` → ``sum``).
   * - ``select``
     - Raw data extraction with no reduction (used when ``operation`` is
       ``None``)

.. note::

   ``mean`` is **not** a native Reductionist operation.  The client sends a
   ``sum`` request and performs the final mean calculation itself.

Request Format
^^^^^^^^^^^^^^

The request body is a JSON object (``Content-Type: application/json``).

.. list-table::
   :header-rows: 1
   :widths: 25 10 65

   * - Field
     - Required
     - Description
   * - ``interface_type``
     - yes
     - Storage interface: ``"s3"`` or ``"https"``
   * - ``url``
     - yes
     - Full URL of the storage object, e.g.
       ``https://endpoint/bucket/key.nc``
   * - ``dtype``
     - yes
     - NumPy dtype name, e.g. ``"float32"``, ``"int16"``
   * - ``byte_order``
     - yes
     - Endianness: ``"little"`` or ``"big"``
   * - ``offset``
     - yes
     - Byte offset of the chunk within the object (integer)
   * - ``size``
     - yes
     - Byte length of the chunk (integer)
   * - ``order``
     - yes
     - Array memory order: ``"C"`` (row-major) or ``"F"`` (column-major)
   * - ``shape``
     - no
     - Shape of the full chunk as a list of integers, e.g. ``[3, 3, 1]``
   * - ``selection``
     - no
     - Sub-selection within the chunk.  A list of ``[start, stop, step]``
       triples, one per dimension.  Integer indices are encoded as
       ``[i, i+1, 1]``.
   * - ``compression``
     - no
     - Compression codec object, e.g. ``{"id": "zlib"}`` or
       ``{"id": "blosc"}``
   * - ``filters``
     - no
     - List of filter codec objects applied *before* compression.  Currently
       only ``shuffle`` is supported:
       ``{"id": "shuffle", "element_size": <N>}``
   * - ``missing``
     - no
     - Missing-data descriptor (see `Missing Data Encoding`_ below)
   * - ``axis``
     - no
     - Tuple of axis indices to reduce over, e.g. ``[0, 1]``.  Omit to
       reduce over all axes.
   * - ``option_disable_chunk_cache``
     - no
     - Boolean.  When ``true``, instructs the server not to cache the raw
       chunk bytes.

Example request (minimal)::

    {
        "interface_type": "s3",
        "url": "https://s3.example.com/mybucket/data.nc",
        "dtype": "float32",
        "byte_order": "little",
        "offset": 4096,
        "size": 2048,
        "order": "C"
    }

Example request (full)::

    {
        "interface_type": "s3",
        "url": "https://s3.example.com/mybucket/data.nc",
        "dtype": "float32",
        "byte_order": "little",
        "offset": 4096,
        "size": 2048,
        "order": "C",
        "shape": [10, 10, 1],
        "selection": [[0, 3, 1], [4, 6, 1], [0, 1, 1]],
        "compression": {"id": "zlib"},
        "filters": [{"id": "shuffle", "element_size": 4}],
        "missing": {"missing_value": -9999.0},
        "axis": [0, 1],
        "option_disable_chunk_cache": true
    }

Missing Data Encoding
^^^^^^^^^^^^^^^^^^^^^

The ``missing`` field is a JSON object with **one** of the following keys:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Value
   * - ``missing_value``
     - A single scalar value to treat as missing
   * - ``missing_values``
     - A list of scalar values, all of which are treated as missing
   * - ``valid_min``
     - Values strictly below this are treated as missing
   * - ``valid_max``
     - Values strictly above this are treated as missing
   * - ``valid_range``
     - A two-element list ``[min, max]``; values outside this range are missing

.. note::

   ``numpy.float32`` values are up-cast to ``float64`` before JSON encoding
   because the standard JSON encoder does not support 32-bit floats natively.

Response Format
^^^^^^^^^^^^^^^

Successful responses (HTTP 200) are **CBOR-encoded** (not JSON).  The decoded
object is a dict with the following fields:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Field
     - Description
   * - ``bytes``
     - Raw bytes of the result array, interpreted with ``dtype``
   * - ``dtype``
     - NumPy dtype string of the result, e.g. ``"float32"``
   * - ``shape``
     - Shape of the result array as a list of integers (absent for scalars)
   * - ``count``
     - Array of the same shape as the result containing the number of
       non-missing values that contributed to each element.  Used by the
       client to mask elements where ``count == 0``.

The client decodes the response as follows:

.. code-block:: python

    import cbor2, numpy as np

    reduction_result = cbor2.loads(response.content)
    dtype  = reduction_result['dtype']
    shape  = reduction_result.get('shape')
    result = np.frombuffer(reduction_result['bytes'], dtype=dtype).reshape(shape)
    count  = reduction_result['count']
    result = np.ma.masked_where(count == 0, result)

Error Responses
^^^^^^^^^^^^^^^

On failure the server returns a non-2xx HTTP status code.

* **HTTP 500** — The response body is JSON.  The client attempts to decode and
  include it in the exception message.
* **All other errors** — The standard HTTP reason phrase is used as the message.

All errors are raised as ``activestorage.reductionist.ReductionistError``::

    ReductionistError: Reductionist error: HTTP 400: {"error": {...}}

Authentication
^^^^^^^^^^^^^^

Credentials (S3 access key / secret key) are passed as HTTP Basic Auth on the
``requests.Session``, not in the JSON request body.  The server uses them to
authenticate against the S3 backend on the client's behalf.
