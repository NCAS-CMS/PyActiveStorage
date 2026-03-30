# p5rem Protocol Documentation

## Protocol Overview

p5rem communicates over a binary CBOR-framed protocol via SSH stdio. Messages are prefixed with a 4-byte little-endian length field, followed by CBOR-encoded payload. Here we demonstrate the usage of the protocol in a simple read remote slice example.

### Message Types

This example uses the following message types

- **FILE_OPEN** / **FILE_INFO** — Open a file, retrieve top-level keys and attributes
- **VAR_OPEN** / **VAR_INFO** — Open a variable, retrieve shape, dtype, chunks, and chunk index
- **GET_CHUNK** / **CHUNK_DATA** — Request raw chunk bytes by offset/size
- **LIST** / **LIST_RESULT** — List directory contents
- **STAT** / **STAT_RESULT** — Get file/directory metadata (size, mtime, mode, etc.)
- **FILE_CLOSE** — Release file handle
- **HEARTBEAT** — Keepalive message
- **ERROR** — Error response (used for all failures)

There is also a slot in the protocol for supporting remote reductions
via PyActiveStorage.

## Example: read_remote_slice.py

The following sequence diagram illustrates a complete read operation, including
starting and stopping a session:

```plantuml
@startuml read_remote_slice
participant "User Code" as User
participant "p5remSession" as Session
participant "Client\n(Paramiko SSH)" as Client
participant "Remote Server\n(remote_server.py)" as Server

User -> Session: bootstrap_session()
Session -> Client: SSH connect\n& upload remote_server.py
Client -> Server: Execute remote_server.py

== session.open(REMOTE_FILE) ==

User -> Session: open(path)
Session -> Session: create rFile proxy
Session -> Client: FILE_OPEN request
Client -> Server: {type: "FILE_OPEN",\npath: "p5test/test1.nc"}
Server -> Server: pyfive.File(path)
Server -> Client: FILE_INFO response
Client -> Session: {type: "FILE_INFO",\nkeys: [...], attrs: {...}, mtime: ...}
Session -> User: rFile proxy

== remote_file[VARIABLE] ==

User -> Session: __getitem__("tas")
Session -> Session: create rDataset proxy
Session -> User: rDataset proxy

== remote_file[VARIABLE][SELECTION] ==

User -> Session: rDataset.__getitem__\n((0, slice(None), slice(None)))
Session -> Session: ensure_meta()\n — lazy metadata load
Session -> Client: VAR_OPEN request
Client -> Server: {type: "VAR_OPEN",\npath: "p5test/test1.nc",\nvarname: "tas"}
Server -> Server: dataset = file["tas"]
Server -> Server: extract\nshape/dtype/chunks/index
Server -> Client: VAR_INFO response
Client -> Session: {type: "VAR_INFO",\nshape: [...], dtype: "float32",\nindex: [...], ...}
Session -> Session: id.get_data(selection, fillvalue)
Session -> Client: GET_CHUNK request(s)
Client -> Server: {type: "GET_CHUNK",\npath: "...", varname: "tas",\nbyte_offset: ..., size: ...}
Server -> Server: _get_raw_chunk(storeinfo)
Server -> Client: CHUNK_DATA response
Client -> Session: {type: "CHUNK_DATA",\ndata: b"...", byte_offset: ..., size: ...}
Session -> Session: decompress & deserialize
Session -> User: numpy.ndarray

== Cleanup ==

User -> Session: with context exit
Session -> Client: FILE_CLOSE request
Client -> Server: {type: "FILE_CLOSE",\npath: "p5test/test1.nc"}
Server -> Server: file.close()
Server -> Client: FILE_CLOSE response
Session -> Client: Close SSH connection

@enduml
```

### Key Design Points

1. **Lazy Evaluation** — Proxies (rFile, rDataset) avoid network traffic until data is actually accessed
2. **Metadata Caching** — Variable metadata is cached by mtime to avoid re-fetching
3. **Streaming Chunks** — Large data transfers happen via GET_CHUNK/CHUNK_DATA pairs, enabling partial reads
4. **pyfive Index** — Server extracts the HDF5 chunk index and sends it to the client, enabling intelligent chunk location
5. **Binary Efficiency** — CBOR serialization + length-prefix framing minimizes overhead

Client-side chunk cache support is available, but not used in this example.
