"""CBOR-framed wire protocol helpers for p5rem."""

from __future__ import annotations

from collections.abc import Mapping
from io import BufferedIOBase
import struct
from typing import Any

import cbor2

LENGTH_PREFIX_SIZE = 4
MAX_MESSAGE_SIZE = (1 << (LENGTH_PREFIX_SIZE * 8)) - 1

# Request message types.
LIST = "LIST"
STAT = "STAT"
FILE_OPEN = "FILE_OPEN"
VAR_OPEN = "VAR_OPEN"
GET_CHUNK = "GET_CHUNK"
GET_CHUNKS = "GET_CHUNKS"
REDUCE = "REDUCE"
FILE_CLOSE = "FILE_CLOSE"
HEARTBEAT = "HEARTBEAT"

# Response message types.
LIST_RESULT = "LIST_RESULT"
STAT_RESULT = "STAT_RESULT"
FILE_INFO = "FILE_INFO"
VAR_INFO = "VAR_INFO"
CHUNK_DATA = "CHUNK_DATA"
CHUNKS_DONE = "CHUNKS_DONE"
REDUCTION_RESULT = "REDUCTION_RESULT"
ERROR = "ERROR"

REQUEST_TYPES = frozenset(
	{
		LIST,
		STAT,
		FILE_OPEN,
		VAR_OPEN,
		GET_CHUNK,
		GET_CHUNKS,
		REDUCE,
		FILE_CLOSE,
		HEARTBEAT,
	}
)
RESPONSE_TYPES = frozenset(
	{
		LIST_RESULT,
		STAT_RESULT,
		FILE_INFO,
		VAR_INFO,
		CHUNK_DATA,
		CHUNKS_DONE,
		REDUCTION_RESULT,
		ERROR,
	}
)
MESSAGE_TYPES = REQUEST_TYPES | RESPONSE_TYPES


class ProtocolError(ValueError):
	"""Base exception for protocol validation and framing errors."""


class MessageTypeError(ProtocolError):
	"""Raised when a message type is missing or invalid."""


class MessageFormatError(ProtocolError):
	"""Raised when a message payload is not a protocol message map."""


class MessageFramingError(ProtocolError):
	"""Raised when framed bytes do not match the expected wire format."""


def make_message(message_type: str, /, **fields: Any) -> dict[str, Any]:
	"""Build and validate a protocol message map."""

	if "type" in fields:
		raise MessageFormatError("message fields must not include 'type'")

	validate_message_type(message_type)
	return {"type": message_type, **fields}


def validate_message_type(message_type: str) -> str:
	"""Validate a protocol message type string."""

	if not isinstance(message_type, str) or not message_type:
		raise MessageTypeError("message type must be a non-empty string")
	if message_type not in MESSAGE_TYPES:
		raise MessageTypeError(f"unknown message type: {message_type!r}")
	return message_type


def validate_message(message: Mapping[str, Any]) -> dict[str, Any]:
	"""Validate and normalise a protocol message map."""

	if not isinstance(message, Mapping):
		raise MessageFormatError("message must be a mapping")

	if "type" not in message:
		raise MessageFormatError("message is missing required 'type' field")

	message_type = validate_message_type(message["type"])
	return {"type": message_type, **{key: value for key, value in message.items() if key != "type"}}


def encode(message_type: str, /, **fields: Any) -> bytes:
	"""Encode a protocol message as length-prefixed CBOR bytes."""

	return encode_message(make_message(message_type, **fields))


def encode_message(message: Mapping[str, Any]) -> bytes:
	"""Encode a validated message map as a framed wire payload."""

	normalised = validate_message(message)
	payload = cbor2.dumps(normalised)

	if len(payload) > MAX_MESSAGE_SIZE:
		raise MessageFramingError(
			f"message payload exceeds maximum size of {MAX_MESSAGE_SIZE} bytes"
		)

	return struct.pack(">I", len(payload)) + payload


def decode(data: bytes) -> dict[str, Any]:
	"""Decode a length-prefixed CBOR message from bytes."""

	if not isinstance(data, bytes):
		raise MessageFramingError("framed message must be bytes")
	if len(data) < LENGTH_PREFIX_SIZE:
		raise MessageFramingError("message is shorter than the 4-byte length prefix")

	payload_length = struct.unpack(">I", data[:LENGTH_PREFIX_SIZE])[0]
	payload = data[LENGTH_PREFIX_SIZE:]

	if payload_length != len(payload):
		raise MessageFramingError(
			"message length prefix does not match payload size"
		)

	return decode_payload(payload)


def decode_payload(payload: bytes) -> dict[str, Any]:
	"""Decode an unframed CBOR payload into a validated message map."""

	if not isinstance(payload, bytes):
		raise MessageFormatError("payload must be bytes")

	try:
		message = cbor2.loads(payload)
	except (EOFError, ValueError, cbor2.CBORDecodeError) as exc:
		raise MessageFormatError("payload is not valid CBOR") from exc

	return validate_message(message)


def read_message(stream: BufferedIOBase) -> dict[str, Any]:
	"""Read a single framed message from a binary stream."""

	prefix = _read_exact(stream, LENGTH_PREFIX_SIZE)
	payload_length = struct.unpack(">I", prefix)[0]
	payload = _read_exact(stream, payload_length)
	return decode_payload(payload)


def write_message(
	stream: BufferedIOBase,
	message: Mapping[str, Any] | str,
	/,
	**fields: Any,
) -> int:
	"""Write a single framed message to a binary stream."""

	framed = encode_message(message if isinstance(message, Mapping) else make_message(message, **fields))
	written = stream.write(framed)

	if written is None:
		written = len(framed)
	if written != len(framed):
		raise MessageFramingError("stream did not write the complete message")

	flush = getattr(stream, "flush", None)
	if callable(flush):
		flush()

	return written


def _read_exact(stream: BufferedIOBase, size: int) -> bytes:
	"""Read exactly *size* bytes from a stream or raise EOFError."""

	chunks: list[bytes] = []
	remaining = size

	while remaining:
		chunk = stream.read(remaining)
		if not chunk:
			raise EOFError(f"unexpected end of stream while reading {size} bytes")
		chunks.append(chunk)
		remaining -= len(chunk)

	return b"".join(chunks)


__all__ = [
	"CHUNK_DATA",
	"ERROR",
	"FILE_CLOSE",
	"FILE_INFO",
	"FILE_OPEN",
	"GET_CHUNK",
	"HEARTBEAT",
	"LENGTH_PREFIX_SIZE",
	"LIST",
	"LIST_RESULT",
	"MAX_MESSAGE_SIZE",
	"MESSAGE_TYPES",
	"MessageFormatError",
	"MessageFramingError",
	"MessageTypeError",
	"ProtocolError",
	"REDUCE",
	"REDUCTION_RESULT",
	"REQUEST_TYPES",
	"RESPONSE_TYPES",
	"STAT",
	"STAT_RESULT",
	"VAR_INFO",
	"VAR_OPEN",
	"decode",
	"decode_payload",
	"encode",
	"encode_message",
	"make_message",
	"read_message",
	"validate_message",
	"validate_message_type",
	"write_message",
]