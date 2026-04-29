"""Tests for protocol framing and message validation."""

from __future__ import annotations

from io import BytesIO
import struct

import cbor2
import pytest

from p5rem.protocol import CHUNK_DATA, FILE_OPEN, LIST_RESULT, MessageFormatError, MessageFramingError, MessageTypeError, VAR_OPEN, decode, decode_payload, encode, make_message, read_message, write_message


def test_round_trip_var_open_message() -> None:
	framed = encode(VAR_OPEN, path="/foo/bar.nc", varname="temperature")

	decoded = decode(framed)

	assert decoded["type"] == VAR_OPEN
	assert decoded["path"] == "/foo/bar.nc"
	assert decoded["varname"] == "temperature"


def test_chunk_message_preserves_binary_payload() -> None:
	payload = b"\x00\x01\x89chunk-data"

	framed = encode(
		CHUNK_DATA,
		byte_offset=1234,
		size=len(payload),
		filter_mask=0,
		data=payload,
	)

	decoded = decode(framed)

	assert decoded["type"] == CHUNK_DATA
	assert decoded["data"] == payload


def test_stream_read_write_round_trip() -> None:
	stream = BytesIO()

	bytes_written = write_message(stream, FILE_OPEN, path="/tmp/example.nc")
	stream.seek(0)
	message = read_message(stream)

	assert bytes_written == len(stream.getvalue())
	assert message == {"type": FILE_OPEN, "path": "/tmp/example.nc"}


def test_decode_rejects_mismatched_length_prefix() -> None:
	payload = cbor2.dumps({"type": LIST_RESULT, "entries": []})
	framed = struct.pack(">I", len(payload) + 1) + payload

	with pytest.raises(MessageFramingError):
		decode(framed)


def test_decode_payload_requires_message_mapping_with_known_type() -> None:
	with pytest.raises(MessageFormatError):
		decode_payload(cbor2.dumps(["not", "a", "mapping"]))

	with pytest.raises(MessageTypeError):
		decode_payload(cbor2.dumps({"type": "NOT_A_REAL_MESSAGE"}))


def test_make_message_rejects_duplicate_type_field() -> None:
	with pytest.raises(MessageFormatError):
		make_message(VAR_OPEN, type="VAR_INFO")