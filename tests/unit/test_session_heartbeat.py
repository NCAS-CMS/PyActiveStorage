"""Tests for background heartbeat policy in p5remSession."""

from __future__ import annotations

import time
from io import BytesIO

from activestorage.session import p5remSession


class _HeartbeatFailingSession(p5remSession):
	def __init__(self) -> None:
		super().__init__(stdin=BytesIO(), stdout=BytesIO())
		self.heartbeat_calls = 0

	def heartbeat(self):
		self.heartbeat_calls += 1
		raise RuntimeError("heartbeat failure")


def test_heartbeat_failure_callback_invoked_after_threshold() -> None:
	session = _HeartbeatFailingSession()
	failures: list[str] = []

	def on_failure(_session, exc: Exception) -> None:
		failures.append(str(exc))
		_session.stop_heartbeat()

	session.set_heartbeat_failure_callback(on_failure)
	session.start_heartbeat(interval=0.01, max_failures=2)
	try:
		# Allow enough time for a couple of heartbeat iterations.
		time.sleep(0.1)
		assert len(failures) >= 1
		assert "heartbeat failure" in failures[0]
		assert session.heartbeat_calls >= 2
	finally:
		session.close()
