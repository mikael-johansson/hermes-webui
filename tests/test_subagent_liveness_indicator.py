"""T2 part (A): a running subagent CHILD session's sidebar row carries liveness.

Root cause: subagent CHILD sessions are driven by a delegate worker, not by an
attached SSE client, so their ``session.active_stream_id`` is ``None`` (see the
ACTIVE_RUNS registrations that omit an attach id). The old row builder
(``_is_streaming_session``) keyed liveness on that attach id alone, so a child
row could never report streaming — no liveness indicator ever appeared next to
the child in the left pane.

The fix routes the row's ``is_streaming`` through the authoritative
``_is_session_working`` helper (which also consults ``ACTIVE_RUNS`` via
``routes._active_run_stream_for_session``). This test pins that: a child
session whose delegate run is alive in ``ACTIVE_RUNS`` but has no attach id must
come out of ``all_sessions()`` with ``is_streaming`` True, and must flip back to
``False`` the moment the worker unregisters.

Asserts on the FLAG value on the built row (never on source text).
"""
import time

import pytest

import api.config as cfg
import api.models as models
from api.models import (
    SESSIONS,
    STREAMS,
    all_sessions,
    new_session,
)


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Redirect session storage to a fresh tmp dir and clear the global
    registries so the liveness registry starts empty for each test."""
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    index_file = session_dir / "_index.json"
    monkeypatch.setattr(models, "SESSION_DIR", session_dir)
    monkeypatch.setattr(models, "SESSION_INDEX_FILE", index_file)
    SESSIONS.clear()
    STREAMS.clear()
    with cfg.ACTIVE_RUNS_LOCK:
        cfg.ACTIVE_RUNS.clear()
    yield session_dir
    # Always tear the run down so no stale ACTIVE_RUNS row leaks into other
    # tests (the sidebar liveness check is global).
    with cfg.ACTIVE_RUNS_LOCK:
        for stream_id in list(cfg.ACTIVE_RUNS.keys()):
            cfg.unregister_active_run(stream_id)
    with cfg.STREAMS_LOCK:
        cfg.STREAMS.clear()
    SESSIONS.clear()


def _make_child_session():
    """A subagent-style child: has a real message (so it is not hidden by the
    empty-Untitled filter) but NO attached SSE client — the child's
    ``active_stream_id`` is None; its liveness lives in ``ACTIVE_RUNS``."""
    s = new_session()
    s.messages.append({"role": "user", "content": "do the delegated work"})
    s.messages.append({"role": "assistant", "content": "working on it"})
    s.title = "Subagent child"
    # The subagent child is not driven by a WebUI SSE client.
    s.active_stream_id = None
    s.save()
    return s


def _register_child_run(session_id):
    """Mirror the delegate worker's ACTIVE_RUNS registration: a live run tied
    to the child session with NO attach id (active_stream_id=None)."""
    stream_id = f"subagent-run-{session_id}"
    cfg.register_active_run(
        stream_id,
        session_id=session_id,
        active_stream_id=None,
        started_at=time.time(),
        phase="running",
    )
    return stream_id


def _child_row(session_id):
    rows = all_sessions()
    by_id = {row["session_id"]: row for row in rows}
    return by_id.get(session_id)


def test_running_subagent_child_row_is_flagged_streaming():
    """While the delegate run is alive, the child row must carry is_streaming."""
    s = _make_child_session()
    stream_id = _register_child_run(s.session_id)
    try:
        row = _child_row(s.session_id)
        assert row is not None, "child session did not appear in the sidebar"
        # The attach id is None (subagent child), yet liveness must be True
        # because the ACTIVE_RUNS row is alive. This is red on the old
        # attach-id-only predicate and green after the fix.
        assert row["active_stream_id"] is None
        assert row["is_streaming"] is True
    finally:
        cfg.unregister_active_run(stream_id)


def test_subagent_child_row_clears_when_run_settles():
    """Once the worker's finally unregisters the run, the flag must clear."""
    s = _make_child_session()
    stream_id = _register_child_run(s.session_id)
    try:
        assert _child_row(s.session_id)["is_streaming"] is True
    finally:
        cfg.unregister_active_run(stream_id)
    # Run is gone -> the child is no longer working.
    assert _child_row(s.session_id)["is_streaming"] is False


def test_idle_subagent_child_row_is_not_flagged():
    """Control: a child with no live run and no attach id is not streaming."""
    s = _make_child_session()
    row = _child_row(s.session_id)
    assert row is not None
    assert row["is_streaming"] is False


def test_compact_liveness_for_child_run():
    """compact() (the in-memory overlay path) must agree with the index loop:
    a child run that is only visible in ACTIVE_RUNS is working."""
    s = _make_child_session()
    stream_id = _register_child_run(s.session_id)
    try:
        compact = s.compact(include_runtime=True)
        assert compact["is_streaming"] is True
    finally:
        cfg.unregister_active_run(stream_id)
    assert s.compact(include_runtime=True)["is_streaming"] is False
