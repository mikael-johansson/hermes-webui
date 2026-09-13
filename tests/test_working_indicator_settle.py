"""Premature-"Done" settle gate: the working indicator must track worker liveness.

The main-transcript turn footer used to stamp a hardcoded "Done" while the
agent run was still alive. The root cause is that liveness was keyed on the
ATTACH id (`session.active_stream_id`), which the worker clears (streaming.py
~11641) well before the run is actually torn down (`unregister_active_run` in
the worker's `finally`, ~12891). So during that window — the sharpest instance
being the DETACHED case, where the SSE stream entry is gone AND
`active_stream_id` is cleared but the worker is still in `ACTIVE_RUNS` — the
old predicate reports "not running" and the client settles the turn, stamping
a fake "Done".

The fix adds an authoritative liveness predicate `_is_session_working` in
api.models. It reports *working* iff the session has a live, non-stale
`ACTIVE_RUNS` row (reusing `routes._active_run_stream_for_session`, which
carries the 180s stale-unwind ceiling) OR its `active_stream_id` is still in
the active set. This test pins the settle decision on the worker lifecycle:

* DETACHED (no SSE stream, attach id cleared, worker in ACTIVE_RUNS) must
  report WORKING while the run is alive, and SETTLED once it unregisters.
  On the pre-fix tree (which had no `_is_session_working`, only the
  attach-id-only `_is_streaming_session`) the first assertion fails — proving
  the test is red-before / green-after.
* A cancel-unwind (phase="cancelling", fresh) still reports working, but a
  cancel past the 180s ceiling with no live stream is reaped -> settled, so a
  wedged worker can never hold the turn open forever.

Asserts on the actual working/is-streaming FLAG value, never on source text.
"""

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from api import config as cfg  # noqa: E402
from api import models as models_mod  # noqa: E402

# Resolve the liveness predicate under test. Prefer the NEW authoritative
# helper; fall back to the LEGACY attach-id-only logic so that, on the pre-fix
# tree (no `_is_session_working`), the test exercises exactly the buggy
# predicate and fails on a real assertion instead of an import error.
_liveness = getattr(models_mod, "_is_session_working", None)


def _is_working(session_id, active_stream_id=None):
    if _liveness is not None:
        return bool(_liveness(session_id, active_stream_id=active_stream_id))
    # Legacy pre-fix logic: only the attach id is consulted.
    return models_mod._is_streaming_session(
        active_stream_id, models_mod._active_stream_ids()
    )


def _clear(stream_id):
    cfg.unregister_active_run(stream_id)
    with cfg.STREAMS_LOCK:
        cfg.STREAMS.pop(stream_id, None)


def test_detached_run_reports_working_then_settles_after_unregister():
    """The DETACHED case: attach id cleared, no SSE stream, worker in ACTIVE_RUNS.

    While the run is alive the settle decision must be WORKING (not "Done"),
    and it must flip to SETTLED the moment the worker unregisters in its finally.
    """
    sid = "sess-working-detached"
    stream_id = "stream-working-detached"
    # 1. The worker enters ACTIVE_RUNS at run start (streaming.py ~8761).
    cfg.register_active_run(
        stream_id, session_id=sid, started_at=time.time(), phase="running"
    )
    try:
        # 2. The attach id is cleared and there is no SSE stream entry — the
        #    11641-equivalent detached state (stream popped / never registered).
        with cfg.STREAMS_LOCK:
            cfg.STREAMS.pop(stream_id, None)
        # 3. The turn must NOT settle: the run is still alive -> working.
        #    RED on the pre-fix tree (attach-id-only logic returns False here).
        assert _is_working(sid, None) is True
    finally:
        _clear(stream_id)
    # 4+5. Once the worker's finally unregisters the run, it is truly settled.
    assert _is_working(sid, None) is False


def test_attached_running_session_is_working():
    """Control: an attached turn whose stream id is in the active set is working."""
    sid = "sess-working-attached"
    stream_id = "stream-working-attached"
    cfg.register_active_run(stream_id, session_id=sid, started_at=time.time(), phase="running")
    with cfg.STREAMS_LOCK:
        cfg.STREAMS[stream_id] = object()  # live SSE channel -> in _active_stream_ids()
    try:
        assert _is_working(sid, stream_id) is True
    finally:
        _clear(stream_id)


def test_cancel_unwind_still_working_then_settles_when_stale():
    """A fresh cancel-unwind is still working; a stale one is reaped -> settled.

    cancel_stream() clears the attach id and removes the SSE stream while the
    worker unwinds, so the only liveness signal left is the ACTIVE_RUNS row.
    """
    sid = "sess-working-cancel"

    # Fresh cancel (5s ago, well under the 180s ceiling) with no SSE stream.
    stream_id = "stream-working-cancel-fresh"
    cfg.register_active_run(
        stream_id,
        session_id=sid,
        phase="cancelling",
        started_at=time.time() - 120.0,
        cancelled_at=time.time() - 5.0,
    )
    with cfg.STREAMS_LOCK:
        cfg.STREAMS.pop(stream_id, None)
    assert _is_working(sid, None) is True
    _clear(stream_id)

    # Stale cancel (200s past the ceiling, no live stream) must be reaped, so a
    # wedged worker cannot hold the "Working…" indicator open forever.
    stream_id2 = "stream-working-cancel-stale"
    cfg.register_active_run(
        stream_id2,
        session_id=sid,
        phase="cancelling",
        started_at=time.time() - 300.0,
        cancelled_at=time.time() - 200.0,
    )
    with cfg.STREAMS_LOCK:
        cfg.STREAMS.pop(stream_id2, None)
    assert _is_working(sid, None) is False
    _clear(stream_id2)


def test_unknown_session_is_settled():
    """Control: a session with no live run and no attach id is settled."""
    assert _is_working("sess-working-nonexistent-xyz", None) is False
