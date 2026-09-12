"""Reproduce the subagent approval re-fire loop (no parked producer).

Root cause
----------
The WebUI registers the approval notify-callback per parent-turn (it is
bound to the turn's ``put()`` writer and is unregistered at turn teardown).
An async batch / subagent child resolves its approval session key to the
parent's WebUI id but fires its guard *after* the parent turn has ended, so
``tools.approval._gateway_notify_cbs`` has no callback for that key. The
guard then takes the non-blocking fallback (``approval.py`` ~4109) which
calls the raw agent-core ``submit_pending``: it writes a *tokenless*
``_pending[sid]`` single-dict (no ``approval_id``, no mirror token) and
returns ``pending_approval`` -- nothing is parked in ``_gateway_queues``.

The card surfaces in the WebUI. A human clicks "Approve (once)".
``_resolve_approval_legacy`` pops the tokenless entry, but:

  * "once" must not persist (the #6017 invariant), and
  * ``resolve_gateway_approval`` returns 0 because ``_gateway_queues`` is
    empty (nothing is blocked).

So the click is a no-op: it unblocks nothing and records no consent. The
child's (or parent's) next retry of the same command hits the guard again,
re-creates the tokenless card, and the loop repeats forever.

The fix
-------
A tokenless, producer-less card is a different animal from a parked
blocking approval. For it, "Approve" has no parked thread to unblock -- the
only meaningful, reversible action is to record *session-scope* consent via
the shared ``tools.approval`` module so the next guard call short-circuits
at its ``is_approved`` pre-check (checked before the blocking/fallback
fork) and the command actually runs.

These tests assert the observable behavior of that decision.
"""
from __future__ import annotations

import importlib

from tests.conftest import requires_agent_modules

pytestmark = requires_agent_modules


def _ta():
    """The shared agent-core approval module (same instance ``api.routes`` uses)."""
    return importlib.import_module("tools.approval")


def _reset(ta, sid: str) -> None:
    """Drop every approval-related piece of state for a session."""
    with ta._lock:
        ta._pending.pop(sid, None)
        ta._gateway_queues.pop(sid, None)
        ta._session_approved.pop(sid, None)
        ta._session_yolo.discard(sid)


def _seed_tokenless_fallback(ta, sid: str, pattern_key: str = "git:push"):
    """Reproduce the non-blocking fallback's exact pending shape.

    The agent-core guard's no-callback branch calls ``submit_pending(sid,
    {...})`` with a single dict and no ``approval_id`` -- this is what the
    stuck card in the WebUI looks like.
    """
    with ta._lock:
        ta._pending[sid] = {
            "command": "git push --force",
            "pattern_key": pattern_key,
            "pattern_keys": [pattern_key],
            "description": "Force push to remote",
        }
        # Ensure there is NO parked producer: the fallback never parks.
        ta._gateway_queues.pop(sid, None)


# --- the loop: approve must record session consent ------------------------
def test_tokenless_fallback_approve_once_records_session_consent():
    ta = _ta()
    from api.routes import (
        _GATEWAY_MIRROR_TOKEN,
        _lock,
        _pending,
        _resolve_approval_legacy,
    )
    from api.route_approvals import _gateway_queues

    # The fix only works because the resolver and the guard share the SAME
    # module objects (single editable install).
    assert _pending is ta._pending
    assert _lock is ta._lock

    sid = "1749ca005777"
    _reset(ta, sid)
    _seed_tokenless_fallback(ta, sid)

    # The tokenless card carries no approval_id and no mirror token, and
    # there is no parked producer.
    with _lock:
        entry = _pending[sid]
    assert not entry.get("approval_id")
    assert not entry.get(_GATEWAY_MIRROR_TOKEN)
    with _lock:
        assert not _gateway_queues.get(sid)

    # No consent before the click.
    assert not ta.is_approved(sid, "git:push")

    # Human clicks "Approve (once)" on the tokenless card.
    resolved = _resolve_approval_legacy(sid, "", "once")

    # The click is accepted ...
    assert resolved is True
    # ... and, crucially, it records session-scope consent so the child's
    # retry short-circuits the guard at its is_approved() pre-check.
    assert ta.is_approved(sid, "git:push")


def test_tokenless_fallback_approve_once_does_not_reappear():
    """The loop-break: a second guard pass must be already approved.

    Before the fix, approving the tokenless card left the session with no
    recorded consent, so the model's retry re-fired a *new* card. After the
    fix the retry sees the session already approved and runs.
    """
    ta = _ta()
    from api.routes import _pending, _resolve_approval_legacy

    sid = "loop-retry"
    _reset(ta, sid)
    _seed_tokenless_fallback(ta, sid)

    _resolve_approval_legacy(sid, "", "once")

    # First card was consumed.
    with ta._lock:
        assert _pending.get(sid) is None
    # A re-fired guard for the same pattern must now be pre-approved, i.e.
    # it does NOT create a new pending card.
    assert ta.is_approved(sid, "git:push")


# --- explicit session/always choices still work on the tokenless card ----
def test_tokenless_fallback_session_choice_persists_session_consent():
    ta = _ta()
    from api.routes import _resolve_approval_legacy

    sid = "tok-session"
    _reset(ta, sid)
    _seed_tokenless_fallback(ta, sid)
    assert _resolve_approval_legacy(sid, "", "session") is True
    assert ta.is_approved(sid, "git:push")


# --- regression guard: a PARKED blocking "once" must NOT persist (#6017) --
# This mirrors the shape test_issue6017 uses: an entry that IS parked in
# _gateway_queues (a real producer) must stay one-shot on "once". It must
# remain green before and after the fix.
def test_parked_blocking_approve_once_does_not_record_session_consent():
    ta = _ta()
    from api.routes import (
        _GATEWAY_MIRROR_TOKEN,
        _resolve_approval_legacy,
    )

    sid = "parked-once"
    _reset(ta, sid)

    # Build a real parked producer exactly like the blocking branch does.
    entry = ta._ApprovalEntry({
        "command": "rm -rf /tmp/x",
        "description": "Remove a temp dir",
        "session_key": sid,
        "pattern_key": "rm:rf",
        "pattern_keys": ["rm:rf"],
        "approval_id": "abc123",
    })
    with ta._lock:
        ta._gateway_queues[sid] = [entry]
        # A blocking card also mirrors a pending entry (with a mirror token).
        ta._pending[sid] = {
            "command": "rm -rf /tmp/x",
            "pattern_key": "rm:rf",
            "pattern_keys": ["rm:rf"],
            "description": "Remove a temp dir",
            _GATEWAY_MIRROR_TOKEN: "mirror-abc",
            "approval_id": "abc123",
        }

    # No consent before.
    assert not ta.is_approved(sid, "rm:rf")

    # "Approve (once)" on a PARKED card.
    resolved = _resolve_approval_legacy(sid, "abc123", "once")
    assert resolved is True

    # The parked producer was woken ...
    with ta._lock:
        assert not ta._gateway_queues.get(sid)
    # ... and, per #6017, it must NOT persist session consent: the next
    # identical guarded command must still produce a fresh card.
    assert not ta.is_approved(sid, "rm:rf")
