"""2026-09-30 subagent-approval re-fire regression: concurrent sibling.

Scenario
--------
A subagent child outlived its parent turn and its guard took the
non-blocking fallback: a *tokenless, producer-less* card
(``_pending[sid]``, no ``approval_id``, no mirror token) for pattern
``tirith:pipe_to_interpreter``.  *Concurrently*, a sibling producer in the
same session is parked (blocking) under the same inherited parent key, with
its *tokened* mirror behind the tokenless card in ``_pending[sid]``.

Before the fix, ``_resolve_approval_legacy`` computed
``has_parked_producer = bool(_gateway_queues.get(sid))`` — a SESSION-WIDE
check — and the Sept-12 Part-1 consent branch required
``not has_parked_producer``.  So on a "once" click:

  * ``reconcile_gateway_pending_mirror_locked`` (called first, under the
    lock) DISCARDED the tokenless card (no mirror token, no live-producer
    identity match), leaving the sibling mirror as the popped head; and
  * even had the tokenless card survived, the session-wide flag would have
    been True and the consent branch skipped.

Result: no session consent recorded for the tokenless card's pattern; the
child's retry of the same command re-fires the same card forever.

The Option-1 (WebUI-only) fix records consent per-card, before reconcile,
for exactly the tokenless, non-blocking-fallback cards the click targets
(``choice == "once"`` and no ``approval_id``).  These tests pin that
behavior; they fail (RED) before the fix and pass (GREEN) after.
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


def _seed_concurrent_sibling(ta, sid: str, include_tokenless: bool = True) -> None:
    """Tokenless fallback card at head + parked live sibling producer.

    The sibling is a real blocking producer (``_gateway_queues`` entry) whose
    mirror in ``_pending[sid]`` carries a token matching its
    ``_webui_mirror_token``, so ``reconcile_gateway_pending_mirror_locked``
    retains the mirror but discards the tokenless card.
    """
    from api.routes import _GATEWAY_MIRROR_TOKEN

    # A real parked producer, exactly like the blocking branch constructs it
    # (ta._ApprovalEntry carries the threading.Event the resolver wakes).
    sibling = ta._ApprovalEntry({
        "command": "git push",
        "approval_id": "sib-uuid",
        "pattern_key": "git:push",
        "pattern_keys": ["git:push"],
        "_webui_mirror_token": "tok-sib",
        "run_id": "",
    })
    cards: list[dict] = []
    if include_tokenless:
        cards.append({
            "command": "git show | python3 -c 'print(1)'",
            "pattern_key": "tirith:pipe_to_interpreter",
            "pattern_keys": ["tirith:pipe_to_interpreter"],
            "description": "Pipe output into an interpreter",
        })
    cards.append({
        "command": "git push",
        "pattern_key": "git:push",
        "pattern_keys": ["git:push"],
        _GATEWAY_MIRROR_TOKEN: "tok-sib",
        "approval_id": "sib-uuid",
    })
    with ta._lock:
        ta._gateway_queues[sid] = [sibling]
        ta._pending[sid] = cards


# --- the regression: a "once" click must record consent for the ----------
# --- tokenless card even while a sibling producer is parked --------------
def test_once_click_records_consent_for_tokenless_card_with_sibling_parked():
    ta = _ta()
    from api.routes import (
        _GATEWAY_MIRROR_TOKEN,
        _lock,
        _pending,
        _resolve_approval_legacy,
    )

    # The fix only works because the resolver and the guard share the SAME
    # module objects (single editable install).
    assert _pending is ta._pending
    assert _lock is ta._lock

    sid = "sib-regression"
    _reset(ta, sid)
    _seed_concurrent_sibling(ta, sid)

    # No consent before the click.
    assert not ta.is_approved(sid, "tirith:pipe_to_interpreter")

    # Human clicks "Approve (once)" on the tokenless card. No approval_id is
    # supplied (tokenless cards carry none), so the resolver's consent block
    # must key on ``approval_id is None``.
    resolved = _resolve_approval_legacy(sid, None, "once")
    assert resolved is True

    # The tokenless card's pattern must now carry session consent so the
    # child's retry short-circuits the guard at is_approved() (no re-fire).
    assert ta.is_approved(sid, "tirith:pipe_to_interpreter")
    # ... while the sibling's pattern must NOT persist (#6017: "once" is
    # one-shot for a real parked producer).
    assert not ta.is_approved(sid, "git:push")


# --- consent is per-card: clicking the SIBLING mirror must not record ----
# --- consent for the tokenless card --------------------------------------
def test_once_click_on_sibling_mirror_does_not_record_tokenless_consent():
    ta = _ta()
    from api.routes import _resolve_approval_legacy

    sid = "sib-mirror-click"
    _reset(ta, sid)
    _seed_concurrent_sibling(ta, sid)

    # Human clicks "Approve (once)" on the sibling's *tokened* mirror by
    # its approval_id.
    resolved = _resolve_approval_legacy(sid, "sib-uuid", "once")
    assert resolved is True

    # Consent must NOT be recorded for the tokenless card's pattern: the
    # click targeted the sibling mirror, and consent is per-card.
    assert not ta.is_approved(sid, "tirith:pipe_to_interpreter")
    # And, per #6017, the sibling's own pattern must not persist either.
    assert not ta.is_approved(sid, "git:push")


# --- #6017 guard: tokened/parked card only, "once" must not persist ------
def test_once_on_tokened_sibling_only_does_not_record_consent():
    ta = _ta()
    from api.routes import _resolve_approval_legacy

    sid = "sib-parked-only"
    _reset(ta, sid)
    _seed_concurrent_sibling(ta, sid, include_tokenless=False)

    resolved = _resolve_approval_legacy(sid, "", "once")
    assert resolved is True

    # A parked, tokened card is not a tokenless fallback: "once" stays
    # one-shot (#6017 invariant, preserved).
    assert not ta.is_approved(sid, "git:push")
