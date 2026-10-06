"""Live state of delegated subagents (hermes-agent ``delegate_task`` children).

A child agent runs in-process inside the parent's WebUI worker, under its own
session id (persisted by hermes-agent to state.db, shown read-only in the
WebUI). It never enters ``ACTIVE_RUNS``, so before this module its session
always read as settled ("Done") while it was still working.

Sources:

- ``note_progress`` is called from the parent worker's tool-progress callback
  (``on_tool``) for every ``subagent.*`` event. hermes-agent's
  ``_ChildProgressRelay`` threads ``subagent_id`` and, once the child exists,
  ``child_session_id`` into each one. That yields the child session →
  subagent mapping and the owning parent run.
- Liveness is hermes-agent's own registry (``list_active_subagents``): a
  child is running while its ``subagent_id`` is listed. It is registered for
  the child's whole run (async children included) and removed when the
  child finishes. Without that API (older agent builds), liveness falls back
  to "seen a ``subagent.*`` event in the last ``EVENT_TTL_S`` and no
  ``subagent.complete`` yet".

Unknown is not running: a child this process never saw an event for, or a
registry read that raises, reports False.

State layer: ``_CHILDREN`` under ``_LOCK`` (leaf; never held while calling
out). Side effects (poller registration, channel nudges) run outside it.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Optional

logger = logging.getLogger(__name__)

EVENT_TTL_S = 900.0
# subagent.start can precede the child's registry entry; count it as running
# for this long so a parent never briefly reads "0 working".
START_GRACE_S = 15.0
NUDGE_INTERVAL_S = 1.5
PARENT_STATUS_EVENT = "subagents_status"
_MAX_CHILDREN = 500

_LOCK = threading.Lock()
# child session id -> {subagent_id, parent_session_id, parent_stream_id,
#                      started_at, last_event_at, done, last_nudge_at}
_CHILDREN: "OrderedDict[str, dict]" = OrderedDict()


def note_progress(
    event_type: Any,
    kwargs: Optional[dict],
    *,
    parent_session_id: Optional[str],
    parent_stream_id: Optional[str],
    now: Optional[float] = None,
) -> None:
    """Record one relayed ``subagent.*`` event. Never raises."""
    try:
        if not isinstance(event_type, str) or not event_type.startswith("subagent"):
            return
        kw = kwargs if isinstance(kwargs, dict) else {}
        child_sid = str(kw.get("child_session_id") or "").strip()
        if not child_sid:
            return
        now = time.time() if now is None else now
        complete = event_type == "subagent.complete"
        with _LOCK:
            rec = _CHILDREN.get(child_sid)
            first = rec is None
            if first:
                rec = {
                    "subagent_id": None,
                    "goal": None,
                    "parent_session_id": parent_session_id,
                    "parent_stream_id": parent_stream_id,
                    "started_at": now,
                    "done": False,
                    "last_nudge_at": float("-inf"),
                }
                _CHILDREN[child_sid] = rec
                while len(_CHILDREN) > _MAX_CHILDREN:
                    _CHILDREN.popitem(last=False)
            else:
                _CHILDREN.move_to_end(child_sid)
            if kw.get("subagent_id") and not rec["subagent_id"]:
                rec["subagent_id"] = str(kw["subagent_id"])
            if kw.get("goal") and not rec["goal"]:
                rec["goal"] = str(kw["goal"])[:200]
            rec["last_event_at"] = now
            was_done = rec["done"]
            if complete:
                rec["done"] = True
            nudge = complete or now - rec["last_nudge_at"] >= NUDGE_INTERVAL_S
            if nudge:
                rec["last_nudge_at"] = now
        from api import prefill_progress

        if first and not complete and parent_stream_id:
            prefill_progress.register_child(child_sid, parent_stream_id)
        if complete and not was_done:
            prefill_progress.close_child(child_sid)
        if nudge:
            _nudge(child_sid)
        if (first or (complete and not was_done)) and parent_session_id:
            notify_parent(parent_session_id, parent_stream_id)
    except Exception:
        logger.debug("subagent_runs: note_progress failed", exc_info=True)


def _active_subagent_ids() -> Optional[set]:
    """Running subagent ids from hermes-agent's registry, or None if the API
    is unavailable (then callers use the event fallback)."""
    try:
        from tools.delegate_tool_registry import list_active_subagents
    except Exception:
        return None
    try:
        return {str(r.get("subagent_id")) for r in list_active_subagents() if r.get("subagent_id")}
    except Exception:
        logger.debug("subagent_runs: list_active_subagents failed", exc_info=True)
        return set()


def is_running(session_id: Any, *, now: Optional[float] = None) -> bool:
    """Whether ``session_id`` is a delegated child that is still working."""
    sid = str(session_id or "").strip()
    if not sid:
        return False
    with _LOCK:
        rec = _CHILDREN.get(sid)
        rec = dict(rec) if rec else None
    return _rec_running(rec, _active_subagent_ids(), time.time() if now is None else now)


def _rec_running(rec: Optional[dict], active: Optional[set], now: float) -> bool:
    if not rec or rec["done"]:
        return False
    if active is not None and rec.get("subagent_id"):
        if rec["subagent_id"] in active:
            return True
        return now - rec.get("started_at", 0) < START_GRACE_S
    return now - rec.get("last_event_at", 0) < EVENT_TTL_S


def running_children(parent_session_id: Any, *, now: Optional[float] = None) -> list:
    """Running children of a parent session, oldest first:
    ``[{session_id, goal, started_at}]``."""
    psid = str(parent_session_id or "").strip()
    if not psid:
        return []
    with _LOCK:
        recs = [(sid, dict(r)) for sid, r in _CHILDREN.items() if r.get("parent_session_id") == psid]
    if not recs:
        return []
    active = _active_subagent_ids()
    now = time.time() if now is None else now
    out = [
        {"session_id": sid, "goal": r.get("goal"), "started_at": r.get("started_at")}
        for sid, r in recs if _rec_running(r, active, now)
    ]
    out.sort(key=lambda c: c.get("started_at") or 0)
    return out


def notify_parent(parent_session_id: str, parent_stream_id: Optional[str] = None) -> None:
    """Push the parent's current running-children list to its open views:
    the parent's live stream while its turn runs, and its per-session channel
    (a turn that already finished while async children keep going). Both
    deliver to connected tabs only. Never raises."""
    try:
        payload = {"session_id": parent_session_id, "children": running_children(parent_session_id)}
        if parent_stream_id:
            from api import config as _cfg

            with _cfg.STREAMS_LOCK:
                channel = _cfg.STREAMS.get(parent_stream_id)
            put = getattr(channel, "put_if_subscribed", None)
            if callable(put):
                put((PARENT_STATUS_EVENT, dict(payload, stream_id=parent_stream_id), None))
        from api.background_process import get_session_channel

        ch = get_session_channel(parent_session_id)
        if ch is not None:
            ch.emit(PARENT_STATUS_EVENT, payload)
    except Exception:
        logger.debug("subagent_runs: notify_parent failed", exc_info=True)
    try:
        # The /api/sessions payload is cached (45 s while a turn streams), so
        # a child row's liveness dot would lag its start/finish: invalidate
        # and tell open sidebars to refetch, like a WebUI run does.
        from api.session_events import publish_session_list_changed

        publish_session_list_changed(reason="subagent_status", session_id=parent_session_id)
    except Exception:
        logger.debug("subagent_runs: session list publish failed", exc_info=True)


def child_parent(session_id: Any) -> Optional[dict]:
    """``{parent_session_id, parent_stream_id}`` for a known child, else None."""
    with _LOCK:
        rec = _CHILDREN.get(str(session_id or ""))
        if not rec:
            return None
        return {"parent_session_id": rec["parent_session_id"], "parent_stream_id": rec["parent_stream_id"]}


def _child_message_count(session_id: str) -> Optional[int]:
    try:
        from api.models import get_cli_session_messages

        return len(get_cli_session_messages(session_id))
    except Exception:
        logger.debug("subagent_runs: message count failed for %s", session_id, exc_info=True)
        return None


def _nudge(child_sid: str) -> None:
    """Tell a tab viewing the child that its transcript moved on.

    Reuses the per-session channel's ``session-updated`` frame, which the
    browser answers with an incremental in-place reload when the count is
    ahead of what it shows. Only when someone is subscribed (the count read
    hits state.db)."""
    from api.background_process import get_session_channel

    ch = get_session_channel(child_sid)
    if ch is None or not getattr(ch, "_subscribers", None):
        return
    count = _child_message_count(child_sid)
    if count is None:
        return
    ch.emit("session-updated", {"session_id": child_sid, "message_count": count, "source": "subagent_progress"})


def _reset_for_tests() -> None:
    with _LOCK:
        _CHILDREN.clear()
