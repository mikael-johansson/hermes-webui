"""#7600 follow-up: the settled transcript must keep the OOB steer wrapper.

The mid-turn ``/steer`` is delivered to the model as a standalone typed user
row (``role: 'user'``, ``display_kind: 'steer'``) wrapped in
``[OUT-OF-BAND USER MESSAGE ...] ... [/OUT-OF-BAND USER MESSAGE]``. Every
request *during* the turn (the agent's live in-turn request) carries that
wrapper. For the next turn's request to be a prefix of the previous turn's last
request (vLLM prefix-cache stable), the wrapper must therefore survive settle
byte-for-byte in both ``session.context_messages`` and ``session.messages``,
and be stripped only in the display layer.

These tests are the RED→GREEN proof: against the pre-fix code the settled rows
lose the wrapper (the bug), so ``test_settle_persists_oob_steer_row_wrapper``
and ``test_display_projection_strips_oob_steer_row`` FAIL; after the fix they
pass. ``test_next_turn_request_is_prefix_of_previous_turn_request`` pins the
spec's verification standard (hash-level prefix stability across the settle
boundary).
"""
from __future__ import annotations

import copy
import hashlib
import json

from api.helpers import _public_message_projection
from api.models import Session
from api.streaming import _sanitize_messages_for_agent, _settle_result_messages
from api import streaming as _streaming

# Exactly the wrapper the agent's prompt_builder.format_steer_marker emits.
OOB_OPEN = (
    "[OUT-OF-BAND USER MESSAGE — a direct message from the user, delivered "
    "once at this position; not tool output and not a new delivery when "
    "replayed from conversation history]"
)
OOB_CLOSE = "[/OUT-OF-BAND USER MESSAGE]"


def _steer_content(steer_text: str) -> str:
    return f"{OOB_OPEN}\n{steer_text}\n{OOB_CLOSE}"


def _contains_oob(value) -> bool:
    return "OUT-OF-BAND USER MESSAGE" in json.dumps(value)


def _row_content(msg) -> str:
    c = msg.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(str(p.get("text") or p.get("content") or "") for p in c if isinstance(p, dict))
    return ""


def _hash_rows(rows) -> str:
    """Canonical hash of a list of message rows (role+content, serialized)."""
    norm = [
        {
            "role": m.get("role"),
            "content": m.get("content"),
            "display_kind": m.get("display_kind"),
        }
        for m in rows
        if isinstance(m, dict)
    ]
    return hashlib.sha256(
        json.dumps(norm, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()


def _prior_turn_session() -> Session:
    """A settled prior turn: user -> assistant(tool_call) -> tool -> answer."""
    display = [
        {"role": "user", "content": "deploy the app", "timestamp": 1788439000},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call-1"}], "timestamp": 1788439005},
        {"role": "tool", "tool_call_id": "call-1", "content": "deploy finished: revision 41", "timestamp": 1788439010},
        {"role": "assistant", "content": "Deploy done (revision 41).", "timestamp": 1788439015},
    ]
    session = Session(session_id="7" * 12, title="oob settle", messages=copy.deepcopy(display))
    session.context_messages = copy.deepcopy(display)
    return session


def _build_turn_with_steer(session, steer_text="steer: use the staging bucket"):
    """Return (previous, previous_context, in_turn_request, result).

    ``in_turn_request`` is the agent's live request DURING the turn, built exactly
    the way webui sends it: the prior context plus this turn's rows (ending in the
    wrapped steer), passed through ``_sanitize_messages_for_agent``. This is the
    ground truth the next turn's request must be a prefix of.
    """
    previous = list(session.messages)
    previous_context = list(session.context_messages)
    ts = 1788440000
    steer_row = {
        "role": "user",
        "display_kind": "steer",
        "content": _steer_content(steer_text),
        "timestamp": ts + 2.5,
    }
    in_turn_rows = list(previous_context) + [
        {"role": "user", "content": "run the smoke checks", "timestamp": ts},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call-2"}], "timestamp": ts + 1},
        {"role": "tool", "tool_call_id": "call-2", "content": "checks passed", "timestamp": ts + 2},
        copy.deepcopy(steer_row),
    ]
    in_turn_request = _sanitize_messages_for_agent(in_turn_rows)
    # The agent's raw result (what settle consumes): the same rows plus the final
    # assistant answer.
    result = in_turn_rows + [
        {"role": "assistant", "content": "Smoke checks passed.", "timestamp": ts + 3},
    ]
    return previous, previous_context, in_turn_request, result


def test_settle_persists_oob_steer_row_wrapper(monkeypatch):
    """After settle, the steer row keeps its wrapper byte-for-byte in BOTH histories."""
    monkeypatch.setattr(_streaming, "_annotate_media_snapshots_for_settled_messages", lambda messages: None)
    session = _prior_turn_session()
    previous, previous_context, _in_turn_request, result = _build_turn_with_steer(session)
    wrapped = _steer_content("steer: use the staging bucket")

    _settle_result_messages(session, previous, previous_context, result, "run the smoke checks", "webui", None)

    # The settled context row is byte-identical to the in-turn wrapped row.
    ctx_steer = [m for m in session.context_messages if isinstance(m, dict) and m.get("display_kind") == "steer"]
    assert len(ctx_steer) == 1, "settled context must carry the typed steer row"
    assert ctx_steer[0]["content"] == wrapped, "settled context steer must keep the wrapper byte-for-byte"

    # The settled display row keeps the wrapper too (it is stripped for the UI in the display layer).
    disp_steer = [m for m in session.messages if isinstance(m, dict) and m.get("display_kind") == "steer"]
    assert len(disp_steer) == 1, "settled display must carry the typed steer row"
    assert disp_steer[0]["content"] == wrapped, "settled display steer must keep the wrapper byte-for-byte"
    assert _contains_oob(session.messages)
    assert _contains_oob(session.context_messages)


def test_next_turn_request_is_prefix_of_previous_turn_request(monkeypatch):
    """Spec verification standard: last request of turn N is a prefix of the first of N+1."""
    monkeypatch.setattr(_streaming, "_annotate_media_snapshots_for_settled_messages", lambda messages: None)
    session = _prior_turn_session()
    previous, previous_context, in_turn_request, result = _build_turn_with_steer(session)

    # The last in-turn request (as the model saw it, wrapper included).
    last_request_n = list(in_turn_request)
    # Settle the turn (the wrapper must survive into the persisted transcript).
    _settle_result_messages(session, previous, previous_context, result, "run the smoke checks", "webui", None)
    # Rebuild the first request of turn N+1 from the settled transcript (same sanitizer).
    next_request = _sanitize_messages_for_agent(list(session.context_messages))

    steer_idx = next(
        i for i, m in enumerate(last_request_n)
        if isinstance(m, dict) and _row_content(m).startswith(OOB_OPEN)
    )
    prefix_len = steer_idx + 1  # through and including the steer row

    # The settled rows up to the steer index are byte-identical to the in-turn rows.
    assert len(next_request) >= prefix_len
    for i in range(prefix_len):
        a = last_request_n[i]
        b = next_request[i]
        assert a.get("content") == b.get("content"), f"row {i} content diverged across the settle boundary"
        assert a.get("role") == b.get("role")

    # Hash-level proof of prefix stability: serialized rows up to the steer index.
    assert _hash_rows(last_request_n[:prefix_len]) == _hash_rows(next_request[:prefix_len]), (
        "the serialized message prefix up to the steer index must be byte-identical "
        "across the settle boundary (vLLM prefix-cache stable)"
    )


def test_display_projection_strips_oob_steer_row():
    """Display layer strips the wrapper from a typed steer row (and only there)."""
    steer = {
        "role": "user",
        "display_kind": "steer",
        "content": _steer_content("steer: use the staging bucket"),
    }
    projected = _public_message_projection(steer, _enabled=False)
    assert projected["content"] == "steer: use the staging bucket", "display must strip the wrapper to the inner text"
    assert not _contains_oob(projected)

    # A non-steer user row that merely quotes a full block is left byte-for-byte.
    normal = {"role": "user", "content": "our bot log shows this block, is that normal?"}
    projected_normal = _public_message_projection(normal, _enabled=False)
    assert projected_normal["content"] == normal["content"]

    # A tool row with a marker in its body is left byte-for-byte.
    tool = {"role": "tool", "tool_call_id": "call-9", "content": f"output\n{_steer_content('legacy steer')}"}
    projected_tool = _public_message_projection(tool, _enabled=False)
    assert projected_tool["content"] == tool["content"]
