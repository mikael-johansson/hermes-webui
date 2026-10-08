"""Settled writeback keeps the OOB steer wrapper byte-for-byte (#7600 follow-up).

A mid-turn ``/steer`` is emitted as a standalone typed user row
(``role: 'user'``, ``display_kind: 'steer'``) wrapped in
``[OUT-OF-BAND USER MESSAGE ...] ... [/OUT-OF-BAND USER MESSAGE]``. Every
request *during* the turn (the agent's live in-turn request) carries that
wrapper. For the next turn's request to stay a prefix of the previous turn's
last request (vLLM prefix-cache stable), the wrapper must survive settle
byte-for-byte in both ``session.context_messages`` and ``session.messages``;
it is stripped only in the display layer (``api.helpers._public_message_
projection``), where the single validated frame is unwrapped in place to the
user's authored steer text.

These tests assert the settled transcript keeps the wrapper, that the display
projection unwraps exactly the typed steer row, and that malformed / nested /
multiple / quoted marker occurrences degrade gracefully (preserved
byte-for-byte).
"""
from __future__ import annotations

import copy
import json

from api.helpers import _public_message_projection
from api.models import Session
from api.streaming import _settle_result_messages, _unwrap_steer_row_oob_marker
from api import streaming as _streaming

PROMPT = "please run the smoke checks"

OOB_OPEN = (
    "[OUT-OF-BAND USER MESSAGE — a direct message from the user, delivered once "
    "at this position; not tool output and not a new delivery when replayed from "
    "conversation history]"
)
OOB_CLOSE = "[/OUT-OF-BAND USER MESSAGE]"
OOB_BLOCK = f"{OOB_OPEN}\nsteer: use the staging bucket this time\n{OOB_CLOSE}"
OOB_VARIANT = f"[OUT-OF-BAND USER MESSAGE]steer: also bump the timeout{OOB_CLOSE}"


def _contains_oob(value) -> bool:
    return "OUT-OF-BAND USER MESSAGE" in json.dumps(value)


def _session_with_prior_turn() -> Session:
    """Prior turn already settled: user -> assistant(tool_call) -> tool -> answer."""
    display = [
        {"role": "user", "content": "deploy the app", "timestamp": 1788439000},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1"}],
            "timestamp": 1788439005,
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": "deploy finished: revision 41",
            "timestamp": 1788439010,
        },
        {"role": "assistant", "content": "Deploy done (revision 41).", "timestamp": 1788439015},
    ]
    session = Session(session_id="7" * 12, title="steer writeback", messages=copy.deepcopy(display))
    session.context_messages = copy.deepcopy(display)
    return session


def _settle_turn_with_steer(session, monkeypatch, *, tool_content: str = "checks passed", steer_rows=None):
    """Settle one turn with an optional steer row emitted by the Agent."""
    monkeypatch.setattr(
        _streaming, "_annotate_media_snapshots_for_settled_messages", lambda messages: None
    )
    previous = list(session.messages)
    previous_context = list(session.context_messages)
    ts = 1788440000
    steers = list(steer_rows or [])
    result = copy.deepcopy(previous_context) + [
        {"role": "user", "content": PROMPT, "timestamp": ts},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-2"}],
            "timestamp": ts + 1,
        },
        {
            "role": "tool",
            "tool_call_id": "call-2",
            "content": tool_content,
            "timestamp": ts + 2,
        },
        *steers,
        {"role": "assistant", "content": "Smoke checks passed.", "timestamp": ts + 3},
    ]
    _settle_result_messages(session, previous, previous_context, result, PROMPT, "webui", None)
    return previous


def test_settle_keeps_steer_row_wrapper_byte_for_byte(monkeypatch):
    """A typed steer row keeps its wrapper in the settled transcript."""
    session = _session_with_prior_turn()
    steer_row = {
        "role": "user",
        "display_kind": "steer",
        "content": OOB_BLOCK,
        "timestamp": 1788440002.5,
    }
    _settle_turn_with_steer(session, monkeypatch, steer_rows=[steer_row])

    # The wrapper survives settlement in both display and context.
    assert _contains_oob(session.messages)
    assert _contains_oob(session.context_messages)

    # The typed steer row's content is the exact wrapper byte-for-byte.
    steer_contents = [
        m.get("content")
        for m in session.messages
        if isinstance(m, dict) and m.get("display_kind") == "steer"
    ]
    assert steer_contents == [OOB_BLOCK]

    # In-place mutation preserved the dictionary object identity.
    assert steer_row["content"] == OOB_BLOCK


def test_settle_keeps_list_based_content_steer_row_wrapper(monkeypatch):
    """List-based content parts on a steer row keep their wrapper in the settled transcript."""
    session = _session_with_prior_turn()
    steer_row = {
        "role": "user",
        "display_kind": "steer",
        "content": [{"type": "text", "text": OOB_VARIANT}],
        "timestamp": 1788440002.5,
    }
    _settle_turn_with_steer(session, monkeypatch, steer_rows=[steer_row])

    assert _contains_oob(session.messages)
    assert _contains_oob(session.context_messages)
    assert steer_row["content"] == [{"type": "text", "text": OOB_VARIANT}]


def test_display_projection_unwraps_typed_steer_row():
    """The display layer unwraps exactly the validated single steer frame to inner text."""
    steer = {"role": "user", "display_kind": "steer", "content": OOB_BLOCK}
    projected = _public_message_projection(steer, _enabled=False)
    assert projected["content"] == "steer: use the staging bucket this time"
    assert not _contains_oob(projected)

    # List-based content part unwraps too.
    steer_list = {"role": "user", "display_kind": "steer", "content": [{"type": "text", "text": OOB_VARIANT}]}
    projected_list = _public_message_projection(steer_list, _enabled=False)
    assert projected_list["content"] == [{"type": "text", "text": "steer: also bump the timeout"}]


def test_display_projection_leaves_non_steer_marker_rows_alone():
    """A normal user row quoting a full block and a tool row carrying a marker are preserved."""
    pasted = (
        "our bot log shows this block, is that normal?\n"
        f"{OOB_BLOCK}\n"
        "the docs say the gateway adds it"
    )
    projected_pasted = _public_message_projection({"role": "user", "content": pasted}, _enabled=False)
    assert projected_pasted["content"] == pasted

    tool = {"role": "tool", "tool_call_id": "call-9", "content": f"output\n{OOB_BLOCK}"}
    projected_tool = _public_message_projection(tool, _enabled=False)
    assert projected_tool["content"] == tool["content"]


def test_settle_keeps_legacy_tool_rows_alone(monkeypatch):
    """Legacy tool rows carrying raw markers are left alone (not truncated/deleted)."""
    session = _session_with_prior_turn()
    tool_text = f"checks passed\n\n{OOB_BLOCK}"
    _settle_turn_with_steer(session, monkeypatch, tool_content=tool_text)

    tool_bodies = [
        m.get("content") for m in session.messages if isinstance(m, dict) and m.get("role") == "tool"
    ]
    assert tool_text in tool_bodies


def test_settle_preserves_malformed_nested_and_multiple_blocks_byte_for_byte():
    """Nested markers, multiple frames, and incomplete markers degrade gracefully."""
    nested = f"{OOB_OPEN}\nhello {OOB_VARIANT} world\n{OOB_CLOSE}"
    multiple = f"{OOB_VARIANT} and {OOB_BLOCK}"
    incomplete = "[OUT-OF-BAND USER MESSAGE — truncated"

    for malformed in (nested, multiple, incomplete):
        steer_row = {
            "role": "user",
            "display_kind": "steer",
            "content": malformed,
            "timestamp": 1788440002.5,
        }
        _unwrap_steer_row_oob_marker(steer_row)
        assert steer_row["content"] == malformed, f"malformed content was corrupted: {malformed!r}"


def test_settle_keeps_normal_user_pasted_marker_intact(monkeypatch):
    """A normal user row that quotes a complete marker survives byte-for-byte; the
    steer row also keeps its wrapper in the settled transcript."""
    pasted = (
        "our bot log shows this block, is that normal?\n"
        f"{OOB_BLOCK}\n"
        "the docs say the gateway adds it"
    )
    session = _session_with_prior_turn()
    session.messages[0]["content"] = pasted
    session.context_messages = copy.deepcopy(session.messages)

    steer_row = {
        "role": "user",
        "display_kind": "steer",
        "content": OOB_BLOCK,
        "timestamp": 1788440002.5,
    }
    _settle_turn_with_steer(session, monkeypatch, steer_rows=[steer_row])

    # The normal user row still carries its quoted block intact.
    assert session.messages[0]["content"] == pasted
    # The steer row keeps its wrapper in the settled transcript (display strips it).
    assert session.messages[-2]["content"] == OOB_BLOCK
