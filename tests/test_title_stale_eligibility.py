"""Core-fix regression tests: title generation ELIGIBILITY honors the
title_provenance stamp, so a title_from()-derived placeholder whose first
message was later EDITED becomes eligible for background regeneration again.

This is the core fix for session ab97ffcb748b, whose title was stuck at the
title_from() placeholder forever after its first user message was edited,
because _is_provisional_title re-derives the placeholder from the CURRENT
first-message text (which no longer matches) and llm_title_generated=False
then blocks all auto-regeneration.

Covers:
  * CORE: an edited placeholder carrying a LIVE title_provenance stamp is
    eligible via _background_title_generation_inputs (this test FAILS on the
    pre-change code and PASSES after the eligibility disjunction is added).
  * negative control: a genuinely LLM-generated title (llm_title_generated=True)
    stays NOT eligible even after its first message is edited (no over-
    regeneration).
  * gates: manual_title precedence (a manually-named session is never
    force-regenerated even if its text matches the provisional placeholder),
    read_only/imported sessions are never force-mutated, and the
    has_live_placeholder_provenance helper resolves by durable identity and
    degrades to the legacy text heuristic when the source message is gone.

Real-Session construction imitates tests/test_title_stale_stamp_survives_edit.py;
the background-update MagicMock pattern imitates tests/test_2235_initial_aux_title.py.
The new helper is imported lazily inside the tests that use it so this file
collects against the pre-change tree (used for the before-fails proof).
"""
import hashlib
import threading
from unittest.mock import MagicMock, patch

import pytest

import api.models as models
import api.profiles as profiles_api
from api.models import Session, title_from
from api.session_ops import (
    session_has_manual_title,
    stamp_placeholder_title_provenance,
)
from tests._aux_client_helpers import auxiliary_client_modules


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _real_session(session_id, user_text, assistant_text="Sure, here is a plan."):
    """Real Session with a durable message_uid on the first user message,
    mirroring real WebUI rows (which carry message_uid + id)."""
    first = {"role": "user", "content": user_text}
    first["message_uid"] = "uid-" + session_id
    first["id"] = 1
    s = Session(
        session_id=session_id,
        title="Untitled",
        messages=[
            first,
            {"role": "assistant", "content": assistant_text},
        ],
    )
    return s


def _stamp_and_title(s):
    """Stream-teardown placeholder transition: stamp provenance, then derive
    the title from the (pre-edit) first user message."""
    stamp = stamp_placeholder_title_provenance(s, s.messages)
    s.title = title_from(s.messages, s.title)
    return stamp


# ── CORE: the ab97ffcb748b reproduction ─────────────────────────────────────


def test_edited_placeholder_with_live_stamp_is_eligible():
    """Core fix: a placeholder that carries a LIVE title_provenance stamp
    (whose source message still resolves by durable identity) is ELIGIBLE for
    background regeneration even after the first message is edited.

    On the pre-change code this returns None (the _is_provisional_title text
    heuristic no longer matches the edited text, so eligible_title is False) —
    this is the exact stale-title bug. After the eligibility disjunction it
    returns the (user, assistant) inputs.
    """
    from api.streaming import _background_title_generation_inputs

    user_text = "Please let me recap where we are with our AI hardware strategy discussion"
    s = _real_session("ab97ffcb748b", user_text)
    stamp = _stamp_and_title(s)
    assert stamp is not None
    assert s.title == user_text[:64]
    assert s.title_provenance["source_message_uid"] == "uid-ab97ffcb748b"
    assert s.llm_title_generated is False

    # The user edits the first message; the durable uid survives, the text does not.
    s.messages[0]["content"] = "Please redesign my whole billing system instead"

    # Pre-change: _is_provisional_title re-derives from the edited text -> no
    # match with the stored (original-derived) title -> NOT eligible -> None.
    # Post-change: the live stamp disjunction makes it eligible.
    inputs = _background_title_generation_inputs(s)
    assert inputs is not None, (
        "edited placeholder with a live title_provenance stamp must be "
        "eligible for background regeneration"
    )
    user, assistant = inputs
    assert user.startswith("Please redesign my whole billing system")
    assert assistant  # non-empty assistant snippet


# ── Negative control: genuinely generated titles stay final ─────────────────


def test_generated_title_not_reeligible_after_first_message_edit():
    """A genuinely LLM-generated title (llm_title_generated=True) must NOT
    become eligible for regeneration after its first message is edited — no
    over-regeneration. The live-stamp disjunction is gated on
    llm_title_generated=False, and the llm_title_generated blocking rule is
    preserved."""
    from api.streaming import _background_title_generation_inputs

    user_text = "Summarize the quarterly report"
    s = _real_session("gen-title-1", user_text)
    s.messages[0]["content"] = "Rewrite the whole report from scratch"
    # A real title write (mark + clear) leaves a genuine generated title.
    s.title = "Quarterly Report Summary"
    s.llm_title_generated = True
    s.title_provenance = None

    assert _background_title_generation_inputs(s) is None


def test_stale_stamp_with_unresolvable_uid_is_not_eligible():
    """When the source message is deleted/replaced so the stamp's uid no longer
    resolves, has_live_placeholder_provenance is False (degrades to the legacy
    text heuristic) and the session is NOT force-eligible via the live path."""
    from api.session_ops import has_live_placeholder_provenance
    from api.streaming import _background_title_generation_inputs

    s = _real_session("stale-1", "Original first message here")
    _stamp_and_title(s)
    # Replace the first message: new durable uid, so the original stamp uid
    # no longer resolves anywhere in the list.
    s.messages[0] = {
        "role": "user",
        "content": "A completely new first message",
        "message_uid": "uid-replaced",
        "id": 2,
    }
    assert has_live_placeholder_provenance(s) is False
    # No live stamp -> falls back to the text heuristic, which does not match a
    # stale stored title, so it is not eligible.
    assert _background_title_generation_inputs(s) is None


# ── Helper unit tests (lazy import keeps this file collectable pre-change) ──


def test_has_live_provenance_resolves_by_message_uid_after_edit():
    from api.session_ops import has_live_placeholder_provenance

    s = _real_session("live-uid", "Design a caching layer for the API gateway")
    _stamp_and_title(s)
    s.messages[0]["content"] = "Redesign the caching layer entirely"
    assert has_live_placeholder_provenance(s) is True


def test_has_live_provenance_falls_back_to_stable_id_alias():
    from api.session_ops import has_live_placeholder_provenance

    s = _real_session("live-alias", "Fix the flaky deploy script")
    s.messages[0].pop("message_uid")  # force the id/message_id alias path
    s.messages[0]["id"] = 42
    stamp = stamp_placeholder_title_provenance(s, s.messages)
    assert stamp["source_message_uid"] == "42"
    s.title = title_from(s.messages, s.title)
    s.messages[0]["content"] = "Rewrite the deploy script from scratch"
    assert has_live_placeholder_provenance(s) is True


def test_has_live_provenance_false_for_cleared_or_non_placeholder():
    from api.session_ops import (
        clear_title_provenance,
        has_live_placeholder_provenance,
    )

    s = _real_session("cleared-1", "Some first message")
    _stamp_and_title(s)
    clear_title_provenance(s)
    assert has_live_placeholder_provenance(s) is False

    # A non-placeholder kind never counts as live.
    s2 = _real_session("kind-1", "Another first message")
    s2.title_provenance = {"kind": "other", "source_message_uid": "x"}
    s2.messages[0]["message_uid"] = "x"
    assert has_live_placeholder_provenance(s2) is False

    # No provenance at all.
    s3 = _real_session("none-1", "Yet another first message")
    assert has_live_placeholder_provenance(s3) is False


# ── Gate: manual_title precedence overrides the live-stamp disjunction ──────


def _run_background_update_mock(s, llm_title, user_text, assistant_text):
    """Drive _run_background_title_update against a MagicMock session, mirroring
    the test_2235 aux-mocking pattern."""
    import api.streaming as streaming

    class _ProfileEnv:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    with auxiliary_client_modules():
        with patch.object(streaming, "get_session", return_value=s):
            with patch.object(streaming, "SESSIONS", {}):
                with patch.object(streaming, "LOCK", threading.Lock()):
                    with patch.object(streaming, "_aux_title_configured", return_value=True):
                        with patch.object(
                            streaming,
                            "_generate_llm_session_title_via_aux",
                            return_value=(llm_title, "llm_aux", llm_title),
                        ):
                            with patch.object(
                                profiles_api,
                                "profile_env_for_background_worker",
                                lambda *a, **k: _ProfileEnv(),
                            ):
                                streaming._run_background_title_update(
                                    session_id=s.session_id,
                                    user_text=user_text,
                                    assistant_text=assistant_text,
                                    placeholder_title=s.title,
                                    put_event=lambda *a, **k: None,
                                    agent=None,
                                )


def test_manual_title_never_force_regenerated_even_with_live_stamp():
    """manual_title precedence: a session whose text matches the provisional
    placeholder but is user-named (manual_title=True) is skipped in
    _run_background_title_update — the live-stamp disjunction must NOT
    override the manual_title gate."""
    user_text = "Design a caching layer for the API gateway"
    s = MagicMock()
    s.title = title_from([{"role": "user", "content": user_text}], "Untitled")
    s.llm_title_generated = False
    s.manual_title = True
    s.messages = [{"role": "user", "content": user_text}]
    s.session_id = "manual-gate"
    s.save = MagicMock()
    stamp = {
        "kind": "placeholder",
        "source_message_uid": "msg-1",
        "source_message_hash": _sha256(user_text),
    }
    s.title_provenance = stamp
    s.messages[0]["message_uid"] = "msg-1"
    assert session_has_manual_title(s) is True

    _run_background_update_mock(s, "Should Not Be Written", user_text, "Sure.")
    # The manual title must be preserved; the generated title must not be written.
    assert s.title == title_from([{"role": "user", "content": user_text}], "Untitled")
    assert s.save.called is False, "manual_title session must not be persisted by the auto path"


# ── Gate: read_only / imported sessions are never force-mutated ─────────────


def test_read_only_imported_session_not_force_regenerated(tmp_path, monkeypatch):
    """A read_only session is rejected by the title-persist path (PermissionError)
    — the live-stamp eligibility disjunction in streaming.py does not alter this
    gate, so imported/read-only sessions are never force-regenerated."""
    import api.routes as routes

    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    monkeypatch.setattr(models, "SESSION_DIR", session_dir)
    monkeypatch.setattr(routes, "SESSIONS", {})

    s = _real_session("import-ro", "Imported CLI first message")
    _stamp_and_title(s)
    s.read_only = True
    s.session_source = "external_agent"
    s.source_tag = "cli"
    s.raw_source = "cli"
    s.save(skip_index=True)  # persist so Session.load() resolves below

    with pytest.raises(PermissionError):
        routes._persist_generated_session_title(
            s,
            "Force-regenerated title",
            event_reason="session_title_regenerate",
            require_default_title=True,
        )
