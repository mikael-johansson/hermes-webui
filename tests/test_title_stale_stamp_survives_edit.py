"""Regression tests for the stale-session-title bug (session ab97ffcb748b).

The placeholder title stamped from the first user message at stream
teardown (title_from() at streaming.py teardown) must carry creation-time
provenance (source message uid + content hash) that SURVIVES later edits
to the message text — the stamp identifies the source message's durable
identity, not a hash that changes with the text.

Covers:
  1. stamp at teardown: first user message gets a title_from() placeholder
     stamped at teardown; after the user EDITS the first message, the
     stamp is still present, still kind='placeholder', and still points at
     the same source message (uid unchanged; hash is the ORIGINAL content's
     hash, not the edited one).
  2. stamp identity fallbacks: message_uid preferred; stable id alias
     (id/message_id) next; pure content hash last.
  3. stamp gating: no stamp when the title is already non-default.
  4. clear: real title writes (mark_session_title_generated consumer in
     _run_background_title_update, apply_session_title_rename,
     clear_title_provenance) set title_provenance=None.
  5. persistence: the stamp survives save() + load() intact.

Imitates the fake-session construction patterns in
tests/test_2235_initial_aux_title.py and the Session construction /
session-dir isolation patterns from tests/test_session_title_provenance.py.
"""
import hashlib

import pytest

import api.models as models
from api.models import Session, title_from
from api.session_ops import (
    apply_session_title_rename,
    clear_title_provenance,
    mark_session_title_generated,
    stamp_placeholder_title_provenance,
)


@pytest.fixture(autouse=True)
def _isolate_session_dir(tmp_path, monkeypatch):
    """Redirect SESSION_DIR and SESSION_INDEX_FILE to a temp directory
    so tests don't touch the real session store.
    """
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    index_file = session_dir / "_index.json"

    monkeypatch.setattr(models, "SESSION_DIR", session_dir)
    monkeypatch.setattr(models, "SESSION_INDEX_FILE", index_file)

    models.SESSIONS.clear()
    if hasattr(models, "_PERSISTED_SESSION_IDS_CACHE"):
        models._PERSISTED_SESSION_IDS_CACHE = (None, None, frozenset())

    yield session_dir, index_file

    models.SESSIONS.clear()
    if hasattr(models, "_PERSISTED_SESSION_IDS_CACHE"):
        models._PERSISTED_SESSION_IDS_CACHE = (None, None, frozenset())


def _make_session(session_id, user_text, **kwargs):
    return Session(
        session_id=session_id,
        title=kwargs.pop("title", "Untitled"),
        messages=[
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": "ok"},
        ],
        **kwargs,
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── Stamp ────────────────────────────────────────────────────────────────────


def test_stamp_at_teardown_survives_first_message_edit():
    """The core regression: first turn stamped the placeholder title from
    the first user message; the user then EDITS that message's text. The
    title_from()-derived title text would change, but the provenance stamp
    must remain present, still identify a placeholder, and still point at
    the same source message — its identity no longer depends on current
    text equality.
    """
    user_text = "How do I make a REST API for user management"
    s = _make_session("ab97ffcb748b", user_text)
    assert s.title == "Untitled"
    assert s.title_provenance is None

    # Stream-teardown placeholder transition (title still default).
    stamp = stamp_placeholder_title_provenance(s, s.messages)
    assert stamp is not None
    s.title = title_from(s.messages, s.title)
    assert s.title == user_text[:64]
    assert s.title_provenance == stamp
    assert s.title_provenance["kind"] == "placeholder"
    # No durable id on the message here -> uid falls back to content hash.
    assert s.title_provenance["source_message_uid"] == _sha256(user_text)
    assert s.title_provenance["source_message_hash"] == _sha256(user_text)

    # The user edits the first message (new content -> new title_from text).
    first = s.messages[0]
    original_hash = s.title_provenance["source_message_hash"]
    original_uid = s.title_provenance["source_message_uid"]
    first["content"] = "Please redesign my whole billing system instead"

    # The placeholder stamp must survive the edit untouched.
    assert s.title_provenance is not None
    assert s.title_provenance["kind"] == "placeholder"
    assert s.title_provenance["source_message_uid"] == original_uid
    assert s.title_provenance["source_message_hash"] == original_hash
    # And it is NOT recomputed from the new text:
    new_text = first["content"]
    assert s.title_provenance["source_message_hash"] != _sha256(new_text)
    assert title_from(s.messages, "x") == new_text[:64]  # title text WOULD differ


def test_stamp_prefers_message_uid():
    """source_message_uid prefers the message's own message_uid over any
    fallback, and does not change when the text is edited."""
    user_text = "Summarize the quarterly report"
    s = _make_session("tp-uid", user_text)
    s.messages[0]["message_uid"] = "msg-abc-123"

    stamp = stamp_placeholder_title_provenance(s, s.messages)
    s.title = title_from(s.messages, s.title)
    assert stamp is not None
    assert s.title_provenance["source_message_uid"] == "msg-abc-123"
    assert s.title_provenance["source_message_hash"] == _sha256(user_text)

    # Edit the message: the uid-based identity must be stable.
    s.messages[0]["content"] = "Rewrite the whole report from scratch"
    assert s.title_provenance["source_message_uid"] == "msg-abc-123"
    assert s.title_provenance["source_message_hash"] == _sha256(user_text)


def test_stamp_falls_back_to_stable_id_alias():
    """Without message_uid, the stable id alias set (id/message_id) is
    used as the source identity."""
    user_text = "Fix the flaky deploy script"
    s = _make_session("tp-alias", user_text)
    s.messages[0]["id"] = "42"

    stamp = stamp_placeholder_title_provenance(s, s.messages)
    assert stamp is not None
    assert s.title_provenance["source_message_uid"] == "42"


def test_stamp_falls_back_to_content_hash():
    """With no durable id at all, the uid is the pure content hash."""
    user_text = "Why is my container OOMing?"
    s = _make_session("tp-hash", user_text)

    stamp = stamp_placeholder_title_provenance(s, s.messages)
    assert stamp is not None
    assert s.title_provenance["source_message_uid"] == _sha256(user_text)
    assert s.title_provenance["source_message_hash"] == _sha256(user_text)


def test_stamp_not_written_for_non_default_title():
    """The stamp only fires on a genuine default/empty -> title_from
    transition; an already-custom-titled session is left alone."""
    user_text = "Some normal first message"
    s = _make_session("tp-custom", user_text, title="My Custom Name")

    stamp = stamp_placeholder_title_provenance(s, s.messages)
    assert stamp is None
    assert s.title_provenance is None
    assert s.title == "My Custom Name"


def test_stamp_not_written_when_no_first_user_message():
    """No first user message text -> nothing to stamp."""
    s = Session(
        session_id="tp-no-user",
        title="Untitled",
        messages=[{"role": "assistant", "content": "hello"}],
    )
    stamp = stamp_placeholder_title_provenance(s, s.messages)
    assert stamp is None
    assert s.title_provenance is None


# ── Clear ────────────────────────────────────────────────────────────────────


def test_clear_title_provenance_sets_none():
    s = _make_session("tp-clear-1", "hi there")
    stamp = stamp_placeholder_title_provenance(s, s.messages)
    assert stamp is not None
    clear_title_provenance(s)
    assert s.title_provenance is None


def test_real_title_write_in_background_update_clears_stamp():
    """The LLM/fallback title write inside _run_background_title_update is
    a real title write: it must clear the placeholder stamp."""
    import threading
    from unittest.mock import MagicMock, patch

    from tests._aux_client_helpers import auxiliary_client_modules

    import api.streaming as streaming

    user_text = "Can you help me design a REST API for user management?"
    s = MagicMock()
    s.title = title_from([{"role": "user", "content": user_text}], "Untitled")
    s.llm_title_generated = False
    s.manual_title = False
    s.messages = [{"role": "user", "content": user_text}]
    s.session_id = "tp-bg-session"
    s.save = MagicMock()
    stamp = {
        "kind": "placeholder",
        "source_message_uid": "msg-1",
        "source_message_hash": _sha256(user_text),
    }
    s.title_provenance = stamp

    llm_title = "REST API Design"
    import api.profiles as profiles_api

    class _ProfileEnv:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    with auxiliary_client_modules():
        with patch.object(streaming, "get_session", return_value=s):
            with patch.object(streaming, "SESSIONS", {}):
                with patch.object(streaming, "LOCK", threading.Lock()):
                    with patch.object(
                        streaming,
                        "_aux_title_configured",
                        return_value=True,
                    ):
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
                                    assistant_text="Sure.",
                                    placeholder_title=s.title,
                                    put_event=lambda *a, **k: None,
                                    agent=None,
                                )
    assert s.title == llm_title
    assert s.llm_title_generated is True
    assert s.title_provenance is None, "real title write must clear the stamp"


def test_manual_rename_clears_stamp():
    """apply_session_title_rename (user rename / reset) is a real title
    write: the placeholder stamp is cleared."""
    user_text = "Design a caching layer for the API gateway"
    s = _make_session("tp-rename", user_text)
    stamp = stamp_placeholder_title_provenance(s, s.messages)
    s.title = title_from(s.messages, s.title)
    assert s.title_provenance == stamp

    apply_session_title_rename(s, "My Renamed Title")
    assert s.title == "My Renamed Title"
    assert s.title_provenance is None


def test_mark_generated_and_clear_coexist():
    """mark_session_title_generated does not touch the stamp by itself;
    the clear is a deliberate step at each real write site."""
    s = _make_session("tp-mark", "hi")
    stamp = stamp_placeholder_title_provenance(s, s.messages)
    mark_session_title_generated(s)
    assert s.llm_title_generated is True
    assert s.title_provenance == stamp  # mark alone does not clear
    clear_title_provenance(s)
    assert s.title_provenance is None


def test_sync_writer_stamps_provenance():
    """The sync (non-streaming) chat path is the second title_from() writer
    (routes.py _handle_chat_sync, 'Untitled'-only predicate). The extracted
    title-derivation branch must call stamp_placeholder_title_provenance
    before deriving the title, so a sync-path placeholder is stamped and
    becomes eligible for regeneration — same as the streaming teardown.

    The full _handle_chat_sync body needs an agent runtime, so this extracts
    just the title-derivation branch from the live source and execs it with
    a minimal session stub.
    """
    import re
    import textwrap

    src = open("api/routes.py").read()
    m = re.search(
        r"def _handle_chat_sync\(.*?\n(.*?)(?=\ndef )", src, re.DOTALL
    )
    assert m, "could not find _handle_chat_sync"
    body = m.group(1)
    branch = re.search(
        r"        if s\.title == \"Untitled\":\n(.*?)(?=^        \S)",
        body,
        re.DOTALL | re.MULTILINE,
    )
    assert branch, "sync 'Untitled' title-derivation branch not found"
    # The branch keeps the ORIGINAL 'Untitled'-only predicate.
    assert 'if s.title == "Untitled":' in branch.group(0)
    # The stamp is invoked before the title is derived.
    stamp_idx = branch.group(0).find("stamp_placeholder_title_provenance(")
    title_idx = branch.group(0).find("s.title = title_from(")
    assert stamp_idx != -1, "sync writer must call stamp_placeholder_title_provenance"
    assert title_idx != -1
    assert stamp_idx < title_idx, "stamp must happen before the title is derived"

    class _SyncStub:
        def __init__(self):
            self.title = "Untitled"
            self.title_provenance = None
            self.messages = [
                {"role": "user", "content": "Synchronous path first message"},
                {"role": "assistant", "content": "ok"},
            ]
        def save(self):
            return None

    s = _SyncStub()
    ns = {
        "s": s,
        "title_from": title_from,
        "stamp_placeholder_title_provenance": stamp_placeholder_title_provenance,
    }
    exec(textwrap.dedent(branch.group(0)), ns)
    assert s.title == "Synchronous path first message"[:64]
    assert s.title_provenance is not None
    assert s.title_provenance["kind"] == "placeholder"
    assert s.title_provenance["source_message_uid"] == _sha256(
        "Synchronous path first message"
    )
    assert s.title_provenance["source_message_hash"] == _sha256(
        "Synchronous path first message"
    )


def test_sync_writer_not_stamped_for_custom_title():
    """Same extracted sync branch: a session whose title is not exactly
    'Untitled' (the sync predicate is narrower than the streaming one)
    must not be stamped and keeps its title."""
    import re
    import textwrap

    src = open("api/routes.py").read()
    m = re.search(r"def _handle_chat_sync\(.*?\n(.*?)(?=\ndef )", src, re.DOTALL)
    branch = re.search(
        r"        if s\.title == \"Untitled\":\n(.*?)(?=^        \S)",
        m.group(1),
        re.DOTALL | re.MULTILINE,
    )
    assert branch

    class _SyncStub:
        def __init__(self):
            self.title = "My Custom Title"
            self.title_provenance = None
            self.messages = [
                {"role": "user", "content": "Synchronous path first message"},
            ]
        def save(self):
            return None

    s = _SyncStub()
    ns = {
        "s": s,
        "title_from": title_from,
        "stamp_placeholder_title_provenance": stamp_placeholder_title_provenance,
    }
    exec(textwrap.dedent(branch.group(0)), ns)
    assert s.title == "My Custom Title"
    assert s.title_provenance is None


# ── Persistence of the stamp ─────────────────────────────────────────────────


def test_stamp_persists_through_save_load_and_survives_edit():
    """The stamp round-trips through save() + load() intact, and the
    reloaded stamp still identifies the original (pre-edit) source message
    after the on-disk message text has been changed."""
    user_text = "Refactor the payment reconciliation service"
    s = _make_session("tp-persist", user_text)
    s.messages[0]["message_uid"] = "msg-persist-1"
    stamp = stamp_placeholder_title_provenance(s, s.messages)
    s.title = title_from(s.messages, s.title)
    assert s.title_provenance == stamp
    s.save()

    # Simulate the user editing the first message after the session is
    # persisted.
    raw = s.messages[0]
    raw["content"] = "Refactor the billing service, not payments"

    loaded = Session.load("tp-persist")
    assert loaded is not None
    assert loaded.title_provenance is not None
    assert loaded.title_provenance["kind"] == "placeholder"
    assert loaded.title_provenance["source_message_uid"] == "msg-persist-1"
    assert loaded.title_provenance["source_message_hash"] == _sha256(user_text)
    # The stored hash is the ORIGINAL text's hash, not the edited one:
    assert loaded.title_provenance["source_message_hash"] != _sha256(
        "Refactor the billing service, not payments"
    )
