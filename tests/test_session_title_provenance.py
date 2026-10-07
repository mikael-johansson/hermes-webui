"""Tests for Session.title_provenance persistence.

title_provenance is a creation-time provenance record (dict) stamped on
placeholder titles so their identity survives edits to the underlying
message text. It must:
  1. round-trip through save() + load() and through load_metadata_only()
     (it is part of the metadata prefix);
  2. load as None on legacy sessions that have no such field;
  3. be preserved intact as a dict (never mangled to a string).

Imitates the Session construction / session-dir isolation patterns from
tests/test_session_index.py and the title patterns in
tests/test_2235_initial_aux_title.py.
"""
import json

import pytest

import api.models as models
from api.models import Session


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

    # Clear the in-memory SESSIONS and persisted-id caches to avoid bleed.
    models.SESSIONS.clear()
    if hasattr(models, "_PERSISTED_SESSION_IDS_CACHE"):
        models._PERSISTED_SESSION_IDS_CACHE = (None, None, frozenset())

    yield session_dir, index_file

    models.SESSIONS.clear()
    if hasattr(models, "_PERSISTED_SESSION_IDS_CACHE"):
        models._PERSISTED_SESSION_IDS_CACHE = (None, None, frozenset())


PROVENANCE = {
    "kind": "placeholder",
    "source": "first_user_message",
    "created_at": 1720000000.0,
}


def _make_session(session_id, **kwargs):
    return Session(
        session_id=session_id,
        title=kwargs.pop("title", "Untitled"),
        messages=[{"role": "user", "content": "hi"}],
        **kwargs,
    )


def test_default_is_none():
    """A session constructed without title_provenance has None (no
    migration of existing sessions)."""
    s = _make_session("tp-default")
    assert s.title_provenance is None


def test_non_dict_is_coerced_to_none():
    """Malformed persisted values must normalize to None, not crash."""
    s = _make_session("tp-str", title_provenance="placeholder")
    assert s.title_provenance is None
    s2 = _make_session("tp-list", title_provenance=[{"kind": "placeholder"}])
    assert s2.title_provenance is None


def test_save_load_roundtrip_preserves_dict():
    """A title_provenance dict persists through save() + load() intact."""
    s = _make_session("tp-roundtrip", title_provenance=PROVENANCE)
    s.save()

    raw = json.loads(s.path.read_text(encoding="utf-8"))
    assert raw["title_provenance"] == PROVENANCE

    loaded = Session.load("tp-roundtrip")
    assert loaded is not None
    assert loaded.title_provenance == PROVENANCE
    assert isinstance(loaded.title_provenance, dict)
    # Must be the dict intact, not a stringified repr.
    assert not isinstance(loaded.title_provenance, str)


def test_load_metadata_only_preserves_dict():
    """title_provenance lives in the metadata prefix, so
    load_metadata_only() must see it without parsing messages."""
    s = _make_session("tp-meta", title_provenance=PROVENANCE)
    s.save()

    meta = Session.load_metadata_only("tp-meta")
    assert meta is not None
    assert meta.title_provenance == PROVENANCE
    assert isinstance(meta.title_provenance, dict)


def test_metadata_only_dict_not_mangled():
    """The metadata-only stub keeps the exact dict value (nested keys
    included), not a str() of it."""
    nested = {"kind": "llm", "source": "aux", "model": "qwen"}
    s = _make_session("tp-nested", title_provenance=nested)
    s.save()

    meta = Session.load_metadata_only("tp-nested")
    assert meta is not None
    assert meta.title_provenance == nested
    assert meta.title_provenance is not s.title_provenance  # fresh copy
    assert isinstance(meta.title_provenance["model"], str)


def test_legacy_session_without_field_loads_as_none():
    """A session file written before title_provenance existed must load
    with title_provenance=None (both full and metadata-only paths)."""
    s = _make_session("tp-legacy")
    s.save()
    # Simulate a legacy sidecar: strip the field from the on-disk JSON.
    raw = json.loads(s.path.read_text(encoding="utf-8"))
    raw.pop("title_provenance", None)
    s.path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded = Session.load("tp-legacy")
    assert loaded is not None
    assert loaded.title_provenance is None

    meta = Session.load_metadata_only("tp-legacy")
    assert meta is not None
    assert meta.title_provenance is None


def test_none_persists_and_reloads_as_none():
    """An explicitly-None value round-trips as None (legacy parity)."""
    s = _make_session("tp-none", title_provenance=None)
    s.save()

    loaded = Session.load("tp-none")
    assert loaded is not None
    assert loaded.title_provenance is None
