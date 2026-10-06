"""Delegated subagent liveness, child PP/TG and parent running-children.

Server side of the subagent live view (api/subagent_runs.py plus the child
paths in api/prefill_progress.py). hermes-agent's registry and the
per-session channels are faked; no agent runs, no network.
"""
from __future__ import annotations

import json
import queue
import re
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import api.config as config
from api import prefill_progress as pp
from api import subagent_runs as sr

REPO_ROOT = Path(__file__).resolve().parent.parent
ORIGIN = "http://ada.vpn:18020"
CFG = {"webui": {"prefill_progress": {"enabled_base_urls": [ORIGIN]}}}
SECRET = "sk-sub-SECRET"


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    sr._reset_for_tests()
    pp._reset_for_tests()
    # Default: hermes-agent registry available and empty; tests set it.
    monkeypatch.setattr(sr, "_active_subagent_ids", lambda: set())
    yield
    sr._reset_for_tests()
    pp._reset_for_tests()


@pytest.fixture
def calls(monkeypatch):
    """Record the side effects note_progress triggers."""
    log = {"register": [], "close": [], "nudge": [], "parent": []}
    monkeypatch.setattr(pp, "register_child", lambda c, p: log["register"].append((c, p)) or True)
    monkeypatch.setattr(pp, "close_child", lambda c: log["close"].append(c))
    monkeypatch.setattr(sr, "_nudge", lambda c: log["nudge"].append(c))
    monkeypatch.setattr(sr, "notify_parent", lambda p, s=None: log["parent"].append((p, s)))
    return log


def _ev(kind, child="c1", sub="sa1", **kw):
    return dict({"child_session_id": child, "subagent_id": sub, "goal": "compute fib"}, **kw)


# ── liveness ────────────────────────────────────────────────────────────────

def test_lifecycle_registry_is_authoritative(monkeypatch, calls):
    active = {"sa1"}
    monkeypatch.setattr(sr, "_active_subagent_ids", lambda: set(active))
    sr.note_progress("subagent.start", _ev("start"), parent_session_id="P", parent_stream_id="S")
    assert sr.is_running("c1") is True
    assert calls["register"] == [("c1", "S")] and calls["parent"] == [("P", "S")]
    sr.note_progress("subagent.tool", _ev("tool"), parent_session_id="P", parent_stream_id="S")
    assert calls["register"] == [("c1", "S")]  # once
    active.clear()  # registry dropped it (finished) before the complete event
    t0 = time.time()
    assert sr.is_running("c1", now=t0 + sr.START_GRACE_S + 1) is False
    sr.note_progress("subagent.complete", _ev("complete", status="completed"),
                     parent_session_id="P", parent_stream_id="S")
    assert sr.is_running("c1") is False
    assert calls["close"] == ["c1"] and calls["parent"][-1] == ("P", "S")


def test_start_grace_before_registry_entry(calls):
    sr.note_progress("subagent.start", _ev("start"), parent_session_id="P", parent_stream_id="S")
    t0 = time.time()
    assert sr.is_running("c1", now=t0) is True  # not in registry yet, just started
    assert sr.is_running("c1", now=t0 + sr.START_GRACE_S + 1) is False


def test_event_ttl_fallback_without_registry(monkeypatch, calls):
    monkeypatch.setattr(sr, "_active_subagent_ids", lambda: None)
    sr.note_progress("subagent.tool", _ev("tool"), parent_session_id="P", parent_stream_id="S", now=1000.0)
    assert sr.is_running("c1", now=1000.0 + sr.EVENT_TTL_S - 1) is True
    assert sr.is_running("c1", now=1000.0 + sr.EVENT_TTL_S + 1) is False


def test_unknown_or_malformed_is_not_running(calls):
    assert sr.is_running("nope") is False and sr.is_running(None) is False
    sr.note_progress("subagent.start", {"subagent_id": "x"}, parent_session_id="P", parent_stream_id="S")
    sr.note_progress("tool.started", _ev("t"), parent_session_id="P", parent_stream_id="S")
    sr.note_progress("subagent.start", None, parent_session_id="P", parent_stream_id="S")
    assert sr._CHILDREN == {} and calls["register"] == []


def test_registry_api_failure_fails_closed(monkeypatch, calls):
    sr.note_progress("subagent.start", _ev("start"), parent_session_id="P", parent_stream_id="S", now=0.0)
    monkeypatch.setattr(sr, "_active_subagent_ids", lambda: set())  # listing raised -> empty set
    assert sr.is_running("c1", now=sr.START_GRACE_S + 5) is False


def test_working_predicate_consults_subagents(monkeypatch, calls):
    import api.models as models
    import api.routes as routes

    monkeypatch.setattr(routes, "_active_run_stream_for_session", lambda sid: None)
    monkeypatch.setattr(sr, "_active_subagent_ids", lambda: {"sa1"})
    sr.note_progress("subagent.start", _ev("start"), parent_session_id="P", parent_stream_id="S")
    assert models._is_session_working("c1") is True
    assert models._is_session_working("someone-else") is False


# ── parent: running children ────────────────────────────────────────────────

def test_running_children_per_parent(monkeypatch, calls):
    monkeypatch.setattr(sr, "_active_subagent_ids", lambda: {"a", "b", "z"})
    sr.note_progress("subagent.start", _ev("s", child="c2", sub="b", goal="second"),
                     parent_session_id="P", parent_stream_id="S", now=200.0)
    sr.note_progress("subagent.start", _ev("s", child="c1", sub="a", goal="first"),
                     parent_session_id="P", parent_stream_id="S", now=100.0)
    sr.note_progress("subagent.start", _ev("s", child="cz", sub="z"),
                     parent_session_id="OTHER", parent_stream_id="S2")
    kids = sr.running_children("P")
    assert [(k["session_id"], k["goal"]) for k in kids] == [("c1", "first"), ("c2", "second")]
    sr.note_progress("subagent.complete", _ev("c", child="c1", sub="a"), parent_session_id="P", parent_stream_id="S")
    assert [k["session_id"] for k in sr.running_children("P")] == ["c2"]
    assert sr.running_children("") == [] and sr.running_children("nobody") == []


def test_notify_parent_reaches_stream_and_session_channel(monkeypatch):
    monkeypatch.setattr(sr, "_active_subagent_ids", lambda: {"sa1"})
    monkeypatch.setattr(pp, "register_child", lambda c, p: True)
    monkeypatch.setattr(sr, "_nudge", lambda c: None)
    ch = config.StreamChannel()
    q = ch.subscribe()
    monkeypatch.setitem(config.STREAMS, "S", ch)
    emitted = []

    class Sess:
        def emit(self, event, data):
            emitted.append((event, data))
            return 1

    import api.background_process as bp
    monkeypatch.setattr(bp, "get_session_channel", lambda sid: Sess() if sid == "P" else None)
    sr.note_progress("subagent.start", _ev("start"), parent_session_id="P", parent_stream_id="S")
    item = q.get_nowait()
    assert item[0] == "subagents_status" and item[2] is None  # ephemeral, no event id
    assert item[1]["stream_id"] == "S" and [c["session_id"] for c in item[1]["children"]] == ["c1"]
    assert emitted[0][0] == "subagents_status" and emitted[0][1]["session_id"] == "P"


def test_nudge_throttled_and_only_with_subscribers(monkeypatch):
    monkeypatch.setattr(pp, "register_child", lambda c, p: True)
    monkeypatch.setattr(pp, "close_child", lambda c: None)
    monkeypatch.setattr(sr, "notify_parent", lambda p, s=None: None)
    monkeypatch.setattr(sr, "_child_message_count", lambda sid: 7)
    emitted = []

    class Ch:
        _subscribers = [queue.Queue()]

        def emit(self, event, data):
            emitted.append((event, data))
            return 1

    import api.background_process as bp
    monkeypatch.setattr(bp, "get_session_channel", lambda sid: Ch())
    for i, t in enumerate((0.0, 0.5, 1.0, 2.0)):
        sr.note_progress("subagent.tool", _ev("t"), parent_session_id="P", parent_stream_id="S", now=t)
    assert [e[1]["message_count"] for e in emitted] == [7, 7]  # t=0 and t=2.0
    assert emitted[0] == ("session-updated", {"session_id": "c1", "message_count": 7, "source": "subagent_progress"})
    sr.note_progress("subagent.complete", _ev("c"), parent_session_id="P", parent_stream_id="S", now=2.1)
    assert len(emitted) == 3  # completion always nudges
    Ch._subscribers = []
    sr._reset_for_tests()
    sr.note_progress("subagent.tool", _ev("t"), parent_session_id="P", parent_stream_id="S", now=10.0)
    assert len(emitted) == 3


# ── child polling, delivery and stats ───────────────────────────────────────

@pytest.fixture
def no_thread(monkeypatch):
    class _T:
        def __init__(self, *a, **k):
            pass

        def start(self):
            pass

        def join(self, timeout=None):
            pass

    monkeypatch.setattr(pp.threading, "Thread", _T)


def test_register_child_inherits_parent_server(monkeypatch, no_thread):
    monkeypatch.setattr(pp, "_settings", lambda config_data=None: ({ORIGIN}, 5.0, 1.0))
    assert pp.register_child("c1", "S") is False  # parent not registered
    pp.register("S", "P", ORIGIN + "/v1", SECRET, config_data=CFG)
    assert pp.register_child("c1", "S") is True
    run = pp._RUNS["child:c1"]
    assert run["session_id"] == "c1" and run["origin"] == ORIGIN and run["api_key"] == SECRET and run["child"]
    pp.unregister("S")  # parent turn ends; async child keeps being observed
    assert "child:c1" in pp._RUNS


def test_child_frames_go_to_session_channel(monkeypatch, no_thread):
    monkeypatch.setattr(pp, "_settings", lambda config_data=None: ({ORIGIN}, 5.0, 1.0))
    pp.register("S", "P", ORIGIN, SECRET, config_data=CFG)
    pp.register_child("c1", "S")
    got = []

    class Sess:
        def emit(self, event, data):
            got.append((event, data))
            return 1

    import api.background_process as bp
    monkeypatch.setattr(bp, "get_session_channel", lambda sid: Sess() if sid == "c1" else None)
    assert pp._deliver("child:c1", "c1", {"x": 1}, event="prefill_progress", child=True) is True
    assert got == [("prefill_progress", {"x": 1})]
    monkeypatch.setattr(bp, "get_session_channel", lambda sid: None)
    assert pp._deliver("child:c1", "c1", {"x": 1}, child=True) is False


def test_child_call_stats_kept_and_attached_by_timestamp(monkeypatch, no_thread):
    monkeypatch.setattr(pp, "_settings", lambda config_data=None: ({ORIGIN}, 5.0, 1.0))
    import api.background_process as bp
    monkeypatch.setattr(bp, "get_session_channel", lambda sid: None)
    pp.register("S", "P", ORIGIN, SECRET, config_data=CFG)
    pp.register_child("c1", "S")
    pp.note_model_call("c1", "chatcmpl-a", 1, logged_at=1000.0)
    pp.note_model_call("c1", "chatcmpl-b", 2, logged_at=1010.0)
    fin = lambda rid, pps: {"request_id": rid, "phase": "finished", "completion_tokens": 10,  # noqa: E731
                            "pp_tok_s": pps, "tg_tok_s": 100.0, "computed_prompt_tokens": 50}

    class Resp:
        status = 200

        def __init__(self, body):
            self.body = body

        def read(self, n=-1):
            return self.body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    body = json.dumps({"running": 0, "waiting": 0, "requests": [fin("chatcmpl-a", 795.6), fin("chatcmpl-b", 887.0)]}).encode()
    monkeypatch.setattr(pp.urllib.request, "urlopen", lambda req, timeout=None: Resp(body))
    pp._poll_once({k: dict(v) for k, v in pp._RUNS.items()}, 5)
    msgs = [{"role": "user", "timestamp": 990.0},
            {"role": "assistant", "timestamp": 1000.4},
            {"role": "tool", "timestamp": 1001.0},
            {"role": "assistant", "timestamp": 1009.8},
            {"role": "assistant", "timestamp": 1300.0}]  # no call near it
    assert pp.attach_child_call_stats("c1", msgs) == 2
    assert msgs[1]["_callStats"]["pp_tok_s"] == 795.6 and msgs[1]["_callStats"]["request_id"] == "chatcmpl-a"
    assert msgs[3]["_callStats"]["pp_tok_s"] == 887.0
    assert "_callStats" not in msgs[4]
    assert pp.attach_child_call_stats("other", [{"role": "assistant", "timestamp": 1000.0}]) == 0


def test_closed_child_dropped_after_pending_stats(monkeypatch, no_thread):
    monkeypatch.setattr(pp, "_settings", lambda config_data=None: ({ORIGIN}, 5.0, 1.0))
    import api.background_process as bp
    monkeypatch.setattr(bp, "get_session_channel", lambda sid: None)
    pp.register("S", "P", ORIGIN, SECRET, config_data=CFG)
    pp.register_child("c1", "S")
    pp.note_model_call("c1", "chatcmpl-a", 1)
    pp.close_child("c1")

    class Resp:
        status = 200

        def __init__(self, reqs):
            self.body = json.dumps({"running": 0, "waiting": 0, "requests": reqs}).encode()

        def read(self, n=-1):
            return self.body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    seq = [[], [{"request_id": "chatcmpl-a", "phase": "finished", "completion_tokens": 3, "pp_tok_s": 1.0}]]
    monkeypatch.setattr(pp.urllib.request, "urlopen", lambda req, timeout=None: Resp(seq.pop(0)))
    pp._poll_once({k: dict(v) for k, v in pp._RUNS.items()}, 5)
    assert "child:c1" in pp._RUNS  # stats still pending
    pp._poll_once({k: dict(v) for k, v in pp._RUNS.items()}, 5)
    assert "child:c1" not in pp._RUNS and "S" in pp._RUNS


def test_dead_child_closed_and_parent_told(monkeypatch, no_thread):
    monkeypatch.setattr(pp, "_settings", lambda config_data=None: ({ORIGIN}, 5.0, 1.0))
    told = []
    monkeypatch.setattr(sr, "notify_parent", lambda p, s=None: told.append((p, s)))
    monkeypatch.setattr(sr, "_nudge", lambda c: None)
    pp.register("S", "P", ORIGIN, SECRET, config_data=CFG)
    sr.note_progress("subagent.start", _ev("s"), parent_session_id="P", parent_stream_id="S")
    told.clear()
    with pp._LOCK:
        pp._RUNS["child:c1"]["registered_at"] -= pp.CHILD_LIVENESS_GRACE_S + 1
    monkeypatch.setattr(sr, "is_running", lambda sid, now=None: False)
    pp._close_dead_children({k: dict(v) for k, v in pp._RUNS.items()})
    assert pp._RUNS["child:c1"]["closing"] is True
    assert told == [("P", "S")]


# ── wiring ──────────────────────────────────────────────────────────────────

def test_wiring_in_worker_and_routes():
    streaming = (REPO_ROOT / "api" / "streaming.py").read_text()
    i = streaming.index("            def on_tool(")
    body = streaming[i:i + 4000]
    assert "event_type.startswith('subagent')" in body
    assert "_subagent_runs.note_progress(" in body
    assert body.index("_subagent_runs.note_progress(") < body.index("if event_type in ('reasoning.available', '_thinking'):")
    routes = (REPO_ROOT / "api" / "routes.py").read_text()
    assert 'parsed.path == "/api/session/working"' in routes
    assert 'sess["working"] = bool(_is_session_working(sid))' in routes
    assert "attach_child_call_stats(sid, sess.get(\"messages\"))" in routes
    assert 'raw["subagents_running"] = _subagent_runs.running_children(' in routes
    assert '_row["is_streaming"] = True' in routes


# ── client ──────────────────────────────────────────────────────────────────

UI_JS = (REPO_ROOT / "static" / "ui.js").read_text(encoding="utf-8")
MESSAGES_JS = (REPO_ROOT / "static" / "messages.js").read_text(encoding="utf-8")
SESSIONS_JS = (REPO_ROOT / "static" / "sessions.js").read_text(encoding="utf-8")
I18N_JS = (REPO_ROOT / "static" / "i18n.js").read_text(encoding="utf-8")
NODE = shutil.which("node")


def _footer_status(session, kids, last=True, chip=None):
    src = """
const src = %s;
function extractFunc(name) {
  const start = src.search(new RegExp('function\\\\s+' + name + '\\\\s*\\\\('));
  let i = src.indexOf('{', start); let depth = 1; i++;
  while (depth > 0 && i < src.length) { if (src[i] === '{') depth++; else if (src[i] === '}') depth--; i++; }
  return src.slice(start, i);
}
global.S = {session: %s, activeStreamId: null};
global.esc = s => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/"/g,'&quot;');
global.t = (k, ...a) => ({working: 'Working…', subagents_working_one: '1 subagent working',
  subagents_working_many: '{0} subagents working'}[k] || k).replace('{0}', a[0]);
let _footerRenderingLastTurn = %s;
const _subagentsRunning = {};
const _LIVE_RUN_PREFILL_FRESH_MS = 15000;
const _liveRunPrefill = null;
global._liveRunPrefillLabel = () => (%s);
for (const f of ['setSubagentsRunning','_subagentsRunningFor','_subagentsLabel','_subagentsHtml',
                 '_footerStatusHtml','_workingFooterChipKey','_workingFooterLabelHtml']) eval(extractFunc(f));
setSubagentsRunning('P', %s);
console.log(JSON.stringify(_footerStatusHtml('Done')));
""" % (json.dumps(UI_JS), json.dumps(session), "true" if last else "false",
       json.dumps(chip), json.dumps(kids))
    res = subprocess.run([NODE], input=src, capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


node = pytest.mark.skipif(NODE is None, reason="node not on PATH")
KIDS = [{"session_id": "c1", "goal": "fib"}, {"session_id": "c2", "goal": "sum"}]


@node
def test_footer_shows_running_subagents_after_parent_settled():
    html = _footer_status({"session_id": "P", "working": False}, KIDS)
    assert "live-run-status-dot" in html and "2 subagents working" in html
    assert 'title="fib\nsum"' in html and "Working" not in html


@node
def test_footer_working_plus_subagents_and_plain_done():
    html = _footer_status({"session_id": "P", "working": True}, KIDS[:1])
    assert "Working…" in html and "1 subagent working" in html
    assert _footer_status({"session_id": "P", "working": False}, []) == '<span class="lf-status">Done</span>'


@node
def test_only_latest_turn_shows_liveness():
    html = _footer_status({"session_id": "P", "working": True}, KIDS, last=False)
    assert html == '<span class="lf-status">Done</span>'


@node
def test_child_footer_shows_chip():
    html = _footer_status({"session_id": "c1", "working": True}, [],
                          chip={"text": "Prefill 4.9k/11.5k · ETA 3s", "stale": False})
    assert "Prefill 4.9k/11.5k · ETA 3s" in html and "lf-prefill" in html


def test_client_wiring():
    for name in ("subagents_status", "prefill_progress", "llm_call_stats"):
        assert f"es.addEventListener('{name}'" in MESSAGES_JS, name
    assert "source.addEventListener('subagents_status'" in MESSAGES_JS
    assert "d.stream_id !== `child:${sid}`" in MESSAGES_JS
    assert "setSubagentsRunning(sid,S.session&&S.session.subagents_running)" in SESSIONS_JS
    assert "_scheduleWorkingSettleCheck(sid)" in SESSIONS_JS
    assert "/api/session/working?session_id=" in UI_JS
    assert "_footerRenderingLastTurn=_isLastAssistantTurn(turn);" in UI_JS
    working = len(re.findall(r"\n    working: '", I18N_JS))
    for k in ("subagents_working_one", "subagents_working_many"):
        assert len(re.findall(r"\n    %s: '" % k, I18N_JS)) == working >= 15, k
