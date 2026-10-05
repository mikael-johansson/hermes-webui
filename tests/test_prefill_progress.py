"""Server-side prefill-progress poller (PREFILL_PROGRESS_WEBUI_SPEC.md §1, §3).

No network: ``urllib.request.urlopen`` is replaced with a scripted fake. The
emission tests use the real ``StreamChannel`` so the ephemeral delivery
contract (subscribers only, never the offline replay buffer, ``None`` event
id) is checked against the production class.
"""
from __future__ import annotations

import io
import json
import logging
import re
import threading
import time
import urllib.error
from pathlib import Path

import pytest

import api.config as config
from api import prefill_progress as pp

REPO_ROOT = Path(__file__).resolve().parent.parent
ORIGIN = "http://ada.vpn:18020"
CFG = {"webui": {"prefill_progress": {"enabled_base_urls": [ORIGIN], "poll_seconds": 5}}}
SECRET = "sk-test-SECRET-do-not-log"


def _req(**over):
    r = {
        "request_id": "chatcmpl-1", "phase": "prefill", "admitted": True,
        "arrived_ms": 1000, "admitted_ms": 500, "prompt_tokens": 180412,
        "computed_tokens": 41200, "cached_tokens": 0,
        "remaining_prompt_tokens": 139212, "prefill_rate_tok_s": 1850.0,
        "prefill_eta_s": 75.2, "restoring_kv": False, "restore_ms": None,
        "preemptions": 0,
    }
    r.update(over)
    return r


def _snap(requests, running=1, waiting=0, age=0.4):
    return {"ts": 1.0, "engine_ts_age_s": age, "running": running,
            "waiting": waiting, "kv_cache_pct": 4.9, "requests": requests}


class _Resp:
    def __init__(self, body: bytes, status=200):
        self._body = io.BytesIO(body)
        self.status = status

    def read(self, n=-1):
        return self._body.read(n)

    def getcode(self):
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeHTTP:
    """Scripted urlopen: each call pops the next outcome (last one repeats)."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def __call__(self, req, timeout=None):
        self.calls.append((req.full_url, dict(req.header_items()), timeout))
        out = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(out, int):
            raise urllib.error.HTTPError(req.full_url, out, "err", {}, io.BytesIO(b""))
        if isinstance(out, Exception):
            raise out
        if isinstance(out, bytes):
            return _Resp(out)
        return _Resp(json.dumps(out).encode())


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    pp._reset_for_tests()
    yield
    pp._reset_for_tests()


@pytest.fixture
def no_thread(monkeypatch):
    """Register without starting the poller so tests drive _poll_once."""
    started = []

    class _T:
        def __init__(self, *a, **k):
            started.append(self)

        def start(self):
            pass

        def join(self, timeout=None):
            pass

    monkeypatch.setattr(pp.threading, "Thread", _T)
    return started


@pytest.fixture
def live_stream(monkeypatch):
    """A registered active run with a real StreamChannel and one subscriber."""

    def make(stream_id="s1", session_id="sess1", subscribe=True):
        ch = config.StreamChannel()
        q = ch.subscribe() if subscribe else None
        monkeypatch.setitem(config.STREAMS, stream_id, ch)
        monkeypatch.setitem(config.ACTIVE_RUNS, stream_id, {"session_id": session_id})
        return ch, q

    return make


def _drain(q):
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def _runs():
    with pp._LOCK:
        return {k: dict(v) for k, v in pp._RUNS.items()}


# ── 1. normalization ────────────────────────────────────────────────────────

@pytest.mark.parametrize("url", [
    "http://ada.vpn:18020/v1/", "http://ADA.vpn:18020", " http://ada.vpn:18020/v1 ",
    "http://ada.vpn:18020/",
])
def test_normalize_matches(url):
    assert pp.normalize_origin(url) == ORIGIN


@pytest.mark.parametrize("url", [
    "http://ada.vpn:18021/v1", "http://ada.vpn:18020/other", "ftp://ada.vpn:18020",
    "", None, "ada.vpn:18020", "http://user:pw@ada.vpn:18020",
])
def test_normalize_rejects(url):
    assert pp.normalize_origin(url) != ORIGIN


# ── 2/3. registration and refcounted poller ─────────────────────────────────

def test_config_off_is_noop(no_thread):
    for cfg in ({}, {"webui": {}}, {"webui": {"prefill_progress": {"enabled_base_urls": []}}}):
        assert pp.register("s1", "sess1", ORIGIN + "/v1", SECRET, config_data=cfg) is False
    assert no_thread == [] and _runs() == {}


def test_non_matching_base_url_never_registers(no_thread):
    assert pp.register("s1", "sess1", "https://api.openai.com/v1", SECRET, config_data=CFG) is False
    assert pp.register("s1", "sess1", ORIGIN + "/v1", "", config_data=CFG) is False
    assert no_thread == [] and _runs() == {}


def test_provider_switch_reregisters_and_drops(no_thread):
    assert pp.register("s1", "sess1", ORIGIN + "/v1", SECRET, config_data=CFG)
    assert pp.register("s1", "sess1", "https://other/v1", SECRET, config_data=CFG) is False
    assert _runs() == {}


def test_poller_starts_on_first_register_and_exits_after_last(monkeypatch):
    http = FakeHTTP(_snap([]))
    monkeypatch.setattr(pp.urllib.request, "urlopen", http)
    pp.register("s1", "sess1", ORIGIN + "/v1", SECRET, config_data=CFG)
    pp.register("s2", "sess2", ORIGIN + "/v1", SECRET, config_data=CFG)
    t = pp._thread
    assert t is not None and t.is_alive()
    pp.unregister("s1")
    assert pp._thread is t
    pp.unregister("s2")
    t.join(timeout=5)
    assert not t.is_alive() and pp._thread is None


def test_register_racing_poller_exit_leaves_one_poller(monkeypatch):
    """The exit decision and the start decision share _LOCK."""
    monkeypatch.setattr(pp.urllib.request, "urlopen", FakeHTTP(_snap([])))
    for _ in range(50):
        pp.register("s1", "sess1", ORIGIN, SECRET, config_data=CFG)
        barrier = threading.Barrier(2)

        def flip(barrier=barrier):
            barrier.wait()
            pp.register("s2", "sess2", ORIGIN, SECRET, config_data=CFG)

        th = threading.Thread(target=flip)
        th.start()
        barrier.wait()
        pp.unregister("s1")
        th.join()
        deadline = time.time() + 2
        while time.time() < deadline:
            pollers = [x for x in threading.enumerate() if x.name == "prefill-progress-poller"]
            if len(pollers) == 1 and pp._thread in pollers:
                break
            time.sleep(0.01)
        pollers = [x for x in threading.enumerate() if x.name == "prefill-progress-poller"]
        assert len(pollers) == 1 and pp._thread is pollers[0]
        pp._reset_for_tests()


# ── 4/5. polling and emission ───────────────────────────────────────────────

def test_one_get_per_origin_fans_out(monkeypatch, no_thread, live_stream):
    _, q1 = live_stream("s1", "sess1")
    _, q2 = live_stream("s2", "sess2")
    http = FakeHTTP(_snap([_req()]))
    monkeypatch.setattr(pp.urllib.request, "urlopen", http)
    pp.register("s1", "sess1", ORIGIN + "/v1", SECRET, config_data=CFG)
    pp.register("s2", "sess2", ORIGIN, SECRET, config_data=CFG)
    pp._poll_once(_runs(), 5)
    assert len(http.calls) == 1
    url, headers, timeout = http.calls[0]
    assert url == ORIGIN + "/v1/prefill_progress"
    assert headers.get("Authorization") == f"Bearer {SECRET}"
    assert timeout == 4
    for q, sid, sess in ((q1, "s1", "sess1"), (q2, "s2", "sess2")):
        (item,) = _drain(q)
        assert item[0] == "prefill_progress" and item[2] is None
        assert item[1]["stream_id"] == sid and item[1]["session_id"] == sess


def test_happy_path_strips_finished_and_goes_quiet(monkeypatch, no_thread, live_stream):
    _, q = live_stream()
    fin = _req(request_id="chatcmpl-old", phase="finished", remaining_prompt_tokens=0)
    http = FakeHTTP(
        _snap([fin, _req()]),
        _snap([fin], running=0),
        _snap([fin], running=0),
    )
    monkeypatch.setattr(pp.urllib.request, "urlopen", http)
    pp.register("s1", "sess1", ORIGIN, SECRET, config_data=CFG)
    pp._poll_once(_runs(), 5)
    (item,) = _drain(q)
    data = item[1]
    assert [r["request_id"] for r in data["requests"]] == ["chatcmpl-1"]
    assert data["requests"][0] == _req()  # verbatim
    assert data["stale"] is False and data["running"] == 1
    pp._poll_once(_runs(), 5)
    (item,) = _drain(q)
    assert item[1]["requests"] == []  # one empty frame drops the chip
    pp._poll_once(_runs(), 5)
    assert _drain(q) == []  # then silence while idle


def test_idle_from_start_emits_nothing(monkeypatch, no_thread, live_stream):
    _, q = live_stream()
    monkeypatch.setattr(pp.urllib.request, "urlopen", FakeHTTP(_snap([], running=0)))
    pp.register("s1", "sess1", ORIGIN, SECRET, config_data=CFG)
    pp._poll_once(_runs(), 5)
    assert _drain(q) == []


def test_sse_frame_format():
    from api.streaming import _sse

    class H:
        def __init__(self):
            self.wfile = io.BytesIO()

    h = H()
    _sse(h, "prefill_progress", {"stream_id": "s1", "requests": []})
    assert h.wfile.getvalue().startswith(b"event: prefill_progress\ndata: {")


@pytest.mark.parametrize("failure", [500, b"not json", TimeoutError("t"), 401])
def test_failure_keeps_last_good_marks_stale_once(monkeypatch, no_thread, live_stream, failure):
    _, q = live_stream()
    http = FakeHTTP(_snap([_req()]), failure, failure, _snap([_req()]))
    monkeypatch.setattr(pp.urllib.request, "urlopen", http)
    pp.register("s1", "sess1", ORIGIN, SECRET, config_data=CFG)
    pp._poll_once(_runs(), 5)
    assert _drain(q)[0][1]["stale"] is False
    pp._poll_once(_runs(), 5)
    (item,) = _drain(q)
    assert item[1]["stale"] is True and item[1]["requests"] == [_req()]
    pp._poll_once(_runs(), 5)
    assert _drain(q) == []  # stale re-emitted only once
    pp._poll_once(_runs(), 5)
    assert _drain(q)[0][1]["stale"] is False  # recovered


def test_wedged_engine_marks_stale(monkeypatch, no_thread, live_stream):
    _, q = live_stream()
    monkeypatch.setattr(pp.urllib.request, "urlopen", FakeHTTP(_snap([_req()], running=1, age=30)))
    pp.register("s1", "sess1", ORIGIN, SECRET, config_data=CFG)
    pp._poll_once(_runs(), 5)
    assert _drain(q)[0][1]["stale"] is True


def test_idle_engine_age_is_not_stale():
    assert pp._engine_wedged(_snap([], running=0, waiting=0, age=2625.7)) is False
    assert pp._engine_wedged(_snap([], running=0, waiting=1, age=20)) is True
    assert pp._engine_wedged(_snap([], running=1, age=None)) is False


def test_two_404s_disable_then_reset_on_new_poller(monkeypatch, no_thread, live_stream, caplog):
    live_stream()
    http = FakeHTTP(404)
    monkeypatch.setattr(pp.urllib.request, "urlopen", http)
    pp.register("s1", "sess1", ORIGIN, SECRET, config_data=CFG)
    with caplog.at_level(logging.INFO, logger=pp.__name__):
        for _ in range(4):
            pp._poll_once(_runs(), 5)
    assert len(http.calls) == 2
    assert len([r for r in caplog.records if r.levelno == logging.INFO]) == 1
    # Poller restart (all runs settled, a new run registers) forgets the verdict.
    pp.unregister("s1")
    with pp._LOCK:
        pp._thread = None
    pp.register("s1", "sess1", ORIGIN, SECRET, config_data=CFG)
    pp._poll_once(_runs(), 5)
    assert len(http.calls) == 3


def test_single_404_does_not_disable(monkeypatch, no_thread, live_stream):
    live_stream()
    http = FakeHTTP(404, _snap([]), 404, _snap([]))
    monkeypatch.setattr(pp.urllib.request, "urlopen", http)
    pp.register("s1", "sess1", ORIGIN, SECRET, config_data=CFG)
    for _ in range(4):
        pp._poll_once(_runs(), 5)
    assert len(http.calls) == 4


# ── delivery contract ───────────────────────────────────────────────────────

def test_no_subscriber_never_touches_offline_buffer(monkeypatch, no_thread, live_stream):
    ch, _ = live_stream(subscribe=False)
    ch.put_nowait(("token", {"t": "x"}, "run:1"))
    before = ch.diagnostic_snapshot()
    monkeypatch.setattr(pp.urllib.request, "urlopen", FakeHTTP(_snap([_req()])))
    pp.register("s1", "sess1", ORIGIN, SECRET, config_data=CFG)
    pp._poll_once(_runs(), 5)
    assert list(ch._offline_buffer) == [("token", {"t": "x"}, "run:1")]
    assert ch.diagnostic_snapshot() == before


def test_owner_mismatch_or_missing_stream_is_dropped(monkeypatch, no_thread, live_stream):
    _, q = live_stream("s1", "someone-else")
    monkeypatch.setattr(pp.urllib.request, "urlopen", FakeHTTP(_snap([_req()])))
    pp.register("s1", "sess1", ORIGIN, SECRET, config_data=CFG)
    pp.register("gone", "sess9", ORIGIN, SECRET, config_data=CFG)
    pp._poll_once(_runs(), 5)
    assert _drain(q) == []


def test_emission_holds_no_registry_lock(monkeypatch, no_thread):
    seen = []

    class Probe:
        def put_if_subscribed(self, item):
            # Each lock must be free: acquire non-blocking and release.
            for name, lock in (("streams", config.STREAMS_LOCK),
                               ("active", config.ACTIVE_RUNS_LOCK),
                               ("module", pp._LOCK)):
                got = lock.acquire(blocking=False)
                seen.append((name, got))
                if got:
                    lock.release()
            return True

    monkeypatch.setitem(config.STREAMS, "s1", Probe())
    monkeypatch.setitem(config.ACTIVE_RUNS, "s1", {"session_id": "sess1"})
    monkeypatch.setattr(pp.urllib.request, "urlopen", FakeHTTP(_snap([_req()])))
    pp.register("s1", "sess1", ORIGIN, SECRET, config_data=CFG)
    pp._poll_once(_runs(), 5)
    assert seen and all(got for _, got in seen), seen


def test_not_journaled_and_relay_dedup_bypassed(monkeypatch, no_thread, live_stream):
    """A None event id: no journal append path, and the SSE relay writes it
    with plain _sse instead of inheriting STREAM_LAST_EVENT_ID (which the
    relay may already have sent and would skip as a duplicate)."""
    _, q = live_stream()
    monkeypatch.setattr(pp.urllib.request, "urlopen", FakeHTTP(_snap([_req()])))
    pp.register("s1", "sess1", ORIGIN, SECRET, config_data=CFG)
    pp._poll_once(_runs(), 5)
    (item,) = _drain(q)
    assert len(item) == 3 and item[2] is None
    src = (REPO_ROOT / "api" / "prefill_progress.py").read_text()
    assert "append_sse_event" not in src and "run_journal" not in src


def test_api_key_never_logged_or_emitted(monkeypatch, no_thread, live_stream, caplog):
    _, q = live_stream()
    http = FakeHTTP(_snap([_req()]), 401, TimeoutError("x"), b"bad", 404, 404)
    monkeypatch.setattr(pp.urllib.request, "urlopen", http)
    pp.register("s1", "sess1", ORIGIN, SECRET, config_data=CFG)
    with caplog.at_level(logging.DEBUG, logger=pp.__name__):
        for _ in range(6):
            pp._poll_once(_runs(), 5)
    assert SECRET not in caplog.text
    assert all(SECRET not in json.dumps(item[1]) for item in _drain(q))


# ── 11. worker wiring ───────────────────────────────────────────────────────

def test_streaming_worker_registers_and_unregisters_in_finally():
    src = (REPO_ROOT / "api" / "streaming.py").read_text()
    start = src.index("def _run_agent_streaming(")
    body = src[start:src.index("\ndef ", start + 1)]
    regs = [m.start() for m in re.finditer(r"_prefill_progress\.register\(stream_id, session_id, resolved_base_url, resolved_api_key\)", body)]
    assert len(regs) == 2  # initial resolve + provider-switch rebuild
    unreg = body.index("_prefill_progress.unregister(stream_id)")
    # Unregister sits in the worker's outermost finally (covers success,
    # error, cancel and self-heal), before STREAMS teardown, outside its lock.
    outer_finally = body.rindex("\n    finally:\n", 0, unreg)
    assert all(r < outer_finally for r in regs)
    teardown = body.index("with STREAMS_LOCK:\n            STREAMS.pop(stream_id, None)", unreg)
    assert unreg < teardown


# ── adaptive cadence: admission shows within ~fast_poll_seconds ─────────────

def test_settings_fast_poll_defaults_and_clamps():
    base = {"enabled_base_urls": [ORIGIN]}
    assert pp._settings({"webui": {"prefill_progress": base}})[1:] == (5.0, 1.0)
    cfg = {"webui": {"prefill_progress": dict(base, poll_seconds=3, fast_poll_seconds=0.1)}}
    assert pp._settings(cfg)[1:] == (3.0, 0.5)
    cfg = {"webui": {"prefill_progress": dict(base, poll_seconds=3, fast_poll_seconds=10)}}
    assert pp._settings(cfg)[1:] == (3.0, 3.0)  # never slower than poll_seconds
    cfg = {"webui": {"prefill_progress": dict(base, fast_poll_seconds="x")}}
    assert pp._settings(cfg)[1:] == (5.0, 1.0)


@pytest.mark.parametrize("live,fast", [
    ([], True),  # call not arrived yet / between tool iterations
    ([_req(phase="queued", admitted=False, prefill_eta_s=None)], True),
    ([_req(phase="restoring", restoring_kv=True, prefill_eta_s=None)], True),
    ([_req(prefill_eta_s=None)], True),  # admitted, no rate sample yet
    ([_req(phase="mystery")], True),
    ([_req()], False),  # steady prefill with ETA
    ([_req(phase="decode", remaining_prompt_tokens=0, prefill_eta_s=None)], False),
    ([_req(), _req(phase="queued", admitted=False)], True),  # any waiting request
])
def test_wants_fast_poll(live, fast):
    assert pp._wants_fast_poll(live) is fast


def test_poll_once_reports_cadence_and_errors_back_off(monkeypatch, no_thread, live_stream):
    live_stream()
    http = FakeHTTP(
        _snap([_req(phase="queued", admitted=False, prefill_eta_s=None)]),
        _snap([_req()]),
        500,
        404,
    )
    monkeypatch.setattr(pp.urllib.request, "urlopen", http)
    pp.register("s1", "sess1", ORIGIN, SECRET, config_data=CFG)
    assert pp._poll_once(_runs(), 5) is True
    assert pp._poll_once(_runs(), 5) is False
    assert pp._poll_once(_runs(), 5) is False  # failures never hammer
    assert pp._poll_once(_runs(), 5) is False


def test_queued_then_admitted_frames(monkeypatch, no_thread, live_stream):
    """Each fast poll emits, so the queued -> admitted flip reaches the tab
    on the first poll that sees it."""
    _, q = live_stream()
    http = FakeHTTP(
        _snap([_req(phase="queued", admitted=False, computed_tokens=0, prefill_eta_s=None)], running=0, waiting=1),
        _snap([_req(phase="restoring", restoring_kv=True, prefill_eta_s=None)]),
        _snap([_req(prefill_eta_s=None)]),
    )
    monkeypatch.setattr(pp.urllib.request, "urlopen", http)
    pp.register("s1", "sess1", ORIGIN, SECRET, config_data=CFG)
    phases = []
    for _ in range(3):
        pp._poll_once(_runs(), 5)
        (item,) = _drain(q)
        r = item[1]["requests"][0]
        phases.append((r["phase"], r["admitted"]))
    assert phases == [("queued", False), ("restoring", True), ("prefill", True)]


def test_loop_uses_fast_interval_while_waiting(monkeypatch):
    waits = []
    real_wait = pp._wake.wait

    def spy(timeout=None):
        waits.append(timeout)
        if len(waits) >= 3:
            pp.unregister("s1")
        return real_wait(0.01)

    monkeypatch.setattr(pp._wake, "wait", spy)
    monkeypatch.setattr(pp.urllib.request, "urlopen", FakeHTTP(
        _snap([]), _snap([_req()]), _snap([_req(phase="queued", admitted=False)]),
    ))
    cfg = {"webui": {"prefill_progress": {"enabled_base_urls": [ORIGIN], "poll_seconds": 4, "fast_poll_seconds": 0.75}}}
    pp.register("s1", "sess1", ORIGIN, SECRET, config_data=cfg)
    t = pp._thread
    t.join(timeout=5)
    assert waits[:3] == [0.75, 4.0, 0.75]
