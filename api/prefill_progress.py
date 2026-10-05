"""Poll an opt-in vLLM ``/v1/prefill_progress`` endpoint for live runs.

See ``PREFILL_PROGRESS_WEBUI_SPEC.md``. A run whose resolved provider
``base_url`` is listed under ``webui.prefill_progress.enabled_base_urls``
registers here for the lifetime of its worker. While at least one run is
registered, a single daemon thread GETs the endpoint once per distinct server
origin (every ``fast_poll_seconds`` while waiting for arrival/admission/
restore/first rate sample, else every ``poll_seconds``) and pushes an
ephemeral ``prefill_progress`` SSE
frame to each registered stream.

Ownership and invariants:

- State layer: this module's ``_RUNS`` / ``_ORIGINS`` / ``_STREAM_STATE``,
  guarded by ``_LOCK``. ``_LOCK`` is a leaf lock: it is never held while
  taking ``STREAMS_LOCK`` / ``ACTIVE_RUNS_LOCK``, during network I/O, or while
  putting to a channel.
- Frames are NOT journaled and never buffered for disconnected tabs: they go
  through ``StreamChannel.put_if_subscribed`` as ``(event, data, None)``.
  A ``None`` event id makes the SSE relay write them without an ``id:`` and
  without the dedup check that a 2-tuple would inherit.
- The api key is only ever placed in the outgoing ``Authorization`` header;
  it is never logged, emitted, or persisted.
- Run workers only call ``register`` / ``unregister``; they never wait on
  the poller, except ``collect_call_stats`` at settle, which waits at most
  its ``wait_s`` and only while this run has calls whose stats are pending.

Per-call stats (PP / TG / TTFT / queue wait, ada spec rev 3): hermes-agent
logs one ``API call #N: ... id=<response id>`` INFO record per successful
model call on ``agent.conversation_loop``, tagged with the turn's session id
(``record.session_tag``). ``_CallLogHandler`` queues that response id on the
session's registered run; the poller matches it to the server's finished
entry with the same ``request_id`` (exact, so side calls such as title
generation never attach to the turn), emits ``llm_call_stats``, and the
worker stamps the collected stats onto the turn's assistant messages.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

logger = logging.getLogger(__name__)

EVENT_NAME = "prefill_progress"
DEFAULT_POLL_SECONDS = 5.0
MIN_POLL_SECONDS = 2.0
# Used while the next state change is the interesting one: no live request
# seen yet (the call has not reached the server, or the agent is between
# tool iterations), queued, restoring KV, or prefill before the first rate
# sample. ada refreshes its engine snapshot ~1/s, so admission shows within
# ~2 s. Steady prefill (ETA known) and decode fall back to poll_seconds.
DEFAULT_FAST_POLL_SECONDS = 1.0
MIN_FAST_POLL_SECONDS = 0.5
_SETTLED_PHASES = ("decode",)
MAX_POLL_SECONDS = 60.0
# A snapshot with live requests whose engine stats are older than this is a
# wedged engine, not progress. While idle the age grows without bound and
# means nothing (no engine steps -> no stats).
ENGINE_STALE_AGE_S = 15.0
_MAX_BODY_BYTES = 1 << 20
_DISABLE_AFTER_404S = 2
CALL_STATS_EVENT = "llm_call_stats"
# Give up on a captured call whose finished entry never shows up (server
# without rev 3 stats, a fallback provider, or an entry past its TTL).
CALL_STATS_TIMEOUT_S = 30.0
_CALL_LOGGER = "agent.conversation_loop"
_CALL_RE = re.compile(r"^API call #(\d+):")
_CALL_ID_RE = re.compile(r"\bid=(\S+)")
_SESSION_TAG_RE = re.compile(r"\[([^\]\s]+)\]")
# Final-stat keys copied from a finished server entry into llm_call_stats.
_CALL_STAT_KEYS = (
    "prompt_tokens", "computed_prompt_tokens", "completion_tokens",
    "pp_tok_s", "tg_tok_s", "ttft_s", "prefill_s", "decode_s", "queue_wait_s",
)

_LOCK = threading.Lock()
# stream_id -> {"session_id", "origin", "api_key"}
_RUNS: dict[str, dict] = {}
# origin -> {"consecutive_404", "disabled", "last", "auth_logged"}
_ORIGINS: dict[str, dict] = {}
# stream_id -> {"had_live", "stale_sent"}
_STREAM_STATE: dict[str, dict] = {}
_thread: Optional[threading.Thread] = None
_wake = threading.Event()
_CALLS_CV = threading.Condition(_LOCK)
_capture_installed = False
_poll_seconds = DEFAULT_POLL_SECONDS
_fast_poll_seconds = DEFAULT_FAST_POLL_SECONDS


def normalize_origin(url: Any) -> str:
    """Return ``scheme://host[:port]`` for matching and polling, or ``''``.

    Strips whitespace, trailing slashes and one trailing ``/v1``; lowercases
    scheme and host. Any other path component makes the URL non-matching,
    so ``http://h:1/other`` never aliases ``http://h:1``.
    """
    if not isinstance(url, str):
        return ""
    raw = url.strip().rstrip("/")
    if raw.lower().endswith("/v1"):
        raw = raw[:-3].rstrip("/")
    try:
        parts = urllib.parse.urlsplit(raw)
    except ValueError:
        return ""
    if parts.scheme.lower() not in ("http", "https") or not parts.hostname:
        return ""
    if parts.path or parts.query or parts.fragment or parts.username or parts.password:
        return ""
    host = parts.hostname.lower()
    if ":" in host:  # IPv6 literal
        host = f"[{host}]"
    try:
        port = parts.port
    except ValueError:
        return ""
    netloc = f"{host}:{port}" if port else host
    return f"{parts.scheme.lower()}://{netloc}"


def _settings(config_data: Optional[dict] = None) -> tuple[set[str], float, float]:
    """Return (enabled origins, poll seconds, fast poll seconds) from
    ``webui.prefill_progress``."""
    defaults = (set(), DEFAULT_POLL_SECONDS, DEFAULT_FAST_POLL_SECONDS)
    if config_data is None:
        try:
            from api.config import get_config

            config_data = get_config()
        except Exception:
            logger.debug("prefill_progress: config read failed", exc_info=True)
            return defaults
    webui_cfg = config_data.get("webui") if isinstance(config_data, dict) else None
    block = webui_cfg.get("prefill_progress") if isinstance(webui_cfg, dict) else None
    if not isinstance(block, dict):
        return defaults
    urls = block.get("enabled_base_urls")
    origins = set()
    if isinstance(urls, list):
        for url in urls:
            origin = normalize_origin(url)
            if origin:
                origins.add(origin)
    poll = _clamped(block.get("poll_seconds"), DEFAULT_POLL_SECONDS, MIN_POLL_SECONDS, MAX_POLL_SECONDS)
    fast = _clamped(block.get("fast_poll_seconds"), DEFAULT_FAST_POLL_SECONDS, MIN_FAST_POLL_SECONDS, poll)
    return origins, poll, fast


def _clamped(raw: Any, default: float, low: float, high: float) -> float:
    try:
        value = float(default if raw is None else raw)
    except (TypeError, ValueError):
        value = default
    if value != value:  # NaN
        value = default
    return min(high, max(low, value))


def register(
    stream_id: str,
    session_id: str,
    base_url: Any,
    api_key: Any,
    *,
    config_data: Optional[dict] = None,
) -> bool:
    """Register a live run if its provider origin is allowlisted.

    Idempotent; a second call for the same stream (provider switch mid-run)
    replaces the entry, and drops it if the new base_url does not qualify.
    Returns whether the stream is registered afterwards.
    """
    if not stream_id or not session_id:
        return False
    enabled, poll, fast = _settings(config_data)
    origin = normalize_origin(base_url)
    qualifies = bool(origin and origin in enabled and isinstance(api_key, str) and api_key.strip())
    global _thread, _poll_seconds, _fast_poll_seconds
    start = None
    with _LOCK:
        if not qualifies:
            _drop_locked(stream_id)
            return False
        prior = _RUNS.get(stream_id)
        _RUNS[stream_id] = {
            "session_id": str(session_id),
            "origin": origin,
            "api_key": api_key.strip(),
            # Captured model calls, in order (see _CallLogHandler). Kept
            # across a provider-switch re-register of the same stream.
            "calls": prior["calls"] if prior and prior.get("session_id") == str(session_id) else [],
        }
        _STREAM_STATE.setdefault(stream_id, {"had_live": False, "stale_sent": False})
        if _thread is None:
            # A fresh poller forgets per-origin verdicts, so enabling the
            # endpoint on the server is picked up without a WebUI restart.
            _ORIGINS.clear()
            _poll_seconds = poll
            _fast_poll_seconds = fast
            _wake.clear()
            start = threading.Thread(
                target=_poll_loop, name="prefill-progress-poller", daemon=True
            )
            _thread = start
    _install_call_capture()
    if start is not None:
        start.start()
    return True


class _CallLogHandler(logging.Handler):
    """Queue each agent model call's response id on its session's run.

    Runs on the agent's thread inside ``logger.info``: cheap prefix check
    first, never raises, takes only the leaf ``_LOCK``.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.msg
            if not isinstance(msg, str) or not msg.startswith("API call #"):
                return
            text = record.getMessage()
            m_call = _CALL_RE.match(text)
            m_id = _CALL_ID_RE.search(text)
            m_sid = _SESSION_TAG_RE.search(str(getattr(record, "session_tag", "") or ""))
            if not (m_call and m_id and m_sid):
                return
            note_model_call(m_sid.group(1), m_id.group(1), int(m_call.group(1)))
        except Exception:
            pass


def _install_call_capture() -> None:
    global _capture_installed
    with _LOCK:
        if _capture_installed:
            return
        _capture_installed = True
    logging.getLogger(_CALL_LOGGER).addHandler(_CallLogHandler(level=logging.INFO))


def note_model_call(session_id: str, response_id: str, api_call_number: int, *, now: Optional[float] = None) -> bool:
    """Queue one completed model call on every registered run of
    ``session_id`` (normally exactly one). Returns whether any run took it."""
    if not session_id or not response_id:
        return False
    now = time.monotonic() if now is None else now
    took = False
    with _LOCK:
        for run in _RUNS.values():
            if run.get("session_id") != session_id:
                continue
            calls = run.setdefault("calls", [])
            if any(c["request_id"] == response_id for c in calls):
                continue
            calls.append({
                "call": len(calls) + 1,
                "api_call_number": api_call_number,
                "request_id": response_id,
                "captured_at": now,
                "stats": None,
                "done": False,
            })
            took = True
        if took:
            _wake.set()  # fetch the finished entry now, not a poll later
    return took


def _pending_calls_locked(run: dict, now: float) -> list:
    pending = []
    for c in run.get("calls") or ():
        if c["done"]:
            continue
        if now - c["captured_at"] > CALL_STATS_TIMEOUT_S:
            c["done"] = True
            continue
        pending.append(c)
    return pending


def collect_call_stats(stream_id: str, wait_s: float = 0.5) -> list:
    """Return this run's captured calls (in order) with whatever stats have
    arrived, waiting at most ``wait_s`` for pending ones. Each item:
    ``{"call", "request_id", "stats"}`` (stats None when unavailable)."""
    deadline = time.monotonic() + max(0.0, wait_s)
    with _CALLS_CV:
        while True:
            run = _RUNS.get(stream_id)
            if run is None:
                return []
            if not _pending_calls_locked(run, time.monotonic()):
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            _wake.set()
            _CALLS_CV.wait(remaining)
        return [
            {"call": c["call"], "request_id": c["request_id"], "stats": dict(c["stats"]) if c["stats"] else None}
            for c in run.get("calls") or ()
        ]


def unregister(stream_id: str) -> None:
    """Drop a run. Idempotent; wakes the poller so it can exit when idle."""
    if not stream_id:
        return
    with _LOCK:
        _drop_locked(stream_id)


def _drop_locked(stream_id: str) -> None:
    _RUNS.pop(stream_id, None)
    _STREAM_STATE.pop(stream_id, None)
    if not _RUNS:
        _wake.set()


def _poll_loop() -> None:
    global _thread
    me = threading.current_thread()
    while True:
        with _LOCK:
            if _thread is not me:
                return
            if not _RUNS:
                # Exit decision and the register()'s start decision share
                # _LOCK, so a racing register always sees either a live
                # poller with its run included or _thread None.
                _thread = None
                return
            runs = {sid: dict(run) for sid, run in _RUNS.items()}
            interval, fast_interval = _poll_seconds, _fast_poll_seconds
        fast = False
        try:
            fast = _poll_once(runs, interval)
        except Exception:
            logger.debug("prefill_progress: poll iteration failed", exc_info=True)
        _wake.wait(fast_interval if fast else interval)
        _wake.clear()


def _fetch(origin: str, api_key: str, timeout: float) -> tuple[Optional[int], Any]:
    """GET the endpoint. Returns (http status or None on transport error, parsed JSON or None)."""
    req = urllib.request.Request(
        f"{origin}/v1/prefill_progress",
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = getattr(resp, "status", None) or resp.getcode()
            body = resp.read(_MAX_BODY_BYTES)
    except urllib.error.HTTPError as exc:
        return exc.code, None
    except Exception as exc:
        # Never log the request object (it carries the Authorization header).
        logger.debug("prefill_progress: fetch %s failed: %s", origin, type(exc).__name__)
        return None, None
    try:
        return status, json.loads(body.decode("utf-8"))
    except Exception:
        logger.debug("prefill_progress: bad JSON from %s", origin)
        return status, None


def _live_requests(snapshot: dict) -> list:
    reqs = snapshot.get("requests")
    if not isinstance(reqs, list):
        return []
    return [r for r in reqs if isinstance(r, dict) and r.get("phase") != "finished"]


def _engine_wedged(snapshot: dict) -> bool:
    try:
        pending = int(snapshot.get("running") or 0) + int(snapshot.get("waiting") or 0)
    except (TypeError, ValueError):
        pending = 0
    age = snapshot.get("engine_ts_age_s")
    return (
        pending > 0
        and isinstance(age, (int, float))
        and not isinstance(age, bool)
        and age > ENGINE_STALE_AGE_S
    )


def _wants_fast_poll(live: list) -> bool:
    """True while the next state change (arrival, admission, restore done,
    first rate sample) is what the user is waiting to see."""
    if not live:
        return True
    for req in live:
        phase = req.get("phase")
        if phase == "decode":
            # Live generated-token count / TG rate: keep it moving.
            return True
        if phase in _SETTLED_PHASES:
            continue
        eta = req.get("prefill_eta_s")
        if phase == "prefill" and isinstance(eta, (int, float)) and not isinstance(eta, bool) and eta > 0:
            continue
        return True
    return False


def _poll_once(runs: dict, interval: float) -> bool:
    """Poll each origin once and emit. Returns whether any origin wants the
    fast cadence (failed or disabled origins never do, so errors back off)."""
    fast = False
    by_origin: dict[str, list[tuple[str, dict]]] = {}
    for stream_id, run in runs.items():
        by_origin.setdefault(run["origin"], []).append((stream_id, run))
    timeout = max(1.0, min(4.0, interval - 1.0))
    for origin, members in by_origin.items():
        with _LOCK:
            state = _ORIGINS.setdefault(
                origin,
                {"consecutive_404": 0, "disabled": False, "last": None, "auth_logged": False},
            )
            if state["disabled"]:
                continue
        status, data = _fetch(origin, members[0][1]["api_key"], timeout)
        ok = status == 200 and isinstance(data, dict)
        frames: list[tuple[str, str, dict]] = []
        call_frames: list[tuple[str, str, dict]] = []
        with _LOCK:
            if status == 404:
                state["consecutive_404"] += 1
                if state["consecutive_404"] >= _DISABLE_AFTER_404S:
                    state["disabled"] = True
                    logger.info(
                        "prefill_progress: %s/v1/prefill_progress not found; "
                        "polling disabled until the next run starts",
                        origin,
                    )
            else:
                state["consecutive_404"] = 0
            if status in (401, 403) and not state["auth_logged"]:
                state["auth_logged"] = True
                logger.info("prefill_progress: %s rejected the provider key (HTTP %s)", origin, status)
            if ok:
                payload = {
                    "requests": _live_requests(data),
                    "running": data.get("running"),
                    "waiting": data.get("waiting"),
                    "stale": _engine_wedged(data),
                    "ts": time.time(),
                }
                state["last"] = payload
                fast = fast or _wants_fast_poll(payload["requests"])
            if ok:
                by_id = {
                    r.get("request_id"): r for r in (data.get("requests") or [])
                    if isinstance(r, dict) and r.get("phase") == "finished"
                    and r.get("completion_tokens") is not None
                }
            now_mono = time.monotonic()
            for stream_id, run in members:
                live_run = _RUNS.get(stream_id)
                if live_run is None:
                    continue
                pending = _pending_calls_locked(live_run, now_mono)
                if pending:
                    fast = True
                for c in pending if ok else ():
                    entry = by_id.get(c["request_id"])
                    if entry is None:
                        continue
                    c["stats"] = {k: entry.get(k) for k in _CALL_STAT_KEYS}
                    c["done"] = True
                    call_frames.append((stream_id, run["session_id"], {
                        "call": c["call"],
                        "request_id": c["request_id"],
                        "stats": dict(c["stats"]),
                    }))
            _CALLS_CV.notify_all()
            for stream_id, run in members:
                sstate = _STREAM_STATE.get(stream_id)
                if sstate is None or stream_id not in _RUNS:
                    continue  # unregistered while we were fetching
                frame = None
                if ok:
                    sstate["stale_sent"] = False
                    if payload["requests"]:
                        sstate["had_live"] = True
                        frame = payload
                    elif sstate["had_live"]:
                        # One empty frame lets the browser drop the chip.
                        sstate["had_live"] = False
                        frame = payload
                elif state["last"] and state["last"]["requests"] and not sstate["stale_sent"]:
                    sstate["stale_sent"] = True
                    frame = dict(state["last"], stale=True)
                if frame is not None:
                    frames.append((stream_id, run["session_id"], frame))
        for stream_id, session_id, frame in frames:
            _deliver(stream_id, session_id, dict(frame, stream_id=stream_id, session_id=session_id))
        for stream_id, session_id, frame in call_frames:
            _deliver(stream_id, session_id, dict(frame, stream_id=stream_id, session_id=session_id),
                     event=CALL_STATS_EVENT)
    return fast


def _deliver(stream_id: str, session_id: str, payload: dict, event: str = EVENT_NAME) -> bool:
    """Push one ephemeral frame to a live stream's connected tabs.

    Snapshots ACTIVE_RUNS and STREAMS under their own locks (never nested,
    never with ``_LOCK``), then puts outside every lock.
    """
    from api import config as _cfg

    with _cfg.ACTIVE_RUNS_LOCK:
        meta = _cfg.ACTIVE_RUNS.get(stream_id)
        owner = meta.get("session_id") if isinstance(meta, dict) else None
    if owner != session_id:
        return False
    with _cfg.STREAMS_LOCK:
        channel = _cfg.STREAMS.get(stream_id)
    put = getattr(channel, "put_if_subscribed", None)
    if not callable(put):
        return False
    try:
        return bool(put((event, payload, None)))
    except Exception:
        logger.debug("prefill_progress: emit failed for stream %s", stream_id, exc_info=True)
        return False


def _reset_for_tests() -> None:
    """Stop the poller and clear all state (tests only)."""
    global _thread
    with _LOCK:
        _RUNS.clear()
        _ORIGINS.clear()
        _STREAM_STATE.clear()
        thread = _thread
        _thread = None
        _wake.set()
    if thread is not None and thread is not threading.current_thread():
        thread.join(timeout=5)
    _wake.clear()


def stamp_call_stats(messages: Any, calls: list) -> int:
    """Stamp ``_callStats`` onto this turn's assistant messages, one per
    captured model call, in order. The turn is everything after the last
    user message. Only when the counts match exactly (one assistant message
    per model call); otherwise nothing is stamped and the stats stay
    live-only, rather than attaching numbers to the wrong call. Returns the
    number of messages stamped."""
    if not calls or not isinstance(messages, list):
        return 0
    turn: list = []
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "user":
            break
        if role == "assistant":
            turn.append(msg)
    turn.reverse()
    if len(turn) != len(calls):
        logger.debug(
            "prefill_progress: %d model calls vs %d assistant messages; call stats not persisted",
            len(calls), len(turn),
        )
        return 0
    stamped = 0
    for msg, call in zip(turn, calls, strict=True):
        if call.get("stats"):
            msg["_callStats"] = dict(call["stats"], request_id=call.get("request_id"))
            stamped += 1
    return stamped
