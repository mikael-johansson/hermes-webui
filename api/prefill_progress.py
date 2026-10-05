"""Poll an opt-in vLLM ``/v1/prefill_progress`` endpoint for live runs.

See ``PREFILL_PROGRESS_WEBUI_SPEC.md``. A run whose resolved provider
``base_url`` is listed under ``webui.prefill_progress.enabled_base_urls``
registers here for the lifetime of its worker. While at least one run is
registered, a single daemon thread GETs the endpoint once per distinct server
origin every ``poll_seconds`` and pushes an ephemeral ``prefill_progress`` SSE
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
  the poller.
"""
from __future__ import annotations

import json
import logging
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
MAX_POLL_SECONDS = 60.0
# A snapshot with live requests whose engine stats are older than this is a
# wedged engine, not progress. While idle the age grows without bound and
# means nothing (no engine steps -> no stats).
ENGINE_STALE_AGE_S = 15.0
_MAX_BODY_BYTES = 1 << 20
_DISABLE_AFTER_404S = 2

_LOCK = threading.Lock()
# stream_id -> {"session_id", "origin", "api_key"}
_RUNS: dict[str, dict] = {}
# origin -> {"consecutive_404", "disabled", "last", "auth_logged"}
_ORIGINS: dict[str, dict] = {}
# stream_id -> {"had_live", "stale_sent"}
_STREAM_STATE: dict[str, dict] = {}
_thread: Optional[threading.Thread] = None
_wake = threading.Event()
_poll_seconds = DEFAULT_POLL_SECONDS


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


def _settings(config_data: Optional[dict] = None) -> tuple[set[str], float]:
    """Return (enabled origins, poll seconds) from ``webui.prefill_progress``."""
    if config_data is None:
        try:
            from api.config import get_config

            config_data = get_config()
        except Exception:
            logger.debug("prefill_progress: config read failed", exc_info=True)
            return set(), DEFAULT_POLL_SECONDS
    webui_cfg = config_data.get("webui") if isinstance(config_data, dict) else None
    block = webui_cfg.get("prefill_progress") if isinstance(webui_cfg, dict) else None
    if not isinstance(block, dict):
        return set(), DEFAULT_POLL_SECONDS
    urls = block.get("enabled_base_urls")
    origins = set()
    if isinstance(urls, list):
        for url in urls:
            origin = normalize_origin(url)
            if origin:
                origins.add(origin)
    try:
        poll = float(block.get("poll_seconds", DEFAULT_POLL_SECONDS))
    except (TypeError, ValueError):
        poll = DEFAULT_POLL_SECONDS
    if poll != poll:  # NaN
        poll = DEFAULT_POLL_SECONDS
    poll = min(MAX_POLL_SECONDS, max(MIN_POLL_SECONDS, poll))
    return origins, poll


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
    enabled, poll = _settings(config_data)
    origin = normalize_origin(base_url)
    qualifies = bool(origin and origin in enabled and isinstance(api_key, str) and api_key.strip())
    global _thread, _poll_seconds
    start = None
    with _LOCK:
        if not qualifies:
            _drop_locked(stream_id)
            return False
        _RUNS[stream_id] = {
            "session_id": str(session_id),
            "origin": origin,
            "api_key": api_key.strip(),
        }
        _STREAM_STATE.setdefault(stream_id, {"had_live": False, "stale_sent": False})
        if _thread is None:
            # A fresh poller forgets per-origin verdicts, so enabling the
            # endpoint on the server is picked up without a WebUI restart.
            _ORIGINS.clear()
            _poll_seconds = poll
            _wake.clear()
            start = threading.Thread(
                target=_poll_loop, name="prefill-progress-poller", daemon=True
            )
            _thread = start
    if start is not None:
        start.start()
    return True


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
            interval = _poll_seconds
        try:
            _poll_once(runs, interval)
        except Exception:
            logger.debug("prefill_progress: poll iteration failed", exc_info=True)
        _wake.wait(interval)
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


def _poll_once(runs: dict, interval: float) -> None:
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


def _deliver(stream_id: str, session_id: str, payload: dict) -> bool:
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
        return bool(put((EVENT_NAME, payload, None)))
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
