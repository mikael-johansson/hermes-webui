# Spec: prefill-progress visibility in Hermes WebUI

Companion to the ada vLLM-fork spec `PREFILL_PROGRESS_SPEC.md` (rev 2,
implemented in `82bf273`, `~/git/vllm/qwen38-27b-rtx3090` on ada.vpn), which provides
`GET /v1/prefill_progress` (opt-in env `PREFILL_PROGRESS=1` on the server,
auth = the same API key the chat endpoint uses).

**Status (verified 2026-10-05):** the ada endpoint (rev 2, `82bf273`) is live — authenticated
`GET http://ada.vpn:18020/v1/prefill_progress` → 200, while an unknown `/v1/*`
path → 404 (so the 200 is real, not middleware). Unauthenticated → 401. The
WebUI side is entirely unimplemented. The poller must still tolerate the
endpoint's absence (§1.5) because ada can be restarted without the env flag.

User priority (verbatim intent): seeing that a request is *admitted* and how
large the processing is. Granularity of the progress % is explicitly
secondary.

---

## 0. Upstream contract (ada spec rev 2, `82bf273`) and what it implies

Spec of record: ada `~/git/vllm/qwen38-27b-rtx3090/PREFILL_PROGRESS_SPEC.md`
(rev 2; implementation commit `82bf273`). Response (all keys always present):

```json
{ "ts": 1760000000.0, "engine_ts_age_s": 0.4, "running": 1, "waiting": 2, "kv_cache_pct": 4.9,
  "requests": [{ "request_id": "chatcmpl-abc123", "phase": "prefill", "admitted": true,
    "arrived_ms": 1234, "admitted_ms": 456, "prompt_tokens": 180412, "computed_tokens": 41200,
    "cached_tokens": 138000, "remaining_prompt_tokens": 139212, "prefill_rate_tok_s": 1850.0,
    "prefill_eta_s": 75.2, "restoring_kv": false, "restore_ms": null, "preemptions": 0 }] }
```

Contract points this spec relies on:

- `phase` ∈ `queued | restoring | prefill | decode | finished`. `queued` also
  covers preempted requests (`admitted` flips back to false, `preemptions`
  increments). `restoring_kv == (phase == "restoring")`. Treat an unknown
  phase like `prefill` if `remaining_prompt_tokens > 0`, else as `Running`.
- `finished` entries stay for `PREFILL_PROGRESS_FINISHED_TTL_S` (default 60
  s). The finish is detected lazily, either on the next engine snapshot or
  on the next GET, so while the engine is idle an entry is marked finished on
  our first poll after it ends and stays for 60 s after that. Observed live: one
  `finished` entry with `running == 0`. Every consumer filters these out first.
- `computed_tokens` **includes** prefix-cache / offload-restore credit and
  generated tokens. `remaining_prompt_tokens = max(0, prompt - computed)` is
  the real work left, already computed upstream. Use it as is; do not
  re-derive. `computed > prompt` is normal in `decode` (observed 129924 >
  129048).
- `cached_tokens` is the admission-time hit (informational, may be null).
  Observed 127296 of 129048: on a cached second turn the actual prefill is
  small, so the label shows both the work and the context size (§2.3).
- `prefill_rate_tok_s` / `prefill_eta_s` are smoothed upstream, null outside
  `prefill` or before a second sample (rate sampled every ~5 s). Show the ETA
  when present; never smooth again in the WebUI.
- `arrived_ms` / `admitted_ms` are *durations* (ms since), not timestamps.
- `engine_ts_age_s` = age of the last engine stats batch. It grows without
  bound while the engine is idle (no steps → no stats), so it only means
  "wedged" when live requests exist: treat a snapshot as stale when
  `running + waiting > 0` and `engine_ts_age_s > 15`.
- `request_id` is the external `chatcmpl-…` id the client sees in its stream,
  so it is known only after the first streamed chunk, i.e. after prefill. It
  cannot identify "this run's" request during the phase we care about.
- Single API server / single engine only (ada spec §6). Fine for ada.
- Auth: the existing `/v1` middleware. 401 without a key; the route 404s when
  `PREFILL_PROGRESS` is unset. A plain GET with only `Authorization` works.

Pass the per-request objects through to the browser verbatim (no field
renaming; the contract belongs to the ada spec). Filtering of finished
entries happens server-side before emit (§1.4) and again client-side
(defense in depth).

## 1. Server side (`api/prefill_progress.py` new, hooks in `api/streaming.py`)

Keep the poller in its own small module (same shape as `api/todo_state.py`)
so `api/streaming.py` only gains register/unregister calls.

### 1.1 Config

Lives under the existing `webui:` block of the Hermes config (read the same
way as `session_save_mode` at `api/config.py:542` — via `get_config()`, never
by re-parsing YAML):

```yaml
webui:
  prefill_progress:
    poll_seconds: 5            # optional, default 5, clamp to [2, 60]
    enabled_base_urls:
      - http://ada.vpn:18020
```

Default: key absent or list empty → no registration, no thread, zero
behavior change. No settings-panel UI; code default only. Rollback story:
remove the block.

**URL normalization** (single helper, used for both matching and polling):
strip whitespace, trailing `/`, and one trailing `/v1`; lowercase scheme and
host. The run's resolved `base_url` (typically `http://ada.vpn:18020/v1`) and
each allowlist entry are normalized and compared for equality. The poll URL is
always `<normalized origin>/v1/prefill_progress`.

### 1.2 Which runs register

Register in `_run_agent_streaming` once `resolved_base_url` /
`resolved_api_key` are final and fed into `_agent_kwargs`
(`api/streaming.py` ~12445). The provider-switch rebuild path that reassigns
`resolved_base_url` (~13440) re-registers (update in place, same stream_id).
Use the values the worker already resolved — do not resolve the provider
again.

- Only in-process agent runs register. Gateway-routed runs
  (`api/gateway_chat.py`) never register: the WebUI does not own that
  connection (see §4).
- Non-matching base_url → no registration, so no polls are caused.
- Unregister in the worker's final cleanup, next to
  `unregister_active_run(stream_id)` (`api/streaming.py` ~15266), on every
  exit path (success, error, cancel, self-heal). Unregister is idempotent.
- Known limitation: if a provider fallback moves the run off ada mid-turn,
  the chip may show ada data that is not this run's. Acceptable on a
  single-user box; the browser hides the chip once the snapshot has no live
  requests.

### 1.3 Registry and poller

In `api/prefill_progress.py`:

- `_RUNS: dict[str, dict]` (stream_id → `{session_id, origin, api_key}`),
  `_STATE: dict[str, dict]` (origin → `{last_snapshot, last_ok_at,
  consecutive_404, disabled}`), `_LOCK = threading.Lock()` — a module-local
  leaf lock. It is never held while taking `STREAMS_LOCK` or
  `ACTIVE_RUNS_LOCK`, and never held across network I/O or channel puts.
- One daemon thread, started on the first `register()` while none is
  running; it exits when `_RUNS` is empty at the top of a loop iteration.
  Start/stop decisions are made under `_LOCK` (a thread flag plus
  `threading.Event` for stop/wake) so a register racing the thread's exit
  always ends with exactly one live poller.
- Each tick: snapshot `_RUNS` under `_LOCK`, group by `origin`, and do **one**
  GET per distinct origin (not per run) using any registered run's api_key
  for that origin. `urllib.request`, `Authorization: Bearer <key>`, timeout
  `min(4, poll_seconds - 1)` s. Then sleep `poll_seconds` on the stop event.
- Never log, emit, or persist the api_key.
- Run workers only register/unregister; they never wait on the poller.

### 1.4 Emission

Do **not** use `_sse` (`api/streaming.py:9583`, the HTTP-handler writer on
the reader side) and do **not** use the worker's `put()` closure
(`api/streaming.py:11020` — worker-local and journaled). Use the cross-thread
pattern of `_emit_to_session_streams` (`api/background_process.py:650`):

1. Snapshot `ACTIVE_RUNS` under `ACTIVE_RUNS_LOCK`, release; look up the
   stream's channel under `STREAMS_LOCK`, release (independent, never nested,
   never with the module lock).
2. Outside all locks, if the run's owner session still matches the
   registration: `channel.put_if_subscribed(('prefill_progress', payload, None))`.

`StreamChannel.put_if_subscribed` (new, `api/config.py`) is used instead of
`put_nowait` for two reasons found during implementation:

- With no tab connected, `put_nowait` appends to the capped offline replay
  buffer (drop-oldest). Ephemeral frames every 5 s during a long prefill would
  evict real stream frames and shift `offline_first_event_id`.
  `put_if_subscribed` delivers to live subscribers only and drops the
  ephemeral frame (never a real one) on a full queue.
- The SSE relay (`api/routes.py`, both stream loops) gives a **2-tuple**
  the stream's *last real* event id (`STREAM_LAST_EVENT_ID`). If that id was
  already sent, the frame is skipped as a duplicate. A 3-tuple with a `None`
  id is written with plain `_sse`, without the dedup check.
  (`_emit_to_session_streams` uses 2-tuples and so is not reused.)

Because this bypasses `run_journal.append_sse_event`, the event is
**not journaled** and is never replayed. After a reconnect, the chip comes back on
the next poll (≤ `poll_seconds`).

Payload:

```
event: prefill_progress
data: {"stream_id":"…","session_id":"…","requests":[…live entries verbatim…],
       "running":0,"waiting":1,"stale":false,"ts":1760000000.0}
```

- `requests` = ada's list with `phase == "finished"` entries removed.
- `ts` = WebUI wall clock at successful poll (for staleness; ada's own `ts`
  is not used for it).
- `stale` = true on fetch failure (§1.5) **or** when
  `running + waiting > 0 and engine_ts_age_s > 15` (wedged engine, §0).
- Emit rules per stream: one frame per poll while live requests exist (the
  poll cadence is the rate bound, and it keeps the browser's 15 s freshness
  window alive). When the filtered list becomes empty after having been
  non-empty, emit **once** with `requests: []` (lets the client drop the
  chip), then stay silent while idle.

### 1.5 Errors

- Timeout / DNS / 5xx / 401 / bad JSON / non-dict: log at debug, keep the
  last-good snapshot, re-emit it once with `stale: true` (then not again
  until it recovers), keep polling. 401 additionally logs one info line per
  origin (wrong key is a config problem worth seeing once).
- Two consecutive 404s on an origin → mark it `disabled`, one info log, stop
  polling that origin. **The disable is reset when the poller thread next
  starts** (after all runs settle and a new qualifying run registers), so
  enabling `PREFILL_PROGRESS=1` on ada is picked up without a WebUI restart.

## 2. Client side (`static/messages.js`, `static/ui.js`, `static/i18n.js`)

### 2.1 Target element

The chip goes in the **live run status** line, `#liveRunStatus`
(`_renderLiveRunStatusContent`, `static/ui.js` ~15367), currently
`● mm:ss · N tokens · Running`. That element exists from submit onward and
is re-rendered by the 1 s `_startLiveRunStatusTimer` tick, which is exactly
the window (queued → prefilling, before any output) this feature is for.

Do **not** use the settled-turn footer `_transparentTurnFooterHtml`
(`static/ui.js` ~13081): it renders only in `transparent_stream` mode, only
for turns that already have event rows, and its "Working…" exists for the
post-answer window. Leave it untouched.

`compact_worklog` mode hides `#liveRunStatus` entirely; the chip is not
shown there either (no new surface).

### 2.2 Listener

In `_wireSSE` (`static/messages.js`, next to
`source.addEventListener('todo_state', …)` ~6091), add
`source.addEventListener('prefill_progress', e => {…})`:

- Parse defensively; drop if `d.session_id` ≠ `activeSid` or
  `S.session.session_id` ≠ `activeSid` (same cross-session guards as
  `todo_state`), or if `d.stream_id` ≠ the stream this source is wired to.
- Store `{stream_id, requests, stale, receivedAt: Date.now()}` in one
  module-level slot in `ui.js` (e.g. `_liveRunPrefill`), via a small setter.
  Do not write it to `S.session`, `INFLIGHT`, or localStorage.
- Call `updateLiveRunStatus({sessionId: activeSid})` to re-render now.
- Clear the slot in `hideLiveRunStatus` (`static/ui.js` ~15405) and when a
  new stream starts. No new liveness source: the run-status lifecycle that
  already shows/hides `#liveRunStatus` is the gate.

### 2.3 Rendering

In `_renderLiveRunStatusContent`, replace the trailing `Running` label
(`lf-status`) with the chip when the slot is for the current stream and
`Date.now() - receivedAt ≤ 15000`; otherwise keep `Running` (also the label
for every non-ada run).

Request selection: filter out `phase === "finished"`; pick the entry with the
largest `remaining_prompt_tokens` (ties → latest `arrived_ms`). No
request_id matching (§0). Empty list → `Running`.

Derived values: `P = prompt_tokens`, `R = remaining_prompt_tokens`,
`C = cached_tokens` (may be null), `N = R + max(0, computed_tokens - (C ?? 0))`
clamped to `[R, P]` = new work this request does (prompt minus cache credit);
`D = N - R` = new work done so far.

| `phase` (first match) | label |
|---|---|
| `restoring` | `Restoring KV · {P}` |
| `queued` | `Queued · {P} prompt` |
| `prefill`, `prefill_eta_s` > 0 | `Prefill {D}/{N} · ETA {eta}` |
| `prefill` (no ETA yet) | `Prefill {N}…` |
| `decode`, `finished`, unknown with `R == 0` | `Running` |
| unknown with `R > 0` | `Prefill {N}…` |

- When `C` is non-null and `C > 0`, append ` ({P} ctx)` to the prefill labels so both
  the size of the work and the context show (`Prefill 1.2k/1.7k · ETA 4s (129k ctx)`).
- `preemptions > 0` on a `queued` entry: same `Queued` label (no new key).
- Token numbers: reuse the existing `_fmtTokens` if it gives `180.4k`-style
  output; otherwise add one formatter (1 decimal, k/M).
- ETA: `prefill_eta_s` → `45s` / `1m15s`; never negative, omitted when null.
- `stale: true` (set by the WebUI poller on fetch failure, or when §0's
  `engine_ts_age_s` rule fires): same text, add class `lf-status-stale`
  (dimmed), no ETA.
- The element's text must truncate (`text-overflow: ellipsis` on the label
  span) so it never pushes the timer/tokens off at 360 px.

### 2.4 i18n

`t()` only interpolates numbered placeholders (`{0}`, `{1}` — see
`static/i18n.js` `function t`). New keys, mirrored into every locale block:

- `prefill_queued`: `"Queued · {0} prompt"`
- `prefill_progress`: `"Prefill {0}/{1} · ETA {2}"`
- `prefill_size`: `"Prefill {0}…"`
- `prefill_restoring`: `"Restoring KV · {0}"`
- `prefill_ctx_suffix`: `"({0} ctx)"`

The existing `Running` label in `_renderLiveRunStatusContent` is hard-coded
English. Leave it as is (out of scope); do not convert it in this PR.

### 2.5 Visual

Reuse the existing `live-run-status-dot tool-card-running-dot` pulse. New
CSS limited to the `lf-status-stale` dim variant and label truncation.

## 3. Tests (`./scripts/test.sh`)

Server — `tests/test_prefill_progress.py` (no network; monkeypatch
`urllib.request.urlopen`; fake channels with `put_nowait`):

1. URL normalization: `http://ada.vpn:18020/v1/`, `http://ADA.vpn:18020`,
   and the allowlist entry all match; a different port does not.
2. Empty/absent config → `register()` is a no-op, no thread started.
3. Refcount: thread starts on first register, exits after last unregister;
   a register racing the exit leaves exactly one live poller (barrier-based).
4. One GET per origin with two runs on the same origin; both streams receive
   the event.
5. Happy path: emitted tuple is `('prefill_progress', payload)` on the
   stream's channel; `finished` entries are stripped; idle after live →
   exactly one `requests: []` emit, then silence.
6. 500 / timeout / bad JSON → last-good re-emitted once with `stale: true`;
   `engine_ts_age_s` 30 with `running` 1 → `stale: true`; same age with
   `running`/`waiting` 0 → not stale.
7. 404 ×2 → origin disabled, polling stops; re-enabled after the poller
   restarts.
8. Lock discipline: emission happens with neither `STREAMS_LOCK`,
   `ACTIVE_RUNS_LOCK`, nor the module lock held (assert from inside the fake
   `put_nowait`).
9. Not journaled: a fake `run_journal` sees no `prefill_progress` append.
10. api_key never appears in log records (caplog) or payloads.
11. Worker cleanup calls unregister on success, error, and cancel paths.

Client — follow the existing `tests/*.test.js` convention:

12. Each table row in §2.3 (incl. cached-prefix, null `cached_tokens`,
    `decode` with `computed > prompt`, unknown phase), stale dimming, >15 s reverting to
    `Running`, `finished`-only snapshot → `Running`, cross-session/
    cross-stream events ignored, slot cleared by `hideLiveRunStatus`.
13. Locale parity: the five keys exist in every locale block. Confirm which
    existing test covers `static/i18n.js` key parity; extend it if it does not
    pick new keys up automatically.

Manual (before/after evidence per AGENTS.md):

14. Live ada session, ~150k-token **uncached** prompt (new content, so the prefix
    cache does not hide the prefill): screenshots at queued (start a second
    request while one is running) → prefill with ETA → decoding (`Running`).
    Desktop and 360 px.
15. Same session, second turn (mostly cached): shows small new-work count
    with ctx suffix.
16. Non-ada provider and ada with `PREFILL_PROGRESS` unset: status line is
    byte-identical to today (regression evidence).

## 4. Out of scope / follow-ups

- **No ada backend change is required** (checked against the rev 2 spec and
  `prefill_progress.py` on ada). Optional nice-to-haves, none blocking: a
  query flag to omit finished entries, and honoring a client `X-Request-Id`
  so a run could match its own request during prefill.
  The latter would also need hermes-agent to send the header, so it belongs
  with the gateway follow-up below.
- Gateway-routed runs: separate spec — the agent forwards a
  `prefill_progress` event on its stream and the WebUI relays it through the
  existing gateway event plumbing.
- No changes to Steer/Stop/cancel semantics; the poller is a pure observer.
- The settled-turn footer's "Working…" and the hard-coded `Running` label are
  untouched.

## 5. Anchors (verified 2026-10-05, current tree)

- Cross-thread emit pattern: `api/background_process.py:650`
  (`_emit_to_session_streams`).
- Worker-local journaled `put()` (do not use): `api/streaming.py:11020`.
- `_sse` handler writer (do not use): `api/streaming.py:9583`.
- Resolved base_url/api_key → agent kwargs: `api/streaming.py` ~12445;
  provider-switch rebuild ~13440; worker cleanup / `unregister_active_run`
  ~15266.
- Config read pattern: `api/config.py:542` (`webui_cfg.get(...)`).
- Live run status line: `static/ui.js` ~15348–15420
  (`placeLiveRunStatusHost`, `_renderLiveRunStatusContent`,
  `updateLiveRunStatus`, `hideLiveRunStatus`, `_startLiveRunStatusTimer`).
- `todo_state` listener (guard template): `static/messages.js:6091`.
- `t()` placeholder semantics: `static/i18n.js` `function t`.
- Server contract: ada `PREFILL_PROGRESS_SPEC.md` rev 2 (impl `82bf273`;
  module `prefill_progress.py`, `handler_payload` / `_reconcile`).
