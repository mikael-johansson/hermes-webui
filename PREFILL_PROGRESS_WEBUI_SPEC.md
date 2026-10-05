# Spec: prefill-progress visibility in Hermes WebUI

Companion to the ada vLLM-fork spec `PREFILL_PROGRESS_SPEC.md` (commit `a82fcb3`
in `~/git/vllm/qwen38-27b-rtx3090` on ada.vpn). That spec provides
`GET /v1/prefill_progress` (opt-in env `PREFILL_PROGRESS=1` on the server,
auth = the same API key the chat endpoint uses). **Nothing in this spec ships
until the ada endpoint is live and enabled** — the poller must tolerate its
absence (see §2.4).

User priority (verbatim intent): seeing that a request is *admitted* and how
large the processing is. Granularity of the progress % is explicitly
secondary.

---

## 1. Server side (`api/streaming.py` + `api/config.py`)

### 1.1 Poller

A single process-wide poller, started lazily and refcounted:

- `PRELOAD`/registry state in `api/config.py` (next to the other runtime
  registries): `PREFILL_PROGRESS_RUNS: dict[str, dict]` (stream_id → {base_url,
  api_key, model, started_at}) + its lock, plus a `PREFFILL_PROGRESS_POLLING`
  flag.
- First active run that qualifies (§1.2) starts a daemon thread; the last one
  to settle stops it. Thread sleeps **5 s** between polls
  (`PREFILL_PROGRESS_INTERVAL_S`, default 5, read once).
- Each poll: `urllib.request` GET to `<base_url.rstrip('/')>/prefill_progress`
  with `Authorization: Bearer <api_key>` and `Content-Type: application/json`
  (the ada server 400s on a bare body — keep the header habit), 4 s timeout.
  Parse into a snapshot dict; merge into `PREFILL_PROGRESS_RUNS`.
- **Never block a run worker on the poll.** The poller runs in its own thread;
  run threads only register/unregister and read.
- Errors (DNS, timeout, 4xx/5xx, bad JSON, unknown fields): log at debug,
  keep the last-good snapshot, mark `stale: true` on it, keep polling. A
  missing endpoint (`404` — feature off on the server) disables the poller
  quietly after two consecutive 404s (no per-5s log spam; one info log).

### 1.2 Which runs qualify

A run qualifies when its resolved custom provider's `base_url` matches a
configured allowlist (new optional config key, see §1.3) **and** the run is
active (registered in `ACTIVE_RUNS`). Resolve via the existing
`resolve_custom_provider_connection` / `merge_custom_provider_runtime_bundle`
helpers in `api/config.py` — do **not** re-parse `~/.hermes/config.yaml`
directly from this module; the WebUI config path already wraps it.

### 1.3 Config key

New optional top-level key in the WebUI config (and exposed via the existing
settings plumbing only if the repo's settings schema requires it for
readability — decide at implementation, keep the code default):

```yaml
prefill_progress:
  poll_seconds: 5
  enabled_base_urls:
    - http://ada.vpn:18020
```

Default: disabled list empty → no polls, no endpoint registration on the
server side, zero behavior change. This keeps the feature off for every
deploy that doesn't opt in, per the repo's "no new long-lived processes
without a rollback story" rule (the rollback story: remove the config block).

### 1.4 SSE emission

When the poller completes a poll with a non-empty `requests` list, emit to
each registered qualifying run's stream using the existing emitter
(`api/streaming.py:9585` — the `event: {event}\ndata: {json...}\n\n`
payload helper; find its exact name and reuse it, do not duplicate):

```
event: prefill_progress
data: {"stream_id":"<run's stream id>","requests":[{...ada fields verbatim...}],"stale":false,"ts":1760000000.0}
```

- Pass ada's per-request objects through verbatim (no field mapping — the
  contract belongs to the server spec).
- Only emit when something changed materially (phase change, admitted flipped
  false→true, or ≥5 s since last emit to this stream) to bound event volume;
  the 5 s poll cadence makes at most one emit per run per poll.
- The event is **ephemeral UI state**: do not persist it to the turn journal
  in a way that re-renders on replay (check how `todo_state` handles replay —
  follow that pattern; if it's journaled, tag this one so the replay path in
  `_wireSSE` consumers simply ignores stale snapshots, §2).

### 1.5 Locks

Register/unregister touches only `STREAMS`-adjacent config state under its own
lock; the poller thread never takes `STREAMS_LOCK` while emitting (emit
appends to the stream's existing outbox, same as every other event — match
the surrounding code, do not invent a new emission path).

## 2. Client side (`static/messages.js` + `static/ui.js` + i18n)

### 2.1 Listener

In `_wireSSE` (`static/messages.js`, next to the existing
`source.addEventListener('todo_state', ...)` at ~line 6091 — that is the
template to copy):

```
source.addEventListener('prefill_progress', e => { ... });
```

- Parse `e.data` JSON; store the latest snapshot on the stream's live state
  object alongside the other per-stream scratch (find where `todo_state`
  stashes its value — same place).
- On receipt: update the transparent-turn footer of the **current turn** if
  this stream is running (§2.2).
- On stream end / `done` / session no longer `working`: clear the chip state
  (the footer's existing "premature-Done fix" logic at `static/ui.js:13081`
  already gates on `S.session.working` — hook into the same condition, do not
  add a second liveness source).
- Stale snapshots (`stale: true`): show the last data with a dimmed style, no
  ETA.

### 2.2 Footer chip

In the footer renderer that builds the `working` branch
(`static/ui.js:13081`, the `live-run-status-dot … Working…` span): when a
fresh (≤15 s old) snapshot exists for this stream, upgrade the label to:

- **not admitted**: `Queued · 180.4k prompt`
- **admitted, prefilling**: `Prefill 41k/180k · ETA 1m15s`
- **restoring**: `Restoring KV · 180k`
- admitted but rate/ETA null: `Prefill 180k…`

Rules:
- Numbers in k units, 1 decimal (180.4k). Prompt size =
  `prompt_tokens`; the user asked for "how large the processing is" —
  always show prompt size, progress fraction only when `computed_tokens`
  is meaningful (>0 and < prompt).
- ETA only when `prefill_eta_s` is non-null; never show a negative ETA.
- If multiple requests are present in the snapshot, show the one matching
  this run's in-flight request if identifiable; otherwise the largest
  `remaining_prompt_tokens` (single-user box, keep it dumb).
- The plain `Working…` label remains the fallback for every non-ada run and
  for stale data older than 15 s.

### 2.3 i18n

New keys (mirror into **every** locale block — the locale key-coverage parity
tests are the gate; see the `working` key's history for the exact pattern):
`prefill_queued` ("Queued · {n} prompt"), `prefill_progress`
("Prefill {done}/{total} · ETA {eta}"), `prefill_restoring`
("Restoring KV · {n}"), `prefill_size` ("Prefill {n}…"). Human-readable
numbers come from the formatter, keys stay static.

### 2.4 Visual

Reuse the existing `live-run-status-dot tool-card-running-dot` pulse; the chip
is a text span in the same `lf-status` position. No new CSS beyond a dim
variant for stale state. Desktop + narrow widths: the footer already
truncates long model text — the chip must not push the time/TTFT spans out;
verify at 360 px.

## 3. Tests (repo conventions: `./scripts/test.sh`, Python 3.11–3.13 venv)

Server (`tests/test_prefill_progress_poller.py`, no GPU, no network):
1. Poller starts on first qualifying run register, stops after the last
   unregister (refcount correctness, barrier-based).
2. Feed the poller a fake HTTP layer (monkeypatch `urllib.request.urlopen`):
   happy path emits `prefill_progress` to the registered stream via the
   existing SSE outbox (assert exact `event:` line format); 404 ×2 disables
   quietly; 500 keeps last-good and sets `stale`; bad JSON same.
3. Non-qualifying run (different base_url) → zero polls.
4. Emission never takes `STREAMS_LOCK` (assert lock non-nesting with a
   barrier, per the repo's lock-invariant test style).
5. Config off (empty allowlist) → no thread created, no emitter registered.

Client (JS test, follow the existing `*.test.js` convention in `tests/`):
6. Listener updates footer text per state in §2.2; clears on `working=false`;
   stale >15 s reverts to plain `Working…`.
7. Locale parity: the new keys exist in every locale block (the existing
   parity test should pick them up automatically — extend if it doesn't).

Manual (before/after evidence, per AGENTS.md):
8. Live session against ada with `PREFILL_PROGRESS=1` on the server, a
   ~150k-token prompt: screenshot the footer at queued → admitted → prefill →
   decode; plus one narrow-width (360 px) shot.
9. Ada endpoint disabled: footer shows plain `Working…` exactly as today
   (regression evidence).

## 4. Out of scope / follow-ups

- **hermes-agent side is NOT part of this spec.** The WebUI talks to the ada
  server directly (it already resolves the provider's base_url + api_key via
  `api/config.py`). If later we want progress on *gateway-routed* runs (where
  the agent, not the WebUI, owns the vLLM connection), that's a separate
  spec: agent forwards a `prefill_progress` event on its stream and the WebUI
  just relays it through the existing gateway event plumbing.
- No changes to Steer/Stop/cancel semantics; the poller is a pure observer.
- No metrics endpoint scraping (the fork's `/metrics` stays for humans).

## 5. Anchors (verified 2026-10-05, current tree)

- SSE payload helper: `api/streaming.py:9585`.
- `todo_state` listener (template for §2.1): `static/messages.js:6091`.
- Working-footer (chip goes here): `static/ui.js:13081`.
- Provider resolution: `api/config.py` (`resolve_custom_provider_connection`,
  `merge_custom_provider_runtime_bundle`, `_named_custom_provider_slug_for_base_url`).
- Live run registry: `ACTIVE_RUNS` in `api/config.py` / `api/streaming.py`.
- Server contract (do not drift): ada repo `PREFILL_PROGRESS_SPEC.md`
  (commit `a82fcb3`).
