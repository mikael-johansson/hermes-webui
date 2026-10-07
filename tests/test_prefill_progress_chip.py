"""Client-side prefill chip (PREFILL_PROGRESS_WEBUI_SPEC.md §2, §3 items 12-13).

Runs the real ``_liveRunPrefillLabel`` / ``_fmtTokens`` / ``_formatTurnDuration``
extracted from static/ui.js under node, with ``t()`` backed by the real English
strings from static/i18n.js.
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.resolve()
UI_JS = (REPO_ROOT / "static" / "ui.js").read_text(encoding="utf-8")
I18N_JS = (REPO_ROOT / "static" / "i18n.js").read_text(encoding="utf-8")
MESSAGES_JS = (REPO_ROOT / "static" / "messages.js").read_text(encoding="utf-8")
NODE = shutil.which("node")
KEYS = ("prefill_queued", "prefill_progress", "prefill_size", "prefill_restoring", "prefill_ctx_suffix",
        "llm_call_pp", "llm_call_tg", "llm_call_queued", "llm_call_stats_title",
        "llm_decode_progress", "llm_decode_tokens", "llm_call_pp_n", "llm_call_tg_n")


def _en_strings():
    out = {}
    for k in KEYS:
        m = re.search(r"\n    %s: '([^']*)'," % k, I18N_JS)
        assert m, k
        out[k] = m.group(1)
    return out


def _label(snap, stream_id="s1", now=100_000):
    return _run_js("_liveRunPrefillLabel(%s, %s, %s)" % (json.dumps(snap), json.dumps(stream_id), json.dumps(now)))


def _call_text(cs):
    return _run_js("_llmCallStatsText(%s)" % json.dumps(cs))


def _run_js(expr):
    src = """
const src = %s;
function extractFunc(name) {
  const start = src.search(new RegExp('function\\\\s+' + name + '\\\\s*\\\\('));
  if (start < 0) throw new Error(name + ' not found');
  let i = src.indexOf('{', start); let depth = 1; i++;
  while (depth > 0 && i < src.length) { if (src[i] === '{') depth++; else if (src[i] === '}') depth--; i++; }
  return src.slice(start, i);
}
const EN = %s;
function t(key, ...args) {
  const v = EN[key]; if (v === undefined) return key;
  return v.replace(/\\{(\\d+)\\}/g, (m, i) => (i in args ? String(args[i]) : m));
}
const _LIVE_RUN_PREFILL_FRESH_MS = 15000;
eval(extractFunc('_fmtTokens'));
eval(extractFunc('_formatTurnDuration'));
eval(extractFunc('_liveRunPrefillLabel'));
eval(extractFunc('_llmCallStatsText'));
console.log(JSON.stringify(%s));
""" % (json.dumps(UI_JS), json.dumps(_en_strings()), expr)
    res = subprocess.run([NODE], input=src, capture_output=True, text=True, timeout=30)
    if res.returncode:
        raise RuntimeError(res.stderr)
    return json.loads(res.stdout)


def _req(**over):
    r = {"request_id": "chatcmpl-1", "phase": "prefill", "admitted": True,
         "arrived_ms": 1000, "prompt_tokens": 180412, "computed_tokens": 41200,
         "cached_tokens": 0, "remaining_prompt_tokens": 139212,
         "prefill_eta_s": 75.2, "restoring_kv": False}
    r.update(over)
    return r


def _snap(*reqs, stale=False, received=99_000, stream="s1"):
    return {"stream_id": stream, "requests": list(reqs), "stale": stale, "receivedAt": received}


def _text(snap, **kw):
    out = _label(snap, **kw)
    return out and out["text"]


node = pytest.mark.skipif(NODE is None, reason="node not on PATH")


@node
@pytest.mark.parametrize("req,expected", [
    (_req(phase="queued", admitted=False, computed_tokens=0, remaining_prompt_tokens=180412, prefill_eta_s=None),
     "Queued · 180.4k prompt"),
    # preempted back to waiting: same label
    (_req(phase="queued", admitted=False, prefill_eta_s=None), "Queued · 180.4k prompt"),
    (_req(phase="restoring", restoring_kv=True, prefill_eta_s=None), "Restoring KV · 180.4k"),
    (_req(), "Prefill 41.2k/180.4k · ETA 1m15s"),
    (_req(prefill_eta_s=None), "Prefill 180.4k…"),
    (_req(prefill_eta_s=12), "Prefill 41.2k/180.4k · ETA 12s"),
    # cached prefix: new work + context size
    (_req(prompt_tokens=129048, cached_tokens=127296, computed_tokens=128000,
          remaining_prompt_tokens=1048, prefill_eta_s=4),
     "Prefill 704/1.8k · ETA 4s (129.0k ctx)"),
    # null cached_tokens
    (_req(cached_tokens=None, prefill_eta_s=None), "Prefill 180.4k…"),
    # decode with computed > prompt (observed live), server before rev 3
    # (no generated_tokens): plain Running
    (_req(phase="decode", prompt_tokens=129048, computed_tokens=129924, remaining_prompt_tokens=0,
          prefill_eta_s=None), None),
    # rev 3 live decode
    (_req(phase="decode", remaining_prompt_tokens=0, prefill_eta_s=None,
          generated_tokens=467, decode_rate_tok_s=137.25), "467 tokens · 137.3 t/s"),
    (_req(phase="decode", remaining_prompt_tokens=0, prefill_eta_s=None,
          generated_tokens=1, decode_rate_tok_s=None), "1 tokens"),
    (_req(phase="decode", remaining_prompt_tokens=0, prefill_eta_s=None,
          generated_tokens=0, decode_rate_tok_s=None), None),
    # unknown phase: prefill-like while work remains, Running otherwise
    (_req(phase="mystery", prefill_eta_s=99), "Prefill 180.4k…"),
    (_req(phase="mystery", remaining_prompt_tokens=0), None),
])
def test_label_table(req, expected):
    assert _text(_snap(req)) == expected


@node
def test_finished_only_snapshot_is_running():
    assert _label(_snap(_req(phase="finished", remaining_prompt_tokens=0))) is None
    assert _label(_snap()) is None


@node
def test_picks_largest_remaining_then_most_recent():
    small = _req(request_id="a", remaining_prompt_tokens=10, prompt_tokens=20, computed_tokens=10, prefill_eta_s=None)
    big = _req(request_id="b", phase="queued", admitted=False, prompt_tokens=50000, remaining_prompt_tokens=50000)
    fin = _req(request_id="c", phase="finished", remaining_prompt_tokens=0, prompt_tokens=999999)
    assert _text(_snap(small, fin, big)) == "Queued · 50.0k prompt"
    older = _req(request_id="o", phase="queued", prompt_tokens=1000, remaining_prompt_tokens=1000, arrived_ms=9000)
    newer = _req(request_id="n", phase="queued", prompt_tokens=2000, remaining_prompt_tokens=1000, arrived_ms=10)
    assert _text(_snap(older, newer)) == "Queued · 2.0k prompt"


@node
def test_stale_dims_and_drops_eta():
    out = _label(_snap(_req(), stale=True))
    assert out == {"text": "Prefill 180.4k…", "stale": True}


@node
def test_freshness_window_and_stream_scoping():
    assert _text(_snap(_req(), received=100_000 - 15_000)) is not None
    assert _label(_snap(_req(), received=100_000 - 15_001)) is None
    assert _label(_snap(_req(), stream="other")) is None
    assert _label(_snap(_req()), stream_id=None) is None
    assert _label(None) is None


def test_locale_parity_for_prefill_keys():
    blocks = re.findall(r"\n  (?:'[A-Za-z-]+'|[a-z]+): \{\n", I18N_JS)
    working = len(re.findall(r"\n    working: '", I18N_JS))
    assert working == len(blocks) >= 15
    for k in KEYS:
        assert len(re.findall(r"\n    %s: '" % k, I18N_JS)) == working, k
    # t() only fills numbered placeholders.
    assert not re.search(r"\n    prefill_[a-z_]+: '[^']*\{[a-z]", I18N_JS)


def test_listener_is_wired_with_cross_session_guards():
    i = MESSAGES_JS.index("source.addEventListener('prefill_progress'")
    body = MESSAGES_JS[i:MESSAGES_JS.index("source.addEventListener('approval'", i)]
    assert "d.session_id!==activeSid" in body
    assert "d.stream_id!==streamId" in body
    assert "S.activeStreamId!==streamId" in body
    assert "setLiveRunPrefill(" in body and "updateLiveRunStatus(" in body
    for forbidden in ("INFLIGHT", "persistInflightState", "localStorage", "S.session."):
        assert forbidden not in body.replace("S.session.session_id", ""), forbidden


def test_hide_clears_slot_and_render_uses_it():
    i = UI_JS.index("function hideLiveRunStatus(")
    assert "_liveRunPrefill=null;" in UI_JS[i:UI_JS.index("\n}", i)]
    j = UI_JS.index("function _renderLiveRunStatusContent(")
    render = UI_JS[j:UI_JS.index("\n}", j)]
    assert "_liveRunPrefillLabel(_liveRunPrefill,S&&S.activeStreamId,Date.now())" in render
    assert "esc(prefill.text)" in render
    assert '<span class="lf-status">Running</span>' in render  # unchanged fallback


@node
def test_decode_stale_drops_rate():
    r = _req(phase="decode", remaining_prompt_tokens=0, prefill_eta_s=None,
             generated_tokens=467, decode_rate_tok_s=137.3)
    assert _label(_snap(r, stale=True)) == {"text": "467 tokens", "stale": True}


@node
@pytest.mark.parametrize("cs,text", [
    # The FINISHED line from the user's ada log.
    ({"pp_tok_s": 1186.5, "tg_tok_s": 120.0, "queue_wait_s": 0.1}, "PP 1186.5 t/s · TG 120.0 t/s"),
    ({"pp_tok_s": 2018.0, "tg_tok_s": 124.43, "queue_wait_s": 12.4}, "PP 2018.0 t/s · TG 124.4 t/s · queued 12s"),
    ({"pp_tok_s": 2018.0, "tg_tok_s": 124.4, "queue_wait_s": 75}, "PP 2018.0 t/s · TG 124.4 t/s · queued 1m15s"),
    ({"pp_tok_s": None, "tg_tok_s": 99.0}, "TG 99.0 t/s"),  # fully cached prompt
    ({"pp_tok_s": 500.0, "tg_tok_s": None}, "PP 500.0 t/s"),
    ({"pp_tok_s": None, "tg_tok_s": None, "queue_wait_s": 3}, None),
    # With counts (ada rev 3 always sends them): tokens, then the rate.
    ({"pp_tok_s": 1601.2, "computed_prompt_tokens": 3628, "tg_tok_s": 127.46, "completion_tokens": 764},
     "PP 3628 (1601.2 t/s) · TG 764 (127.5 t/s)"),
    ({"pp_tok_s": None, "computed_prompt_tokens": 0, "tg_tok_s": 99.0, "completion_tokens": 12},
     "TG 12 (99.0 t/s)"),
    ({"pp_tok_s": 0, "tg_tok_s": -1}, None),
    (None, None),
])
def test_call_stats_text(cs, text):
    out = _call_text(cs)
    assert (out and out["text"]) == text


@node
def test_call_stats_title():
    out = _call_text({"pp_tok_s": 1186.5, "tg_tok_s": 120.0, "prompt_tokens": 89933,
                      "computed_prompt_tokens": 1741, "ttft_s": 1.6, "completion_tokens": 601})
    assert out["title"] == "Prompt 89.9k (1.7k computed) · TTFT 1.6s · 601 output tokens"


def test_call_stats_listener_and_render_wiring():
    i = MESSAGES_JS.index("source.addEventListener('llm_call_stats'")
    body = MESSAGES_JS[i:MESSAGES_JS.index("source.addEventListener('approval'", i)]
    for guard in ("d.session_id!==activeSid", "d.stream_id!==streamId", "S.activeStreamId!==streamId"):
        assert guard in body, guard
    assert "appendLiveLlmCallStats(d)" in body
    for forbidden in ("INFLIGHT", "persistInflightState", "localStorage"):
        assert forbidden not in body, forbidden
    # Settled: placed by a post-render pass after every row source (activity
    # loop and recorded anchor scenes, which rebuild rows without message
    # indexes), keyed by the message's tool-call ids.
    j = UI_JS.index("_renderSettledAnchorSceneForMessage(msg, seg, rawIdx);")
    assert "_placeLlmCallStatsRows(inner);" in UI_JS[j:j + 500]
    f = UI_JS.index("function _llmCallEndRow(")
    end_fn = UI_JS[f:UI_JS.index("\n}", f)]
    assert "data-tool-disclosure-key" in end_fn and "'id:'+id" in end_fn
    assert "assistant-segment-worklog-source" in end_fn  # hidden sources never anchor
    assert "if(ids.length) return null;" in end_fn  # tools not found: no line
    # Live rows dedupe by request id; after the newest tool row unless a line
    # already follows it, else before the run-status line.
    f = UI_JS.index("function appendLiveLlmCallStats(")
    live = UI_JS[f:UI_JS.index("\n}", f)]
    assert "data-request-id" in live and "blocks.insertBefore(row,status)" in live
    assert "querySelectorAll('.tool-card-row')" in live
    # textContent only: no HTML injection from server numbers.
    g = UI_JS.index("function _llmCallStatsRow(")
    assert "innerHTML" not in UI_JS[g:UI_JS.index("\n}", g)]


def _settle_run(replies, session, active_stream=None):
    """Drive _scheduleWorkingSettleCheck with an immediate timer queue and a
    scripted /api/session/working (an Error reply throws)."""
    src = """
const src = %s;
function extractFunc(name) {
  const start = src.search(new RegExp('function\\\\s+' + name + '\\\\s*\\\\('));
  let i = src.indexOf('{', start); let depth = 1; i++;
  while (depth > 0 && i < src.length) { if (src[i] === '{') depth++; else if (src[i] === '}') depth--; i++; }
  return src.slice(start, i);
}
const delays = []; const queue = [];
global.setTimeout = (fn, ms) => { delays.push(ms); queue.push(fn); return queue.length; };
global.clearTimeout = () => {};
const replies = %s; const urls = []; let reloads = 0; let settled = 0;
global.api = async (url) => { urls.push(url); const r = replies.shift(); if (r === 'ERR') throw new Error('x'); return r; };
global.loadSession = () => { reloads++; };
global.refreshLastTurnFooterStatus = () => { settled++; };
global.S = {session: %s, activeStreamId: %s};
const _WORKING_SETTLE_DELAYS_MS=[500,1000,2000,3000,5000,8000,13000,20000,30000];
const _WORKING_SETTLE_MAX_ERRORS=3;
const _workingSettleTimers={};
eval(extractFunc('_scheduleWorkingSettleCheck'));
eval(extractFunc('_settleWorkingFooters'));
(async () => {
  _scheduleWorkingSettleCheck('a');
  while (queue.length) { await queue.shift()(); }
  console.log(JSON.stringify({delays, calls: urls.length, url: urls[0] || '', working: S.session.working, settled, reloads}));
})();
""" % (json.dumps(UI_JS), json.dumps(replies), json.dumps(session), json.dumps(active_stream))
    res = subprocess.run([NODE], input=src, capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


@node
def test_working_settle_check_flips_footer_once_server_reports_done():
    out = _settle_run([{"working": True}, {"working": True}, {"working": False}],
                      {"session_id": "a", "working": True})
    assert out["delays"] == [500, 1000, 2000]
    assert out["url"] == "/api/session/working?session_id=a"
    assert out["calls"] == 3 and out["working"] is False and out["settled"] == 1
    assert out["reloads"] == 0  # a WebUI-owned session needs no reload


@node
def test_working_settle_check_keeps_going_while_working_and_caps_delay():
    """A delegated child can run for many minutes: keep checking (capped at
    30 s) for as long as the server says working."""
    out = _settle_run([{"working": True}] * 14 + [{"working": False}],
                      {"session_id": "a", "working": True, "read_only": True})
    assert out["calls"] == 15 and out["delays"][-6:] == [30000] * 6
    assert out["working"] is False and out["reloads"] == 1  # read-only child: final reload


@node
def test_working_settle_check_gives_up_after_consecutive_errors_and_yields_to_new_run():
    out = _settle_run(["ERR", "ERR", "ERR", {"working": False}], {"session_id": "a", "working": True})
    assert out["calls"] == 3 and out["working"] is True and out["settled"] == 0
    out = _settle_run([{"working": False}], {"session_id": "a", "working": True}, active_stream="new")
    assert out["calls"] == 0 and out["working"] is True  # a newer run owns the flag


def test_done_handler_schedules_working_settle():
    i = MESSAGES_JS.index("S.session=d.session;S.messages=_carryForwardEphemeralTurnFields(")
    assert "_scheduleWorkingSettleCheck(activeSid)" in MESSAGES_JS[i:i + 600]


def test_live_call_stats_survive_anchor_scene_repack():
    """The live anchor scene re-packs its rows above #liveRunStatus on every
    update, sweeping non-scene nodes (the live PP/TG lines) to the top. Lines
    remember their tool row and are re-seated after every re-pack."""
    f = UI_JS.index("function appendLiveLlmCallStats(")
    live = UI_JS[f:UI_JS.index("\n}", f)]
    # Stats usually land BEFORE their own call's tool row: an unclaimed newest
    # tool row is this call's, otherwise the line is pending and claims the
    # first tool row that appears next.
    assert "if(newest&&!_liveToolKeyClaimed(blocks,newest)) row.setAttribute('data-after-tool-key',newest);" in live
    assert "else row.setAttribute('data-pending-tool-count',String(keys.length));" in live
    assert "_relocateLiveCallStats(blocks);" in live
    g = UI_JS.index("function _relocateLiveCallStats(")
    reloc = UI_JS[g:UI_JS.index("\n}", g)]
    assert "data-after-tool-key" in reloc and "data-tool-disclosure-key" in reloc
    assert "data-pending-tool-count" in reloc and "row.removeAttribute('data-pending-tool-count')" in reloc
    assert "if(anchor.nextElementSibling!==row)" in reloc  # idempotent
    h = UI_JS.index("function _renderLiveAnchorActivitySceneTransparent(")
    scene = UI_JS[h:UI_JS.index("\nfunction ", h + 10)]
    assert scene.index("_relocateLiveCallStats(blocks)") > scene.index("blocks.insertBefore(renderedNode,expectedNextSibling)")
