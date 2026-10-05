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
KEYS = ("prefill_queued", "prefill_progress", "prefill_size", "prefill_restoring", "prefill_ctx_suffix")


def _en_strings():
    out = {}
    for k in KEYS:
        m = re.search(r"\n    %s: '([^']*)'," % k, I18N_JS)
        assert m, k
        out[k] = m.group(1)
    return out


def _label(snap, stream_id="s1", now=100_000):
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
console.log(JSON.stringify(_liveRunPrefillLabel(%s, %s, %s)));
""" % (json.dumps(UI_JS), json.dumps(_en_strings()), json.dumps(snap), json.dumps(stream_id), json.dumps(now))
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
    # decode with computed > prompt (observed live): plain Running
    (_req(phase="decode", prompt_tokens=129048, computed_tokens=129924, remaining_prompt_tokens=0,
          prefill_eta_s=None), None),
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
