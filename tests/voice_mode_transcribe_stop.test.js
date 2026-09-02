// Regression test: _vmTranscribeStop must build its POST blob from the audio
// chunks already buffered, NOT from a fresh read of _vmChunks after
// _vmStopCapture() has reset it to null.
//
// BUG (the exact "eternal Listening" incident, 2026-09-02):
//     const recorder=_vmRecorder;
//     const mime=recorder?recorder.mimeType:_vmRecorderMime;
//     _vmStopCapture();                                   // ← _vmChunks = null
//     const blob=_vmChunks&&_vmChunks.length?new Blob(...)  // ← ALWAYS null
//     if(!blob||!blob.size){ ... _startListening(); return; } // ← loops forever
// _vmStopCapture() nulls _vmChunks; the old code then read _vmChunks *after*
// that reset, so the blob was always empty. Every gap-stop took the "no
// audio" branch and re-armed listening → the UI sat on "Listening…" with a
// silent re-arm loop and no /api/transcribe POST.
//
// FIX: snapshot `const chunks=_vmChunks;` BEFORE _vmStopCapture() and build
// the blob from that snapshot.
//
// This test extracts the REAL _vmVadTick/_vmStopCapture/_vmTranscribeStop
// block from boot.js and executes it in a sandbox with a stubbed recorder,
// File/FormData/fetch. It asserts the POST is made with the buffered audio
// and the state moves to 'thinking'. It FAILS on the old read-after-reset
// code (no fetch, state stays 'listening') and PASSES on the fixed code.
//
// NOTE: the stubs close over a plain `state` object (arrows, no `this`)
// because this test module is strict and the vm's sloppy code calls the
// stubs as free functions — a `this`-based stub would see `this===undefined`.
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const bootSrc = fs.readFileSync(
  path.join(__dirname, '..', 'static', 'boot.js'), 'utf8'
);

// Extract the server-STT stop path: _vmVadTick + _vmStopCapture +
// _vmTranscribeStop (consecutive in the file, ending right before
// _vmRecorderMime).
function extractStopBlock(src) {
  const start = src.indexOf('function _vmVadTick(');
  if (start === -1) throw new Error('_vmVadTick not found');
  const end = src.indexOf('let _vmRecorderMime=', start);
  if (end === -1) throw new Error('_vmRecorderMime anchor not found');
  return src.slice(start, end);
}

const stopSrc = extractStopBlock(bootSrc);

// Minimal Blob stub with a real .size (ondataavailable chunks are Blobs).
class StubBlob {
  constructor(parts, opts) {
    this.parts = parts;
    this.type = opts && opts.type;
    this.size = parts.reduce((a, p) => a + (p.size || p.byteLength || 0), 0);
  }
}
class StubFile extends StubBlob {
  constructor(parts, name, opts) { super(parts, opts); this.name = name; }
}
class StubFormData {
  constructor() { this.entries = []; }
  append(k, v) { this.entries.push([k, v]); }
}

function runStopCase({ recorderState, chunks }) {
  // Live state shared between the sandbox globals and the stubs.
  const state = {
    _voiceModeActive: true,
    _voiceModeState: 'listening',
    events: [],
  };
  const recorder = {
    state: recorderState,
    mimeType: 'audio/webm;codecs=opus',
    stop() { this.state = 'inactive'; },
  };
  const fetches = [];
  const sandbox = {
    console: { info: () => {}, log: () => {}, warn: () => {} },
    Date, JSON, Math,
    setInterval: () => 0,
    clearInterval: () => {},
    // Module-scoped voice-mode state as seen by the extracted block.
    _vmSttFails: 0,
    _vmStopping: false,
    _vmOnsetFrames: 8,
    _vmSilenceFrames: 12,
    _vmRecorderMime: 'audio/webm;codecs=opus',
    _vmChunks: null,
    _vmRecorder: recorder,
    _vmStream: { getTracks: () => [] },
    _vmAudioCtx: null,
    _vmAnalyser: null,
    _vmVadLoop: null,
    _vmVadState: null,
    ta: { value: '' },
    // DOM / fetch stubs.
    Blob: StubBlob,
    File: StubFile,
    FormData: StubFormData,
    // Stubs the extracted boot.js code calls — arrows closing over `state`.
    _setState: (s) => { state._voiceModeState = s; state.events.push('setState:' + s); },
    _deactivate: () => { state._voiceModeActive = false; state.events.push('deactivate'); },
    setComposerStatus: (s) => { state.events.push('status:' + s); },
    t: (k) => k,
    showToast: (m) => { state.events.push('toast:' + m); },
    _startListening: () => { state.events.push('startListening'); },
    _voiceModeSend: () => { state.events.push('send'); },
    autoResize: () => {},
    _vmDebug: () => {},
  };
  sandbox.fetch = (url, opts) => {
    fetches.push({ url: String(url), method: opts && opts.method, body: opts && opts.body });
    return Promise.resolve({
      ok: true, status: 200,
      json: () => Promise.resolve({ ok: true, transcript: 'hello world' }),
    });
  };
  // Populate the buffered ondataavailable chunks (real stub-Blobs, 10B each).
  sandbox._vmChunks = chunks
    ? [new StubBlob([new Uint8Array(10)], { type: 'audio/webm' }),
       new StubBlob([new Uint8Array(10)], { type: 'audio/webm' }),
       new StubBlob([new Uint8Array(10)], { type: 'audio/webm' })]
    : null;
  // The extracted block reads _voiceModeActive/_voiceModeState as free
  // identifiers; bind them to `state` so the stubs and the code agree.
  Object.defineProperty(sandbox, '_voiceModeActive', {
    get: () => state._voiceModeActive,
    set: (v) => { state._voiceModeActive = v; },
    configurable: true,
  });
  Object.defineProperty(sandbox, '_voiceModeState', {
    get: () => state._voiceModeState,
    set: (v) => { state._voiceModeState = v; },
    configurable: true,
  });

  const context = vm.createContext(sandbox);
  vm.runInContext(stopSrc, context);
  vm.runInContext('globalThis._vmTranscribeStop("stop");', context);
  // Flush microtasks so the fetch .then/.catch chains run.
  return new Promise((resolve) => {
    setTimeout(() => {
      const entry = fetches.find(f => String(f.url).includes('transcribe') && f.method === 'POST');
      resolve({
        postMade: !!entry,
        // form.append('file', file) → entries = [['file', File]] → index 0.
        postSize: entry && entry.body && entry.body.entries
          ? (entry.body.entries[0] && entry.body.entries[0][1] ? entry.body.entries[0][1].size : null) : null,
        postName: entry && entry.body && entry.body.entries && entry.body.entries[0] && entry.body.entries[0][1]
          ? entry.body.entries[0][1].name : null,
        state: state._voiceModeState,
        active: state._voiceModeActive,
        events: state.events,
        sendFired: state.events.includes('send'),
      });
    }, 25);
  });
}

function assert(cond, msg) {
  if (!cond) { console.error('FAIL: ' + msg); process.exit(1); }
  console.log('  ok - ' + msg);
}

(async () => {
  console.log('vmTranscribeStop chunk-snapshot semantics (executes real boot.js code):');

  // 1. Normal gap-stop: recorder 'recording' + 3 buffered chunks.
  const normal = await runStopCase({ recorderState: 'recording', chunks: true });
  assert(normal.postMade, 'gap-stop POSTs /api/transcribe (old code: never — blob was null after _vmStopCapture)');
  assert(normal.postSize === 30, 'POST blob is built from the 3 buffered chunks (30 bytes)');
  assert(normal.state === 'thinking', 'state moves listening -> thinking on the POST');
  assert(normal.sendFired, 'transcript is placed in the composer and auto-sent');

  // 2. Barge-in already tore the capture down: recorder 'inactive'.
  const torn = await runStopCase({ recorderState: 'inactive', chunks: true });
  assert(!torn.postMade, 'recorder already inactive (barge-in) -> no transcribe POST');
  assert(!torn.events.includes('deactivate'), 'no spurious deactivate on the already-inactive path');

  // 3. No audio was ever buffered (recorder started but no data).
  const noaudio = await runStopCase({ recorderState: 'recording', chunks: false });
  assert(!noaudio.postMade, 'no buffered chunks -> no POST');
  assert(noaudio.events.includes('startListening'), 'no-audio path re-arms listening (expected when truly silent)');

  console.log('all vmTranscribeStop cases behave correctly (regression locked)');
})().catch((e) => { console.error('ERROR', e); process.exit(1); });
