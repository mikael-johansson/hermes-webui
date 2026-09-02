// Regression test: the voice-mode capability probe's re-read guard.
//
// BUG (this exact incident): the guard was written as
//     const stored = localStorage.getItem(key);
//     if (stored !== 'null') { _vmServerSttOk = stored==='1'; return; }
// localStorage.getItem() returns JS null (NOT the string 'null') when the key
// is absent. So on a fresh browser, stored===null and `null !== 'null'` is
// true → the probe returned BEFORE ever calling fetch() and NEVER wrote the
// key. The server-STT engine was silently disabled and voice mode fell back
// to browser SpeechRecognition, which (behind the VPN / in Brave) re-armed
// itself forever → the UI stuck on "Listening…" with no console or network.
//
// FIX: `if (stored != null)` — a loose null check so an ABSENT key (null)
// falls through to the fetch, while a real prior result ('1'/'0') short-circuits.
//
// This test extracts the real `_probeVoiceModeServerStt` body from boot.js and
// executes it against a stubbed localStorage + fetch, asserting the guard
// behaves correctly for all three key states. It FAILS on the old `!=='null'`
// code (absent key → no fetch) and PASSES on the fixed `!= null` code.
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const bootSrc = fs.readFileSync(
  path.join(__dirname, '..', 'static', 'boot.js'), 'utf8'
);

// Extract the probe function source (from its definition up to the closing
// brace that precedes its call site).
function extractProbe(src) {
  const start = src.indexOf('async function _probeVoiceModeServerStt(');
  if (start === -1) throw new Error('probe function not found');
  const call = src.indexOf('_probeVoiceModeServerStt();', start);
  return src.slice(start, call);
}

const probeSrc = extractProbe(bootSrc);

function runProbe({ storedValue, fetchReturns }) {
  // Minimal DOM-free sandbox that provides exactly what the probe touches.
  const sandbox = {
    window: {
      MediaRecorder: function () {}, // present
    },
    navigator: {
      mediaDevices: { getUserMedia: function () {} }, // present
    },
    console,
  };
  // localStorage stub: getItem returns the stored value, or null if absent.
  const store = {};
  if (storedValue !== undefined) store['hermes-voice-mode-server-stt'] = storedValue;
  sandbox.localStorage = {
    getItem: (k) => (k in store ? store[k] : null),
    setItem: (k, v) => { store[k] = String(v); },
  };
  // fetch stub: record whether it was called and what it "returns".
  let fetchCalled = false;
  let written = null;
  sandbox.fetch = async (url) => {
    fetchCalled = true;
    return { ok: true, json: async () => ({ ok: true, available: !!fetchReturns }) };
  };
  sandbox.__result = { fetchCalled: () => fetchCalled, written: () => written };

  // Run probe in a fresh context; expose the module-scoped _vmServerSttOk by
  // capturing the localStorage write (the only observable side effect).
  const context = vm.createContext(sandbox);
  // The probe references module-scoped `_vmServerSttKey` and `_vmServerSttOk`.
  // Provide them in the sandbox so the extracted source is self-contained.
  sandbox._vmServerSttKey = 'hermes-voice-mode-server-stt';
  sandbox._vmServerSttOk = false;
  // The probe emits _vmDebug() trace lines (added 2026-09-02 for live
  // diagnosis). The extracted source is the probe function only, so stub the
  // helper as a no-op — it has no bearing on the guard semantics under test.
  sandbox._vmDebug = () => {};
  vm.runInContext(probeSrc + '\n;globalThis.__ran=true;', context);
  // Await the (async) probe to settle.
  return new Promise((resolve) => {
    // The probe is async and self-invoking? No — we extracted only the
    // FUNCTION DEFINITION, not the call. Invoke it explicitly and await.
    vm.runInContext(
      '(async()=>{ await globalThis._probeVoiceModeServerStt(); })();',
      context
    );
    // Poll for the async completion (probe does one fetch + setItem).
    let tries = 0;
    const timer = setInterval(() => {
      tries++;
      const wrote = context.localStorage.getItem('hermes-voice-mode-server-stt');
      if (wrote !== null || fetchCalled || tries > 200) {
        clearInterval(timer);
        resolve({
          fetchCalled: context.__result.fetchCalled(),
          written: wrote,
          ok: context._vmServerSttOk,
        });
      }
    }, 5);
  });
}

function assert(cond, msg) {
  if (!cond) { console.error('FAIL: ' + msg); process.exit(1); }
  console.log('  ok - ' + msg);
}

(async () => {
  console.log('probe guard semantics (executes real boot.js code):');

  // 1. ABSENT key (fresh browser) — the exact bug scenario.
  const absent = await runProbe({ storedValue: undefined, fetchReturns: true });
  assert(absent.fetchCalled, 'absent key -> probe MUST call fetch (was the bug: it skipped)');
  assert(absent.written === '1', "absent key + available -> writes '1'");
  assert(absent.ok === true, "absent key + available -> _vmServerSttOk === true");

  // 2. Prior result '1' — must short-circuit (no fetch) and stay enabled.
  const on = await runProbe({ storedValue: '1', fetchReturns: true });
  assert(!on.fetchCalled, "stored '1' -> short-circuits, no fetch");
  assert(on.written === '1', "stored '1' -> stays '1'");
  assert(on.ok === true, "stored '1' -> _vmServerSttOk === true");

  // 3. Prior result '0' (server said unavailable) — short-circuit, stay off.
  const off = await runProbe({ storedValue: '0', fetchReturns: true });
  assert(!off.fetchCalled, "stored '0' -> short-circuits, no fetch");
  assert(off.written === '0', "stored '0' -> stays '0'");
  assert(off.ok === false, "stored '0' -> _vmServerSttOk === false");

  // 4. Absent key but server UNAVAILABLE — must fetch, write '0', stay off.
  const absentUnavailable = await runProbe({ storedValue: undefined, fetchReturns: false });
  assert(absentUnavailable.fetchCalled, 'absent key -> fetches even when server unavailable');
  assert(absentUnavailable.written === '0', "absent + unavailable -> writes '0'");
  assert(absentUnavailable.ok === false, 'absent + unavailable -> _vmServerSttOk === false');

  console.log('all probe-guard cases behave correctly (regression locked)');
})().catch((e) => { console.error('ERROR', e); process.exit(1); });
