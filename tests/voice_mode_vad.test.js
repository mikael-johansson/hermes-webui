// Decision-logic test for the voice-mode SERVER-STT engine's onset/silence VAD
// (static/boot.js _vmVadTick). This is the new capture path used when the Hermes
// server confirms a local STT provider is available: it records with
// MediaRecorder and transcribes via /api/transcribe, so it has NO dependency on
// the browser's cloud SpeechRecognition backend (the one that silently fails
// behind corporate/VPN networks and leaves the UI stuck on "Listening…").
//
// The VAD must (a) detect SPEECH ONSET (fire on real speech, never on silence
// alone) and (b) emit a STOP after a silence gap following speech, so the clip
// is transcribed + auto-sent once the user pauses.
//
// Design note (why the baseline is speech-guarded): a naive floor that updates
// on EVERY frame (the barge-in rule) chases a sustained utterance — the floor
// converges to the speech level, the threshold (floor*3.5) climbs above the
// signal, and onset never latches. So this VAD only updates the ambient baseline
// while the signal is QUIET. Same family as barge-in (adaptive floor, 3.5x
// multiplier, 16.7ms frames) but speech-guarded, which is what makes onset +
// gap-stop reliable.
//
// This file replicates _vmVadTick's decision body EXACTLY and drives it with
// synthetic 16.7ms frames. The Python source-test separately asserts boot.js
// still contains these constants/behavior so the two can't drift. Run with:
//   node tests/voice_mode_vad.test.js
// Exits non-zero if any case misbehaves.

const ONSET=8;                 // consecutive voice frames -> speech started (~133ms)
const SILENCE=12;              // consecutive quiet frames after onset -> stop (~200ms)
const MIN=0.015, MULT=3.5;     // threshold = max(baseline*3.5, 0.015)

function makeState(){ return { noise:0.015, onset:0, silence:0, started:false }; }

// Faithful replica of static/boot.js _vmVadTick(rms).
function _vmVadTick(rms, st){
  const th=Math.max(st.noise*MULT, MIN);
  const quiet=rms<=th;
  if(!st.started){
    if(!quiet){ if(++st.onset>=ONSET){ st.started=true; st.silence=0; } }
    else { st.onset=0; st.noise=st.noise*0.95+rms*0.05; } // track ambient only
  }else{
    if(quiet){ if(++st.silence>=SILENCE) return 'stop'; st.noise=st.noise*0.95+rms*0.05; }
    else st.silence=0; // still talking — do NOT let the floor chase the utterance
  }
  return st.started?'speaking':'waiting';
}

function run(seq){
  const st=makeState();
  for(let i=0;i<seq.length;i++){ if(_vmVadTick(seq[i],st)==='stop') return i; }
  return null;
}

const rep=(v,n)=>Array(n).fill(v);
const cases=[
  // Pure silence must NEVER stop the clip. (The old browser-only engine's
  // failure mode was the opposite: it cycled "Listening…" forever without ever
  // transcribing, because SR produced no results and onend just re-armed it.)
  ["silence only (NEVER fire)",                rep(0.002,400), null],
  ["room hum below adaptive floor (NEVER fire)",rep(0.01,400),  null],
  // Real speech, then a genuine ~200ms pause -> engine stops right after the pause.
  ["speech then ~200ms pause (SHOULD fire)",
      [...rep(0.002,40), ...rep(0.12,150), ...rep(0.002,15)], 40+150+11],
  // A short cough (below the 8-frame onset requirement) is not a real turn.
  ["short cough, no sustained speech (NOT fire)",
      [...rep(0.002,40), ...rep(0.3,4), ...rep(0.002,300)], null],
  // Speech, a brief sub-gap pause, more speech -> ONE clip, stops after final pause.
  ["speech, brief pause, more speech (fires once, after final pause)",
      [...rep(0.002,40), ...rep(0.12,60), ...rep(0.002,6), ...rep(0.12,40), ...rep(0.002,15)],
      40+60+6+40+11],
  // Loud ambient noise must not itself start a clip, but speech over it must.
  ["loud room, then speech (SHOULD fire)",
      [...rep(0.02,200), ...rep(0.25,20), ...rep(0.002,15)], 220+11],
];

let bad=0;
for(const [name,seq,want] of cases){
  const got=run(seq);
  const ok=(want===null)?(got===null):(got!==null&&Math.abs(got-want)<=2);
  if(!ok){ bad++; console.log(`  FAIL ${name}: stopped at ${got}, want ${want}`); }
  else console.log(`  ok   ${name}${got!==null?` (stopped frame ${got}, ~${Math.round(got*16.7)}ms)`:''}`);
}

// Explicit latency sanity: onset should latch ~8 frames into speech, and the
// clip should stop ~12 frames after the speech ends.
const latSeq=[...rep(0.002,100), ...rep(0.12,50), ...rep(0.002,15)];
const latSt=makeState();
let firedAt=null, stoppedAt=null;
for(let i=0;i<latSeq.length;i++){
  const r=_vmVadTick(latSeq[i],latSt);
  if(firedAt===null&&latSt.started) firedAt=i;
  if(r==='stop'){ stoppedAt=i; break; }
}
const onsetLat=firedAt-100, stopLat=stoppedAt-(100+50);
console.log(`\n  speech-onset latency: ${onsetLat} frames ≈ ${Math.round(onsetLat*16.7)} ms`);
console.log(`  stop latency after speech end: ${stopLat} frames ≈ ${Math.round(stopLat*16.7)} ms`);
if(onsetLat<6||onsetLat>10){ bad++; console.log(`  FAIL onset latency ${onsetLat}`); }
if(stopLat<10||stopLat>14){ bad++; console.log(`  FAIL stop latency ${stopLat}`); }
console.log(bad? `${bad} FAILURES` : "all voice-mode VAD cases behave as intended");
process.exit(bad?1:0);
