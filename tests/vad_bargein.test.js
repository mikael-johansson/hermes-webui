// Decision-logic test for voice-mode barge-in (static/boot.js _bargeTick).
// Mirrors the thresholds there; run with:  node tests/vad_bargein.test.js
// Exits non-zero if any case misbehaves.
// Replicates _bargeTick's decision logic exactly, driven by synthetic frames.
const FRAMES=8, MULT=3.5, MIN=0.015;
function run(seq){
  let hot=0, noise=0.01, firedAt=null;
  seq.forEach((rms,i)=>{
    if(firedAt!==null) return;
    const th=Math.max(noise*MULT,MIN);
    if(rms>th){ if(++hot>=FRAMES){ firedAt=i; } }
    else { hot=0; noise=noise*0.95+rms*0.05; }
  });
  return firedAt;
}
const rep=(v,n)=>Array(n).fill(v);
const cases=[
  ["silence only (should never fire)",            rep(0.002,300), null],
  ["quiet room hum (should never fire)",          rep(0.008,300), null],
  ["speech after 100 frames of silence",          [...rep(0.002,100),...rep(0.12,20)], 107+100-100+100],
  ["short cough: 4 frames (should NOT fire)",     [...rep(0.002,100),...rep(0.3,4),...rep(0.002,50)], null],
  // A loud room must not itself fire, but real speech over it must.
  ["loud room, no speech (should NOT fire)",      rep(0.02,300), null],
  ["loud room, then speech (SHOULD fire)",        [...rep(0.02,200),...rep(0.25,20)], 207],
  // Residual TTS leaking past echo cancellation sits just above the floor and
  // must not trip the interrupt.
  ["echo leak just over floor (should NOT fire)", [...rep(0.002,60),...rep(0.012,200)], null],
];
let bad=0;
for(const [name,seq,want] of cases){
  const got=run(seq);
  const ok = (want===null) ? (got===null) : (got!==null);
  if(!ok){ bad++; console.log(`  FAIL ${name}: fired at ${got}`); }
  else console.log(`  ok   ${name}${got!==null?` (fired frame ${got}, ~${Math.round(got*16.7)}ms into speech-start)`:''}`);
}
// explicit latency check
const seq=[...rep(0.002,100),...rep(0.12,20)];
const at=run(seq);
console.log(`\n  trigger latency after speech onset: ${at-100} frames ≈ ${Math.round((at-100)*16.7)} ms`);
console.log(bad? `${bad} FAILURES` : "all VAD cases behave as intended");
process.exit(bad?1:0);
