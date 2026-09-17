# Ada TP=2 P2P Campaign — Full Session Report (2026-09-17)

**Session:** `87f5d6c1720c` ("Ada benchmarks"), Hermes WebUI, profile `default`
**Remote box:** `ada.vpn` (`mikaeljo@ada.vpn`) — Ryzen 7 5700X / 30 GiB RAM / Ubuntu 26.04 / kernel 7.0.0-31 / NVIDIA driver 615.71.09, **2× RTX 3090** (PCIe BDF `42:00.0` / `43:00.0`, each behind its own host bridge)
**Reference box:** vast.ai rental `49939153` "box A", run 2026-08-30/09-05 — 5600T, 2× 3090, `p2p=CNS`
**Model:** Qwen3.8-27B-W4A16-AutoRound (MORLEY tree) on **vLLM 0.27.1 + 43-patch tree** (HEAD `339a286`)
**Durable results:** `~/investigations/tp2-pcie-gen4` — commits `5f1936a` (raw outputs + findings), `ea76e03` (ledger pointer), `7158752` (full cohort matrix)

---

## 1. Campaign goal

The standing question (campaign `tp2-pcie-gen4`, plan §1): **does GPU P2P availability change
TP=2 throughput for Qwen3.8-27B on a PCIe-only, no-NVLink 3090 pair?**

The 08-30 rental run was the *lower bound*: P2P reported `CNS`, vLLM allreduce dispatched
`['PYNCCL']` only (host-staged). The ada campaign is the **first measured P2P-enabled
datapoint**: `nvidia-smi topo -p2p r/w` = `OK` on both GPUs (after the user's P2P-enable
hack), so vLLM dispatches `['CUSTOM','PYNCCL']` — direct GPU↔GPU DMA. This is the only
variable deliberately allowed to differ.

Everything else was held equal to 08-30: same model, same vLLM tree, same launch config
(TP=2, `max-model-len 256000`, fp8 KV, `--mamba-ssm-cache-dtype float16`,
`--language-model-only`, util 0.93, max-seqs 32, max-batched-tokens 2048,
`--async-scheduling`, MTP×3, port 18020, host 0.0.0.0).

Known residual difference (recorded, not hidden): PCIe **width** under load is x8 on ada vs
x16 on 08-30 (topology of that board). Both negotiate **Gen4 (16 GT/s)** under load —
the "Gen1 x8" reading sometimes reported by `nvidia-smi` is the idle P8 power-state
artifact, identical on both boxes.

---

## 2. What was done, in order

### 2.1 Session recovery

The session had been compacted mid-campaign (prior context: the 08-19 MORLEY venv probe,
the 08-30 rental bench, the ada bring-up story). Recovery re-anchored on session ID
`87f5d6c1720c` and re-verified the live state on ada (server, GPUs, logs) before touching
anything.

### 2.2 Environment repair — the venv brought over from morley was dead, for four stackable reasons

The MORLEY tree (`/home/mikaeljo/git/vllm.MORLEY.NOT_WORKING/qwen38-27b-rtx3090/`) was
rsynced from morley; on ada the venv was non-functional. Diagnosis and fixes:

1. **Stale venv interpreter.** `pyvenv.cfg` / `venv/bin/python` pointed at morley's
   `/usr/bin/python3.12`, which **does not exist on ada** (ada only has system 3.14).
   *Fix:* repointed the venv to **uv CPython 3.12.14**; original files backed up in
   `venv/bin/.orig_backup`.
2. **Dead entry-point shebangs** (`venv/bin/vllm` etc. still pointed at the morley
   python path). *Fix:* rewrote the shebangs (script `ada_shebang_fix.sh`).
3. **nvcc ↔ toolkit header skew.** The venv carried `nvidia-cuda-nvcc==13.3.73` +
   CCCL 13.3.3 + nvvm/crt 13.3, while its own `nvidia/cu13/include` headers were CUDA
   13.0 (`CUDART_VERSION 13000`). On a *cold* FlashInfer JIT cache the first real
   compile hits `cuda_toolkit.h:41: error: "CUDA compiler and CUDA toolkit headers are
   incompatible"` → `ninja: build stopped` → `Worker VllmWorker-1 died` at engine init.
   *Fix:* downgraded the six compiler-side `nvidia-*` packages to the 13.0 line matching
   torch 2.13.0+cu130's own dist-info pins (`ada_nvcc_downgrade.sh`).
   *Why morley never showed this:* its FlashInfer JIT cache is **warm** (built 08-30 when
   the toolchain was coherent), so it never recompiles. Ada's empty cache forced the first
   real compile → the skew detonated. This is very likely the true "NOT_WORKING" origin
   of the tree label.
4. **Empty `nvidia/cu13/lib64/`.** After the version check passed, the *link* step
   failed: `ld: cannot find -lcudart` (build.ninja hard-links
   `-L .../nvidia/cu13/lib64`; the wheel installs into `lib/`, and the `lib64` dir was
   empty in the copied venv). *Fix:* rebuilt `lib64/` as symlinks into `lib/` + a
   `libcuda.so` driver stub (`ada_lib64_fix.sh`); re-ran the exact failing link → `OK`.

   Post-repair versions: Python 3.12.14, torch 2.13.0+cu130, vLLM 0.27.1,
   flashinfer 0.6.16.post3, triton 3.7.1, nvidia-cuda-nvcc 13.0.x line. Both GPUs pass
   the CUDA smoke test; `vllm --version` clean.

### 2.3 Boot and gates

Launched via `~/ada_boot.sh` (`nohup setsid`). Boot gates matched the 08-30 reference:

| Gate | 08-30 | ada |
|---|---|---|
| Model load per rank | 7.49–7.67 GiB | **7.67 GiB** |
| Available KV per rank | ~13.14 GiB | **13.22 GiB** |
| GPU KV cache | ~779,200 tokens | **784,000 tokens** |
| FlashInfer JIT | warm cache | **compiled + linked from cold** |
| Allreduce dispatch | `['PYNCCL']` | **`['CUSTOM','PYNCCL']`** |
| `/health` | 200 | **200** |

First real request: correct output (`1..5`, `finish_reason: stop`), 33 s (cold — absorbed
one-time FlashInfer autotune + cudagraph capture). Warmup + clean measurement followed:
~10.2 tok/s e2e, 6.5 ms/step at C1. MTP per-position acceptance tracked healthy
(24–58% per window; a degenerate ~0% window in the *last* 1–2 windows of a run is the
known mid-generation-kill artifact, present on 08-30 too — not a real signal).

### 2.4 The bench suite

Driver `~/ada_bench.sh` mirrors the 08-30 `tp2_long` leg exactly:

1. **warmup** (3 prompts — discarded)
2. **cohort sweep** — 8 real prompts (`prompts_real.jsonl`) × 1024 output, T=default and
   T=0, at **C=1/2/4/8** (the 8 cohorts)
3. **1×100k-in / 256-out** long-context prompt (greedy)
4. **4×60k-in / 1024-out, conc 4** long-context mix (greedy)
5. **3× real_rep** — the MTP-acceptance-under-real-load measurement (greedy, C1)

**Hiccup, documented:** the first launch (07:28) ran but its output dir never existed
(diagnostics had been run locally on laplace instead of on ada; `mkdir` hit a
`Permission denied` on the wrong host). The run was killed, stale partial outputs
(`real_single_1.json` et al., + `bench_outputs/ada_20260917/`) were removed, and the
suite was **relaunched clean at 08:52** together with the capture companion
`~/ada_pcie_capture.sh`.

**Capture companion** (10 s cadence, 173 samples): `nvidia-smi` per-GPU link
gen/width/P-state/util + sysfs `current_link_speed`/`width` (BDF 42/43) →
`pcie_underload.log`; VRAM + `topo -p2p` readback → `vram_p2p.log`; plus the repo's
`mpstat`/`top` thread samplers into `tp2_results_256k/`.

Run window: **08:52–09:18 UTC**. GPU util 83–100% throughout; PCIe held **Gen4 x8,
P2** — no downclock events; VRAM stable ~23.9 GiB/rank.

---

## 3. Results — ada (P2P=OK) vs 08-30 (no-P2P), 256k leg

### 3.1 Standard cohorts (8 real prompts / 1024 out)

**T = default**

| Cohort | GEN ada | GEN 08-30 | Δ GEN | PP ada | PP 08-30 | TTFT ada | TTFT 08-30 |
|---|---|---|---|---|---|---|---|
| C1 | 140.37 | 106.20 | **+32%** | 14.02 | 15.33 | 86 ms | 122 ms |
| C2 | 241.90 | 189.27 | **+28%** | 24.23 | 26.46 | 118 ms | 175 ms |
| C4 | 429.00 | 303.32 | **+41%** | 42.83 | 43.35 | 158 ms | 254 ms |
| C8 | 627.05 | 422.46 | **+48%** | 62.71 | 61.24 | 387 ms | 712 ms |

**T = 0 (greedy)**

| Cohort | GEN ada | GEN 08-30 | Δ GEN | PP ada | PP 08-30 | TTFT ada | TTFT 08-30 |
|---|---|---|---|---|---|---|---|
| C1 | 146.50 | 122.50 | **+20%** | 14.68 | 17.79 | 78 ms | 128 ms |
| C2 | 258.27 | 198.71 | **+30%** | 25.87 | 28.89 | 117 ms | 169 ms |
| C4 | 461.68 | 307.96 | **+50%** | 46.38 | 44.71 | 142 ms | 252 ms |
| C8 | 684.35 | 461.70 | **+48%** | 68.51 | 67.00 | 392 ms | 714 ms |

GEN = output/generation tok/s (vLLM "Output token throughput"); PP = prefill tok/s
(Total − Output).

**Second keep-run the same day (server left warm):** C1/C2/C4/C8 default
**138.1 / 233.5 / 457.4 / 638.1** on ada vs 08-30's own second run
112.3 / 185.6 / 301.4 / 415.3 → **+23% / +26% / +52% / +54%**. Greedy set mirrors
(+22/31/50/47%). Pattern holds across runs.

### 3.2 Long-context legs

| Leg | GEN ada | GEN 08-30 | Δ | PP ada | PP 08-30 | TTFT ada | TTFT 08-30 |
|---|---|---|---|---|---|---|---|
| 1×100k in / 256 out | 3.33 | 2.98 | +12% | 1302.6 | 1163.7 | **74.3 s** | 83.5 s (−11%) |
| 4×60k in / 1024 out, conc4 | 21.56 | 18.49 | **+17%** | 1263.0 | 1083.2 | medITL 31.2 ms | 37.8 ms (−17%) |

### 3.3 MTP under real load (`real_rep`: 8 prompts × 1024 out, C1, greedy)

| | 08-30 (PYNCCL-only) | ada (CUSTOM+PYNCCL) |
|---|---|---|
| tok/step | 2.73 | **2.97 (+9%)** |
| ms/step per window | 22.5–24.3 (jittery) | **20.1–20.4 (<1.5% variance)** |
| e2e | 112–121 tok/s | **145.7–147.2 tok/s (+21%)** |
| server-side draft acceptance windows | same 56–79% regime | same 56–79% regime |

### 3.4 System state under load

- **PCIe** (10 s cadence, 173 samples): Gen4 x8, P2, util 83–100% — link never dropped.
  08-30 reference: Gen4 x16, P0, 100%.
- **CPU** (`mpstat` `_all` avg): ada sys 3.2% / idle 82.1%; 08-30 sys 1.1% / idle 93.8%.
  Neither CPU-bound; ada's higher sys is consistent with the live DMA path doing real
  work the host-staged NCCL path skips.
- **VRAM**: ~23.9 GiB used/rank, stable all run (fp8 KV budget ~13.2 GiB/rank).

### 3.5 Interpretation

- **Generation throughput wins everywhere, widening with concurrency** (+20–54%): the
  signature of the P2P custom-allreduce lever — more concurrent requests = more
  collectives per step = more benefit from direct GPU↔GPU DMA.
- **Prefill is tied at C1–C8** (14–68 tok/s both sides) — expected: the real prompts are
  tiny (~800–1100 tokens), and small-batch prefill is GEMM-bound, not collective-bound.
- **Prefill *does* win on the long-context legs** (+12–17%): 100k-prompt prefill with
  MTP draft forward passes is communication-amplified.
- **TTFT drops ~30–45%** across cohorts (86 vs 122 ms at C1; 387 vs 712 ms at C8).
- **Run-to-run variance collapses** (the real_rep ms/step spread) — P2P removes the
  host-staging hop that made the no-P2P box jitter.

---

## 4. Operational artifacts left on ada (in `~/`)

| Script | Purpose |
|---|---|
| `ada_boot.sh` | Background boot (exact 08-30 config), log → `ada_server.log` |
| `ada_run_fg.sh` | **Foreground runner** — live log on screen, `tee -a` to `ada_server.log`, port-18020 guard, Ctrl+C graceful stop |
| `ada_bench.sh` | Full suite driver (warmup → 8 cohorts → long legs → 3× real_rep) |
| `ada_pcie_capture.sh` | 10 s PCIe/VRAM/P2P capture companion |
| `ada_nvcc_downgrade.sh`, `ada_lib64_fix.sh`, `ada_shebang_fix.sh`, `ada_repoint*.sh` | the environment-repair chain |
| `ada_nvcc_diag.sh`, `ada_pypi_check.sh`, `ada_skew_dir.sh`, `ada_cudart_fix.sh`, `ada_ar_check.sh`, `ada_warmup.sh`, `ada_firstreq.sh` | diagnostics used along the way |

**Manual stop / start:**

```bash
pkill -f "vllm serve"                 # stop; wait for VRAM to drop
bash ~/ada_run_fg.sh                   # start in foreground with live tee'd log
```

The server was left **running** at end of session (pid 71275, `:18020`, serving).

---

## 5. Deliverables & where they live

- **This report** — `/home/hermes/webui/ada-tp2-2026-09-17-report.md` (repo root).
- **Durable campaign repo** `~/investigations/tp2-pcie-gen4`:
  - `findings/ada-20260917-bench.md` — the finding, incl. full cohort matrix +
    environment-repair record (commits `5f1936a`, `7158752`)
  - `local_ada_20260917/` — all raw outputs: `tp2_results_256k/` (cohort `.log`s,
    `real_single_1.log`, `real_rep1-3.log`, `real_60k_batch4.log`, `bench_run.out`,
    `real_run_summary.out`, `pcie_underload.log`, `vram_p2p.log`, `mpstat.log`,
    `top_thread.log`) + `ada_server_20260917.log` (contains the allreduce dispatch
    line and SpecDecoding acceptance windows)
  - `ledger/notes.md` — pointer entry
  - 08-30 reference: `rental_49939153_boxA/tp2_results_256k/`
- **Session scripts** — `/tmp/ada_*.sh` on laplace (canonical copies on ada in `~/`).

---

## 6. Open items / honest caveats

1. **Confounded variable: link width.** ada is Gen4 **x8**, 08-30 was Gen4 **x16**.
   Part of the delta may be the P2P DMA path, part the width change. The tok/step gain
   and variance collapse are the robust parts; the absolute e2e delta should be read
   with the caveat. Clean control = a box that is **x16 *and* P2P-OK** (or an ada run
   with the P2P hack disabled to get the ada-side no-P2P bound).
2. **Tokenizer difference across the two vLLM builds**: same 8 prompts tokenize to
   1130 vs 794 total input tokens (chat-template version), so cross-box **PP**
   compares directionally, not exactly; **GEN** (≈7.9k output tokens both sides) is
   like-for-like.
3. The venv is the repaired MORLEY tree, **not** a from-scratch rebuild (the 43-patch
   application stays delicate; a cold FlashInfer JIT now works, so a rebuild *would*
   succeed, but the repair is the documented proven state).
4. `ada_boot.sh` still exports the old 13.3-line `CUDA_HOME`/`LD_LIBRARY_PATH` paths —
   harmless today (JIT cache is warm, 13.0 line is coherent) but the boot script should
   be regenerated from `ada_run_fg.sh`'s env block for a fresh-machine reinstall.
5. The 5700X/30 GiB box is a **low-end** consumer platform (vs 08-30's 5600T + 64 GiB
   rental tier): ada's slightly higher CPU sys% and lower RAM headroom are worth
   watching if larger cohorts (C16+) are added.
