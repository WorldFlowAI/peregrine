# Peregrine — Internal Benchmark Framework Plan

> Goal: a reproducible, contamination-resistant harness that proves Peregrine's
> performance claims and gives us a **standing baseline against llama.cpp** — at
> the kernel level today, end-to-end the moment a model runs.

This is a design + rollout plan, not yet code. It is deliberately strict about
methodology because almost every "X is faster than Y" inference benchmark on the
internet is wrong for a measurable, avoidable reason (warm-vs-cold confounds,
different quant, different sampling, thermal throttling, no statistics). We don't
ship a number we can't defend.

---

## 1. Why now, and what we can actually measure

Peregrine today has hand-written kernels (dot; soon rmsnorm/softmax/rope/gemm/
attention) but **no end-to-end engine** (model load + decode lands in M3). So the
framework is phased to match reality:

| Phase | When | Compares | Headline metric |
|-------|------|----------|-----------------|
| **P0 kernel** | now (M1–M2) | a Peregrine kernel vs ggml's equivalent op | GFLOP/s, GB/s, cycles/elem |
| **P1 end-to-end** | M3+ | `peregrine run` vs `llama-cli`/`llama-bench` on the *same* GGUF | prefill tok/s, decode tok/s, TTFT |
| **P2 sweep** | M4+ | both, across models × quant × threads × ISA × hardware | speedup surface + Pareto |

P0 starts immediately: it gives early, honest baselines and catches regressions
while the engine is still being built. It reuses the checkasm bench plumbing we
already have (`--bench`, the timing table, the seedable RNG).

## 2. The non-negotiable fairness contract

A comparison is only valid if **everything except the implementation under test
is identical**. Every benchmark run records and asserts these:

- **Same weights.** Byte-identical model file (P1) or the same dequantised
  reference tensors (P0). No "Peregrine fp16 vs llama.cpp Q4" cross-quant
  apples-to-oranges.
- **Same quantisation + tokenizer.** GGUF carries both; load from the same file.
- **Same numerics contract.** Greedy / temperature 0 / fixed seed so output is
  deterministic and we compare *equal work*, not "one got luckier with sampling."
- **Same prompt set.** A fixed, versioned prompt corpus (varying lengths) checked
  into `bench/corpus/`. Report prefill and decode separately — they have totally
  different bottlenecks (compute-bound vs memory-bound).
- **Same hardware state.** Pinned threads, fixed thread count, turbo/governor
  fixed, thermals settled (see §5). Record CPU model, microcode, RAM, OS.
- **Same compiler tier, pinned competitor commit.** llama.cpp built at a recorded
  SHA with comparable `-O3 -march=native`; we record both build flags.

The harness writes all of this into every result record (§4). A result missing
any field is invalid and rejected by the reporter.

## 3. Metrics taxonomy

**Kernel (P0):**
- Throughput: **GFLOP/s** and **GB/s** (a dot is memory-bound — GB/s is the
  honest metric; a GEMM is compute-bound — GFLOP/s is). Always report both so
  the bottleneck is visible.
- **cycles/element** where a cycle counter is available (x86 `rdtsc`/`rdpmc`;
  ARM PMU via `perf` on Linux — *not* available from userspace on macOS, where
  we fall back to ns and say so).
- **Working-set regimes**, reported as distinct rows: L1-resident, L2-resident,
  out-of-cache. The same kernel can be 12× over scalar in L1 and 9× out of cache
  (we already see exactly this for dot) — collapsing them hides the story.

**End-to-end (P1):**
- **Prefill throughput** (tok/s) at several context lengths.
- **Decode throughput** (tok/s) — the number users feel.
- **TTFT** (time to first token) — and measure it *correctly*: cold vs warm KV /
  weights changes it by 10×; report both states explicitly, never one unlabeled.
- **Latency distribution**: p50 / p99 inter-token latency, not just the mean.
- **Peak RSS** and model load time. **Energy** (optional, via `powermetrics` /
  RAPL) once the basics are solid.

## 4. Harness architecture

```
bench/
  corpus/              versioned prompt sets (short / medium / long / batch)
  runners/
    peregrine.sh       run our engine, emit result JSON
    llamacpp.sh        build@pinned-SHA + run llama-bench, emit result JSON
  kernels/             P0 microbench drivers (reuse checkasm bench plumbing)
    bench_dot.c        ... bench_gemm.c, bench_attention.c
  compare.py           load JSONs, assert fairness fields, print tables + speedup
  results/             committed JSON history (one file per run, git-tracked)
  env-capture.sh       CPU/microcode/RAM/OS/thermal snapshot -> JSON
```

- **One result = one JSON record** with: `{impl, version/sha, kernel|model,
  params, metric values, env snapshot, seed, timestamp}`. Machine-readable so we
  can diff across commits and plot trends.
- **`compare.py`** is the gate: it refuses to compare records whose fairness
  fields (model hash, quant, thread count, host) differ, and prints a speedup
  table with confidence intervals.
- **Regression tracking:** `results/` is committed. A CI job runs the kernel
  benches on a fixed runner and flags a >X% regression vs the last commit's JSON
  (kernel benches are stable enough on a pinned machine; end-to-end is noisier
  and runs nightly, not per-PR).

## 5. Methodology controls (how we avoid lying to ourselves)

- **Warmup discarded.** First N iterations prime caches/branch predictors/clocks;
  only steady-state is measured. (The checkasm bench already warms up once;
  P1 needs a few warm decode steps.)
- **Repeat + robust statistics.** ≥ k runs; report **median + MAD**, not mean
  (resists thermal/scheduler outliers). Flag runs whose spread exceeds a
  threshold rather than silently averaging them.
- **Pinning & clocks.** `taskset`/`pthread_setaffinity` to fixed cores; disable
  turbo or fix frequency where possible; record the governor. On Apple Silicon,
  pin to P-cores and note that frequency isn't user-controllable (so we lean on
  median-of-many + thermal settling).
- **Thermal settling.** Idle to a baseline temperature before a run; abort/redo
  if throttling is detected mid-run.
- **Cache-state control for kernels.** Explicitly choose cold (flush / stream
  fresh buffers) vs warm, and *label* it. Never report a warm in-L1 number as if
  it were representative of real workloads.
- **No dead-code elimination.** Sink results to a `volatile` (already done) so
  the compiler can't delete the work being timed.
- **Isolation.** One workload at a time; no other heavy processes; ideally a
  dedicated bench host. (Mirrors the discipline in our vLLM/prefix-cache
  benchmark-isolation experience — contamination from a shared box is the single
  most common silent error.)

## 6. The llama.cpp head-to-head, concretely

llama.cpp ships two tools we should both *use as the comparison target* and
*learn from*:

- **`llama-bench`** — the canonical end-to-end benchmark (prefill `pp` + decode
  `tg` tok/s, with proper warmup and repetition). This is our P1 reference: same
  GGUF, same thread count, same prompt/gen lengths; we run `llama-bench` and our
  engine through the identical matrix and diff the JSON.
- **`test-backend-ops`** — ggml's per-op correctness+perf harness. It is the
  direct analogue of our checkasm and the right model for P0: for each op (mul_mat,
  rms_norm, rope, soft_max, …) it fuzzes correctness and reports GFLOP/s. For P0
  we benchmark Peregrine's kernel against ggml's same op on identical tensors.

Build discipline: pin llama.cpp to a recorded commit, build with a documented
flag set, record `system_info` (which ISA paths ggml actually selected — AVX2 vs
AVX-512 vs NEON), and store it alongside our result. A speedup claim names the
exact llama.cpp SHA and build flags or it doesn't ship.

## 7. Rollout, mapped to the roadmap

1. **P0.1 (now):** generalise the checkasm `--bench` table into a standalone
   `bench/kernels/` driver that emits JSON; add an env-capture step. Baseline
   `dot` vs `ggml_vec_dot` on identical buffers.
2. **P0.2 (M2):** add GEMM + attention kernel benches vs ggml `mul_mat` /
   flash-attn; introduce `compare.py` + `results/` regression tracking in CI.
3. **P1 (M3):** once `peregrine run` decodes a model, wire `runners/` + a fixed
   model/quant/prompt matrix vs `llama-bench`; report prefill/decode/TTFT with
   median+MAD and cold/warm labelling.
4. **P2 (M4+):** sweep models × quant × threads × ISA × hardware; publish a
   reproducible results page (numbers + the exact commands + env to regenerate).

## 8. Pitfalls we explicitly refuse to commit

- Reporting a single mean instead of a distribution.
- Comparing across different quant levels, thread counts, or prompt sets.
- Unlabeled warm-vs-cold TTFT (the classic 10× confound).
- Benchmarking on a shared/thermally-unsettled box.
- Letting the compiler elide the timed work.
- Quoting a speedup without the competitor's exact commit + build flags + the
  ISA path it actually ran.

Every one of these is a known way these benchmarks go wrong; the harness is
designed so that doing the right thing is the path of least resistance.
