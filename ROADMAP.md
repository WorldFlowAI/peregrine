# peregrine — Roadmap

Milestones are vertical slices: each ends with something runnable and tested,
not a half-built layer. "Done" always includes a passing checkasm entry for any
new kernel.

## M1 — Kernel foundation
- ✅ `x86inc.asm` macro layer (cglobal / ABI / auto-vzeroupper) + ARM `asm.S`
  macros, vendored from dav1d (licenses kept in `src/ext/`); `dot` refactored
  onto them.
- ✅ Install Meson and verify — native arm64 build + `meson test` green (see
  the Apple Silicon / Rosetta-Python note in README).
- ✅ checkasm framework: per-variant fuzzing (random size/alignment/magnitude +
  edge sizes), `--bench` throughput table with speedups, `--seed`/`--test`.
- ✅ Register-clobber check (callee-saved preservation via asm trampolines):
  AArch64 done + tested (proven to catch a real clobber); x86-64 SysV done +
  assembled-verified. *TODO:* Win64 ABI (rsi/rdi/xmm6-15) + runtime-verify x86
  on real hardware/CI.
- ✅ Detailed plan for the internal benchmark framework + llama.cpp head-to-head
  baseline — see `doc/benchmarking-plan.md` (phased: kernel-level now,
  end-to-end at M3).
- Primitive kernels (C ref + NEON + AVX2 + checkasm + bench):
  - ✅ `axpy` (NEON ties auto-vectorised C — memory-bound trivial loop, no asm edge),
    `rmsnorm` (NEON ~6× over C; fused two-pass + rsqrt). NEON fuzz+clobber-tested,
    AVX2 assembled-verified.
  - TODO non-transcendental: `sum`/`max` reductions, elementwise `mul`/`add`
    (mechanical copies of dot/axpy).
  - TODO transcendental: `softmax`, `silu`/`gelu`, `rope` — these need a
    hand-written vectorised `exp` / `sincos` (polynomial + range reduction);
    worth its own focused, accuracy-validated task. C references can land first
    so dispatch works while the SIMD paths follow.
- ✅ CI matrix (`.github/workflows/ci.yml`): build + checkasm on x86-64 (native +
  Intel SDE forcing SSE4/AVX2/AVX-512) and AArch64 (native NEON + QEMU `-cpu max`
  for SVE2 as it lands).

## M2 — GEMM
- Register-blocked, cache-tiled f32 GEMM/GEMV microkernels per ISA.
- bf16/fp16 paths. Benchmark vs OpenBLAS/cuBLAS-on-CPU baselines.
- Threadpool (`util/thread`) + tiled parallel driver.

## M3 — Run a model (the "it works" moment)
- `model/`: mmap GGUF + safetensors loaders (zero-copy tensor views).
- `token/`: BPE / SentencePiece tokenizer.
- `graph/`: executor for a Llama-class decoder, contiguous KV cache.
- `sample/`: greedy + top-k/top-p/temperature.
- `peregrine run -m model.gguf -p "..."` produces tokens. Target: one 7B model,
  f32/fp16, correct output.

## M4 — Quantization & decode speed
- Quantized matmul: Q4_K/Q8_0 (GGUF parity) + fp8, dequant fused into the
  microkernel. Decode-loop arena allocation (no malloc per token).
- Fused FlashAttention-style attention with GQA/MQA + causal mask.
- Publish tokens/sec vs llama.cpp on identical models/hardware.

## M5 — Breadth & surface
- Split `tensor/kernels/` into a standalone tensor library (libavcodec-style).
- Stable public engine ABI (`pg_model_load` / `pg_eval` / `pg_sample`).
- MoE (grouped GEMM + routing). More architectures (Qwen, Mixtral).
- Paged KV cache + prefix sharing (hook for semantic-KV reuse).

## M6 — Accelerator backends
- CUDA/PTX backend behind the existing DSP seam (datacenter perf flagship).
- Metal backend (Apple GPU). Optional HTTP serving front-end
  (OpenAI-compatible), the `ffserver` analog.

---

### Cross-cutting, always-on
- Every kernel: C ref + checkasm before any asm merges.
- No intrinsics in `kernels/`.
- Benchmarks published per milestone — performance claims are reproducible or
  they don't count.
