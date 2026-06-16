# peregrine — Roadmap

Milestones are vertical slices: each ends with something runnable and tested,
not a half-built layer. "Done" always includes a passing checkasm entry for any
new kernel.

## M2 — GEMM  (in progress)
Register-blocked, cache-tiled f32 GEMM/GEMV microkernels per ISA; bf16/fp16
paths; threadpool + tiled parallel driver.

- [x] f32 GEMM C reference + checkasm (double oracle, reduction-aware tolerance)
- [x] Register-blocked SGEMM microkernels per ISA — NEON 8x8, x86 AVX2 6x16
- [x] Threadpool (`util/thread`) + parallel driver (partition over row-blocks)
- [x] GEMV per ISA — threaded dot-per-row over the existing per-ISA dot kernels
      (a GEMV row is a dot; reuse beats a redundant microkernel)
- [~] Cache tiling — measured unnecessary on Apple Silicon (compute-bound through
      2048^3); deferred to hardware where the working set spills, behind a
      C-accumulating microkernel
- [~] bf16/fp16 paths
      - [x] bf16 storage path (bf16->f32 convert in pack, reuse f32 microkernel)
            — portable fallback + correctness reference
      - [x] native bf16 compute microkernel — ARM BFMMLA (8x8, 16 MAC/instr).
            Correct + tested. Measured SLOWER than f32 FMLA on Apple (matrix
            throughput is in AMX, not NEON BFMMLA); wins on full-rate-BFMMLA
            server cores. Kept tested/benched; not auto-selected on Apple.
      - [ ] native bf16 on x86 (AVX512-BF16 VDPBF16PS)
      - [ ] fp16 path
- [ ] Benchmark vs OpenBLAS / cuBLAS-on-CPU baselines

Measured (Apple Silicon, 12-core): f32 SGEMM 105 GFLOP/s single-core ->
825 GFLOP/s @ 2048^3 (~31x scalar C). x86 AVX2 paths correctness-validated on CI
(Intel SDE).

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
