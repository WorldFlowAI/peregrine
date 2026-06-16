# peregrine — Roadmap

Milestones are vertical slices: each ends with something runnable and tested,
not a half-built layer. "Done" always includes a passing checkasm entry for any
new kernel.

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
