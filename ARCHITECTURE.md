# peregrine — Architecture

> FFmpeg for AI inference: a C core with hand-written assembly, extreme
> performance, broad hardware support, run from a single CLI or linked as a
> library.

This document is the durable design. For *what ships when*, see
[ROADMAP.md](ROADMAP.md). For *how to add a kernel*, see
[doc/writing-asm.md](doc/writing-asm.md).

---

## 1. Thesis

The fast inference engines today fall into two camps:

- **Ease-of-use first** (llama.cpp): C++ with compiler *intrinsics*, huge
  community, runs everywhere — but the compiler still owns register allocation
  and scheduling, leaving performance on the table.
- **Vendor-locked peak** (TensorRT, cuBLAS): fast on one vendor's silicon,
  closed, not portable.

`peregrine` takes FFmpeg's path instead: **a clean C reference for every operation,
plus hand-written assembly per ISA, selected at runtime.** No intrinsics in the
hot path — we own the registers, the scheduling, and the tail handling. The
payoff is FFmpeg's: portable *and* faster than anything that lets a compiler
drive the vector units.

The closest modern blueprint is **dav1d** (VideoLAN's AV1 decoder): pure C +
NASM (x86) + GAS (ARM), a `checkasm` harness, runtime dispatch. We borrow its
structure wholesale and aim it at transformers.

## 2. The FFmpeg / dav1d mapping

| FFmpeg / dav1d            | peregrine                          | Role                                   |
|---------------------------|--------------------------------|----------------------------------------|
| `libavutil`               | `util/`                        | CPU detect, aligned mem, threads, log  |
| `libavcodec` DSP + asm    | `tensor/kernels/`              | hand-written compute kernels           |
| codec context / decode    | `graph/`                       | model graph, executor, KV cache        |
| `libavformat` (demux)     | `model/`                       | GGUF / safetensors loaders             |
| (n/a)                     | `token/`                       | tokenizers (BPE/SPM)                   |
| `ffmpeg` CLI              | `cli/` → `peregrine`               | run a model end-to-end                 |
| `checkasm`                | `tests/checkasm/`              | asm-vs-C correctness + microbench      |

## 3. Layered design

```
            ┌─────────────────────────────────────────────┐
   CLI /    │  peregrine run -m model.gguf -p "..."           │   cli/
   embed    │  libperegrine public C ABI (include/peregrine/)     │
            ├─────────────────────────────────────────────┤
   engine   │  graph executor · KV cache · sampler         │   graph/
            ├─────────────────────────────────────────────┤
   model    │  GGUF / safetensors loaders · tokenizer      │   model/ token/
            ├─────────────────────────────────────────────┤
   kernels  │  DSP structs of fn-ptrs  (matmul, attn, …)   │   tensor/kernels/
            │   ├ C reference  (always correct)            │
            │   ├ x86: sse2 / avx2 / avx512  (NASM)        │
            │   └ arm: neon / sve2           (GAS .S)      │
            ├─────────────────────────────────────────────┤
   util     │  cpu flags · aligned alloc · threadpool      │   util/
            └─────────────────────────────────────────────┘
```

**Strict dependency rule:** arrows point down only. Kernels never call into the
graph; util never calls up. This keeps `tensor/kernels/` extractable as a
standalone tensor library (a real possibility — see Roadmap M5).

## 4. The dispatch mechanism (the heart of the project)

Every kernel domain follows one shape. `dot` is the reference implementation
([src/tensor/kernels/dot/](src/tensor/kernels/dot/)); copy it.

1. **C reference** `pg_<op>_c(...)` — obvious, unvectorised, the correctness
   oracle. Never optimised.
2. **Assembly variants** `pg_<op>_<isa>(...)` — one per ISA extension, under
   `x86/*.asm` (NASM) and `arm/*.S` (GAS). No intrinsics.
3. **A DSP struct** of function pointers + `pg_<op>_dsp_init(dsp, cpu_flags)`
   that starts at the C reference and upgrades to the best routine the host
   supports.

```c
PgDotDSP dsp;
pg_dot_dsp_init(&dsp, pg_get_cpu_flags());
float s = dsp.dot_f32(a, b, n);   // callers never branch on CPU features
```

Arch is resolved **twice**:

- **Compile time** (`util/arch.h`, `PG_ARCH_*`): which asm files and dispatch
  branches are compiled in at all. Other ISAs cost zero bytes.
- **Run time** (`util/cpu.h`, `PG_CPU_*`): which of the compiled-in routines
  this specific CPU may execute. One binary, SSE2→AVX-512 and NEON→SVE2.

This is exactly `av_get_cpu_flags()` + per-codec `dsp_init`.

## 5. Hardware targets

First-class at launch — **x86-64** and **ARM (AArch64)** CPU:

| Arch    | Extensions (priority order)            | Assembler | Notes                          |
|---------|----------------------------------------|-----------|--------------------------------|
| x86-64  | SSE2 → AVX2+FMA → AVX-512 (+VNNI/AMX)  | NASM      | `x86inc.asm` macro layer (M2)  |
| AArch64 | NEON → SVE2 (+ Apple AMX via kernels)  | GAS `.S`  | NEON is the mandatory baseline |

Later, as *backends* behind the same DSP-struct seam (not v0.1):
**CUDA/PTX** (the perf flagship for datacenter), **Metal** (Apple GPU). The
function-pointer indirection means a GPU backend slots in without touching the
graph layer.

## 6. Kernel catalog & priority (LLM-first)

Ordered by impact on a decoder transformer's runtime. Each is a `dot`-shaped
domain with C ref + asm + checkasm.

1. **GEMM / GEMV** (f32, then bf16/fp16) — dominates prefill. The hard,
   high-payoff kernel: register-blocked microkernels, cache tiling.
2. **Quantized matmul** — q4/q8/fp8 weight × f32 activation. Decode is
   weight-bandwidth-bound; this is where llama.cpp wins and where we must too.
3. **Attention** — fused scores·softmax·value (FlashAttention-style, online
   softmax), GQA/MQA, causal masking. KV-cache aware.
4. **RoPE**, **RMSNorm/LayerNorm**, **SwiGLU/SiLU/GELU**, **softmax**,
   **elementwise** — smaller but ubiquitous; cheap wins.
5. **MoE** routing + grouped GEMM.
6. **dot / axpy / reductions** — primitives (the worked example).

## 7. Memory & threading

- **Aligned everything** (`PG_ALIGN = 64`): covers AVX-512 loads and ARM cache
  lines. All tensor buffers route through `pg_aligned_alloc`.
- **Arena allocation** for activations (planned): per-token scratch reset each
  step, no malloc in the decode loop.
- **Threadpool** (planned, `util/thread`): static work-stealing pool; kernels
  expose a tiled, partitionable form so the executor parallelises rows/heads.
- **KV cache**: contiguous per-layer ring with paged-attention-style block
  allocation on the roadmap (your semantic-KV work plugs in here).

## 8. Quantization

Weight formats target GGUF parity first (Q4_K, Q8_0, …) so existing model files
just work, plus fp8 for newer hardware. Dequant is fused into the matmul
microkernel (never a separate pass) — the whole point of hand asm is keeping
dequantised weights in registers.

## 9. Model loading & tokenizer

- **`model/`**: zero-copy `mmap` of GGUF / safetensors; a tensor is a view
  (ptr + shape + dtype + strides) into the mapped file. Endian/Dtype validated
  on load.
- **`token/`**: BPE and SentencePiece-style tokenizers, loaded from the model
  file's embedded vocab where present.

## 10. Public ABI & versioning

- Stable **C ABI** (`include/peregrine/`), `pg_` prefix, semver. Opaque structs +
  accessor functions so the ABI survives internal change.
- SONAME bump only on incompatible change. Embedders link `libperegrine`; the CLI
  is a thin client of the same ABI.

## 11. Testing philosophy

Non-negotiable for hand asm: **every assembly routine is proven equal to the C
reference.** `tests/checkasm/` fuzzes inputs across edge sizes (0, sub-vector,
odd tails, large) and microbenchmarks each variant. CI runs the matrix across
ISA levels (using QEMU / SDE to exercise extensions the runner lacks). A kernel
without a passing checkasm entry does not merge.

## 12. Build

- **Canonical: Meson + Ninja** (`meson.build`) — first-class NASM, integrated
  `.S`, `meson test` wires checkasm into `ninja test`. Chosen over a custom
  configure because contributors already know it (dav1d uses it).
- **Bootstrap: `Makefile.bootstrap`** — host-arch only, C compiler + nasm, zero
  other deps. Proves the worked kernel without installing a build system.

## 13. Design principles

- **No intrinsics in the hot path.** Hand asm or the C reference — nothing in
  between. (Intrinsics are fine in throwaway prototypes, never in `kernels/`.)
- **C reference is sacred.** Always present, always correct, never optimised.
- **Dispatch at the seam, never in the loop.** Resolve the function pointer
  once; the inner loop has no feature branches.
- **Many small files.** One kernel domain per directory; one ISA per asm file.
- **Portability is a feature, not an afterthought.** If it doesn't run on a
  laptop with no GPU, it isn't the FFmpeg of inference.
