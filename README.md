# peregrine

**FFmpeg for AI inference.** A C core with hand-written assembly — no
intrinsics — built for extreme performance and broad hardware support, usable
as a single CLI or linked as a library.

> Status: **pre-alpha (M0).** The architecture, build, and runtime CPU-dispatch
> pipeline are in place and proven end-to-end on one kernel. The inference
> engine itself is on the [roadmap](ROADMAP.md). `peregrine` is a placeholder
> codename.

## Why

`llama.cpp` made local inference ubiquitous with compiler *intrinsics*.
`peregrine` follows FFmpeg and dav1d instead: a portable C reference for every
operation, plus **hand-written assembly per instruction set, dispatched at
runtime** — owning the registers and scheduling the compiler can't. Portable
*and* peak. See [ARCHITECTURE.md](ARCHITECTURE.md).

## Try it now (no dependencies but a C compiler)

```sh
make -f Makefile.bootstrap test
```

This builds the worked kernel (`dot_f32`: C reference + the host's hand-written
asm — NEON on ARM, AVX2 on x86) and runs the checkasm correctness + throughput
harness. Example on Apple Silicon:

```
peregrine checkasm: dot_f32
  cpu flags: neon
  n=16       ref=-0.32910      simd=-0.32910      ok
  ...
  throughput: 22.16 GFLOP/s  (0.095 ms/call, n=1048576)
CHECKASM: PASS
```

CLI:

```sh
make -f Makefile.bootstrap peregrine && ./peregrine info
```

## Build (canonical: Meson)

```sh
meson setup build && meson compile -C build && meson test -C build
./build/tests/checkasm --bench        # opt-in throughput table
```

NASM is required on x86-64; on ARM the `.S` kernels assemble via the C compiler.

> **Apple Silicon gotcha:** Meson infers the target arch from the *Python it
> runs under*. If your `python3` is an x86-64 build (common with a
> pyenv/Homebrew install under Rosetta), Meson reports `cpu family: x86_64` on
> an arm64 Mac and forces an `-arch x86_64` (Rosetta) build. Build with a
> native-arch Meson instead:
> ```sh
> /usr/bin/python3 -m pip install --user meson      # Apple's universal python3 is arm64
> /usr/bin/python3 -m mesonbuild.mesonmain setup build
> ```
> Or just use the dependency-free path: `make -f Makefile.bootstrap test`.

## Layout

```
include/peregrine/        public C ABI (pg_ prefix)
src/util/             cpu detection, aligned memory          (libavutil analog)
src/tensor/kernels/   hand-written compute kernels + dispatch (the asm lives here)
src/graph/  model/  token/  cli/    engine, loaders, tokenizer, CLI  (planned)
tests/checkasm/       asm-vs-C correctness + microbenchmarks
```

## Contributing

The kernel-adding workflow, the no-intrinsics rule, and the checkasm
requirement are in [CONTRIBUTING.md](CONTRIBUTING.md) and
[doc/writing-asm.md](doc/writing-asm.md). Start by copying
`src/tensor/kernels/dot/`.

## License

BSD-2-Clause (see [LICENSE](LICENSE)). Vendored asm macro layers retain their
upstream licenses.
