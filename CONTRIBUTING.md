# Contributing to peregrine

peregrine is a hand-assembly project. The bar is correctness-proven speed, and the
review process is built around that.

## The golden rules

1. **No intrinsics in `src/tensor/kernels/`.** Hand-written asm or the C
   reference. (Prototyping with intrinsics to find an approach is fine — the
   merged kernel is asm.)
2. **Every asm routine ships with a passing checkasm entry** proving it equals
   the C reference across edge sizes. No checkasm, no merge.
3. **The C reference is the source of truth** — always present, always correct,
   never "optimised."
4. **Small files, one ISA per asm file, one kernel domain per directory.**

## Adding a kernel

Copy `src/tensor/kernels/dot/` and follow
[doc/writing-asm.md](doc/writing-asm.md). A complete PR touches:

- `<op>.h`, `<op>_ref.c`, `<op>_init.c`
- at least one `x86/*.asm` or `arm/*.S` (more ISAs welcome)
- `tests/checkasm/checkasm_<op>.c`
- `meson.build` + `tests/meson.build` registration

## Before you open a PR

```sh
meson test -C build        # correctness + bench, all built ISAs
make -f Makefile.bootstrap test   # quick host-arch sanity
```

Include benchmark numbers in the PR description (machine + before/after).
Performance claims must be reproducible.

## Style

C11, 4-space indent, `pg_` prefix on public symbols, no global mutable state in
kernels. Match the surrounding code. See the repo rules under
`coding-style.md` if present.

## Commits

Conventional commits (`feat:`, `fix:`, `perf:`, …). Sign off your commits.
