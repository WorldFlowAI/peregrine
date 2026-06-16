# Writing a kernel in peregrine

Every kernel domain has the same four-part shape. To add one, copy
`src/tensor/kernels/dot/` and rename. This doc is the checklist plus the
hard-won ABI gotchas.

## The four parts

```
src/tensor/kernels/<op>/
  <op>.h          # DSP struct, fn-ptr typedef, extern asm decls, C ref decl
  <op>_ref.c      # portable C reference — the correctness oracle
  <op>_init.c     # pg_<op>_dsp_init(): C ref -> best asm for cpu_flags
  x86/<op>_<isa>.asm   # NASM, one file per ISA (sse2/avx2/avx512)
  arm/<op>_<isa>.S     # GAS,  one file per ISA (neon/sve2)
tests/checkasm/checkasm_<op>.c   # correctness vs ref + microbench
```

Then register the new C/asm sources in `meson.build` (and
`Makefile.bootstrap` if you want it in the zero-dep build), and add the test in
`tests/meson.build`.

## The assembly frame: use the macro layer

Never hand-roll `.global` / symbol underscores / `vzeroupper`. Peregrine
vendors the battle-tested macro layers (`src/ext/`, see `src/ext/README.md`)
and supplies their build config (`src/config.asm`, `src/config.h`). They handle
ABI, symbol mangling (`pg_` prefix + Mach-O/ELF underscore), alignment, and
AArch64 BTI/PAC markers for you.

**x86 (NASM, `ext/x86/x86inc.asm`):**
```nasm
%include "config.asm"
%include "ext/x86/x86inc.asm"
SECTION .text
INIT_YMM avx2                    ; marks it AVX -> RET auto-emits vzeroupper
cglobal dot_f32, 3, 4, 8, a, b, n ; 3 args, 4 gpregs, 8 vregs; a/b/n = r0/r1/r2
    ...                          ; INIT_YMM appends the ISA suffix:
    RET                          ; symbol exported is pg_dot_f32_avx2
```

**ARM (GAS, `ext/arm/asm.S`):**
```asm
#include "ext/arm/asm.S"
function dot_f32_neon, export=1  ; exports pg_dot_f32_neon
    ...
    ret
endfunc
```

Both must see the include path `src/` (already wired into both builds: `-Isrc`
for the compiler, `-I src/` for nasm).

## Rules

1. **No intrinsics.** Hand asm or the C reference — nothing in between.
2. **Never optimise the C reference.** It must stay obviously correct; it is
   what checkasm trusts.
3. **Dispatch at the seam.** Resolve the function pointer once in `_init`; the
   inner loop has zero feature branches.
4. **No merge without a passing checkasm entry** covering edge sizes: `0`, one
   element, sub-vector, odd tail, and large.

## Calling-convention quick reference

With `cglobal`/`function` you reference args via the abstract names, so most of
this is handled for you — but know what's underneath.

### x86-64 (System V AMD64 — Linux/macOS)
- Native SysV: integer args `rdi, rsi, rdx, rcx, r8, r9`; float return `xmm0`.
- Under x86inc you instead use `r0..rN` / named args, and `RET` emits the
  `vzeroupper` for any `INIT_YMM`/`INIT_ZMM` function automatically.
- Windows x64 differs (`rcx,rdx,r8,r9`; `xmm6–15` callee-saved) — `cglobal`
  abstracts it; raw nasm does not. This is why we use the macro layer.

### AArch64 (AAPCS64)
- Integer args: `x0–x7`. FP/SIMD args+return: `v0–v7` (`s0`/`d0` for scalar).
- Callee-saved: `x19–x28`, and the **low 64 bits of `v8–v15`** — save them if
  you use those vector regs. `v0–v7` and `v16–v31` are caller-saved (free).
- Return with `ret` (branches `x30`).

## Gotchas we already hit

- **Mach-O vs ELF symbol names.** Apple prefixes C symbols with `_`, Linux does
  not. The macro layer handles this (`PREFIX` in `config.asm`/`config.h`) — so
  just use `cglobal`/`function` and never write a raw `.global`.
- **ISA suffix is automatic on x86.** `INIT_YMM avx2` makes `cglobal foo`
  export `pg_foo_avx2`. Name the function bare (`dot_f32`), not `dot_f32_avx2`,
  or you get a doubled suffix. (ARM's `function` does *not* auto-suffix — there
  you write the full `dot_f32_neon`.)
- **Don't fold directives onto one cpp line.** LLVM's AArch64 assembler does
  *not* treat `;` as a statement separator, so a macro expanding to
  `".global x; x:"` silently drops the symbol (you get an "undefined symbol"
  link error). Keep `.global`, the label, and each directive on their own
  physical line.
- **`.p2align N` not `.align N`.** On Mach-O `.align` is a power of two but on
  ELF it's a byte count — `.p2align` (log2) means the same thing everywhere.

## Verifying

```sh
make -f Makefile.bootstrap test      # host arch, quick
meson test -C build                  # full, wired into ninja test
```

A correct kernel matches the C reference within f32 tolerance across all sizes
*and* reports throughput so regressions are visible.
