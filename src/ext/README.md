# Vendored assembly macro layers

Third-party files, kept **verbatim** with their upstream license headers. Do
not edit them; they are updated by re-vendoring from upstream. The build-time
symbols they expect are supplied by Peregrine's own shims (`src/config.asm`,
`src/config.h`).

| File              | Upstream                              | License                     |
|-------------------|---------------------------------------|-----------------------------|
| `x86/x86inc.asm`  | x264 project (also used by FFmpeg, dav1d) | ISC (see file header)   |
| `arm/asm.S`       | VideoLAN / dav1d authors              | BSD-2-Clause (see file header) |

## What they give us

- **`x86inc.asm`** — `cglobal`/`cextern` (ABI-correct symbol decls + automatic
  Win64/SysV/x86-32 calling-convention handling), abstract registers
  (`r0..rN`, `m0..mN`), `INIT_XMM/YMM/ZMM`, and `RET` with **automatic
  `vzeroupper`** for AVX functions.
- **`arm/asm.S`** — `function`/`endfunc`/`const` macros handling the Mach-O vs
  ELF symbol underscore, alignment, `.type`/`.size`/`.hidden`, and AArch64
  BTI/PAC branch-protection markers.

## Updating

```sh
curl -fsSL -o src/ext/x86/x86inc.asm \
  https://code.videolan.org/videolan/dav1d/-/raw/master/src/ext/x86/x86inc.asm
curl -fsSL -o src/ext/arm/asm.S \
  https://code.videolan.org/videolan/dav1d/-/raw/master/src/arm/asm.S
```

Then re-run the checkasm suite. If upstream adds a new required build symbol,
add it to `src/config.asm` / `src/config.h`, never to the vendored file.
