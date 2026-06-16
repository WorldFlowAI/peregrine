/*
 * config.h - Peregrine build-config shim for ext/arm/asm.S
 *
 * dav1d's asm.S (#include "config.h") expects a few build-time symbols that
 * dav1d generates from meson. Peregrine authors them here (config values are
 * not part of asm.S's BSD license). This file is found via -Isrc when a NEON
 * kernel includes "ext/arm/asm.S".
 *
 * NOTE: this is the ASSEMBLY build config, deliberately minimal. It is not the
 * place for C feature defines.
 */
#ifndef PEREGRINE_CONFIG_H
#define PEREGRINE_CONFIG_H

/* Target arch: AArch64 only for now (32-bit ARM is future work). */
#define ARCH_AARCH64 1
#define ARCH_ARM     0

/* Symbol mangling: function foo, export=1 -> <PRIVATE_PREFIX>foo, i.e. pg_foo.
 * PRIVATE_PREFIX carries its own trailing underscore (dav1d convention). */
#define PRIVATE_PREFIX pg_

/* Mach-O prepends '_' to C symbols; PREFIX makes EXTERN add it. */
#if defined(__APPLE__) || defined(__MACH__)
#define PREFIX 1
#endif

/* LLVM's integrated assembler (and Apple as) lack the GNU `.func`/`.endfunc`
 * directives, so do not emit them. */
#define HAVE_AS_FUNC 0

#endif /* PEREGRINE_CONFIG_H */
