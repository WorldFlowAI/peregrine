/*
 * peregrine - f32 dot product kernel (dispatch interface)
 *
 * This is the canonical example of peregrine's kernel pattern. Every kernel
 * domain follows the same three-part shape:
 *
 *   1. A portable C reference (`*_c`) that is ALWAYS correct. It is the
 *      oracle that checkasm tests every asm variant against.
 *   2. One hand-written assembly routine per ISA extension, declared here
 *      and implemented under x86/ and arm/.
 *   3. A DSP struct of function pointers plus an `_init` that fills it with
 *      the fastest routine the host CPU supports.
 *
 * Callers hold an PgDotDSP (initialised once) and call through the pointer;
 * they never branch on CPU features themselves.
 */
#ifndef PEREGRINE_TENSOR_KERNELS_DOT_H
#define PEREGRINE_TENSOR_KERNELS_DOT_H

#include <stddef.h>
#include "util/arch.h"

typedef float (*pg_dot_f32_fn)(const float *a, const float *b, size_t n);

typedef struct PgDotDSP {
    pg_dot_f32_fn dot_f32;
} PgDotDSP;

/* Fill `dsp` with the best routines available for `cpu_flags`. */
void pg_dot_dsp_init(PgDotDSP *dsp, unsigned cpu_flags);

/*
 * One entry per implementation, ordered fastest-first with the C reference
 * last. This is the single source of truth: pg_dot_dsp_init() picks the first
 * entry whose req_flags are all satisfied, and checkasm walks the whole table
 * to test/bench every variant the host can run.
 */
typedef struct PgDotVariant {
    const char    *name;       /* "avx2", "neon", "c"            */
    pg_dot_f32_fn  fn;
    unsigned       req_flags;  /* CPU flags required (0 = always) */
} PgDotVariant;

const PgDotVariant *pg_dot_variants(size_t *count);

/* Portable reference (the correctness oracle). */
float pg_dot_f32_c(const float *a, const float *b, size_t n);

/* Hand-written assembly entry points. */
#if PG_ARCH_X86_64
float pg_dot_f32_avx2(const float *a, const float *b, size_t n);
#endif
#if PG_ARCH_AARCH64
float pg_dot_f32_neon(const float *a, const float *b, size_t n);
#endif

#endif /* PEREGRINE_TENSOR_KERNELS_DOT_H */
