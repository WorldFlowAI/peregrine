/*
 * peregrine - f32 vectorised exp:  out[i] = e^in[i]
 *
 * The shared building block for silu / gelu / softmax. The NEON/AVX2 variants
 * use the cephes range-reduction + degree-6 minimax polynomial:
 *   e^x = 2^m * e^r,  m = round(x*log2e),  r = x - m*ln2,  |r| <= ln2/2
 * reconstructing 2^m by building the float exponent field directly. Measured
 * max relative error vs libm expf is ~1e-7 (under 1 ULP) on the accurate range;
 * inputs are clamped to +/-88.376 (above/below that, expf overflows/underflows).
 *
 * The C reference is libm expf (the accuracy oracle); the asm variants are the
 * fast approximation, checked against it.
 */
#ifndef PEREGRINE_TENSOR_KERNELS_EXP_H
#define PEREGRINE_TENSOR_KERNELS_EXP_H

#include <stddef.h>
#include "util/arch.h"

typedef void (*pg_exp_f32_fn)(const float *in, float *out, size_t n);

typedef struct PgExpDSP { pg_exp_f32_fn exp_f32; } PgExpDSP;

typedef struct PgExpVariant {
    const char    *name;
    pg_exp_f32_fn  fn;
    unsigned       req_flags;
} PgExpVariant;

void pg_exp_dsp_init(PgExpDSP *dsp, unsigned cpu_flags);
const PgExpVariant *pg_exp_variants(size_t *count);

void pg_exp_f32_c(const float *in, float *out, size_t n);
#if PG_ARCH_X86_64
void pg_exp_f32_avx2(const float *in, float *out, size_t n);
#endif
#if PG_ARCH_AARCH64
void pg_exp_f32_neon(const float *in, float *out, size_t n);
#endif

#endif /* PEREGRINE_TENSOR_KERNELS_EXP_H */
