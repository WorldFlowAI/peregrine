/*
 * peregrine - f32 axpy kernel:  y[i] += alpha * x[i]   (in place)
 *
 * Same three-part shape as dot (C reference + per-ISA asm + variant table),
 * but with a scalar arg and an in-place output - so it exercises FP-arg
 * passing (alpha rides in v0/xmm0) and load-modify-store, not just reduction.
 */
#ifndef PEREGRINE_TENSOR_KERNELS_AXPY_H
#define PEREGRINE_TENSOR_KERNELS_AXPY_H

#include <stddef.h>
#include "util/arch.h"

typedef void (*pg_axpy_f32_fn)(float alpha, const float *x, float *y, size_t n);

typedef struct PgAxpyDSP {
    pg_axpy_f32_fn axpy_f32;
} PgAxpyDSP;

typedef struct PgAxpyVariant {
    const char     *name;
    pg_axpy_f32_fn  fn;
    unsigned        req_flags;
} PgAxpyVariant;

void pg_axpy_dsp_init(PgAxpyDSP *dsp, unsigned cpu_flags);
const PgAxpyVariant *pg_axpy_variants(size_t *count);

void pg_axpy_f32_c(float alpha, const float *x, float *y, size_t n);
#if PG_ARCH_X86_64
void pg_axpy_f32_avx2(float alpha, const float *x, float *y, size_t n);
#endif
#if PG_ARCH_AARCH64
void pg_axpy_f32_neon(float alpha, const float *x, float *y, size_t n);
#endif

#endif /* PEREGRINE_TENSOR_KERNELS_AXPY_H */
