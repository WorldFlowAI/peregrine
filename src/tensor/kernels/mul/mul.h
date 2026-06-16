/*
 * peregrine - f32 elementwise multiply:  out[i] = a[i] * b[i]
 */
#ifndef PEREGRINE_TENSOR_KERNELS_MUL_H
#define PEREGRINE_TENSOR_KERNELS_MUL_H

#include <stddef.h>
#include "util/arch.h"

typedef void (*pg_mul_f32_fn)(const float *a, const float *b, float *out, size_t n);

typedef struct PgMulDSP { pg_mul_f32_fn mul_f32; } PgMulDSP;

typedef struct PgMulVariant {
    const char    *name;
    pg_mul_f32_fn  fn;
    unsigned       req_flags;
} PgMulVariant;

void pg_mul_dsp_init(PgMulDSP *dsp, unsigned cpu_flags);
const PgMulVariant *pg_mul_variants(size_t *count);

void pg_mul_f32_c(const float *a, const float *b, float *out, size_t n);
#if PG_ARCH_X86_64
void pg_mul_f32_avx2(const float *a, const float *b, float *out, size_t n);
#endif
#if PG_ARCH_AARCH64
void pg_mul_f32_neon(const float *a, const float *b, float *out, size_t n);
#endif

#endif /* PEREGRINE_TENSOR_KERNELS_MUL_H */
