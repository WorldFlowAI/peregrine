/*
 * peregrine - f32 sum reduction:  sum_i x[i]
 */
#ifndef PEREGRINE_TENSOR_KERNELS_SUM_H
#define PEREGRINE_TENSOR_KERNELS_SUM_H

#include <stddef.h>
#include "util/arch.h"

typedef float (*pg_sum_f32_fn)(const float *x, size_t n);

typedef struct PgSumDSP { pg_sum_f32_fn sum_f32; } PgSumDSP;

typedef struct PgSumVariant {
    const char    *name;
    pg_sum_f32_fn  fn;
    unsigned       req_flags;
} PgSumVariant;

void pg_sum_dsp_init(PgSumDSP *dsp, unsigned cpu_flags);
const PgSumVariant *pg_sum_variants(size_t *count);

float pg_sum_f32_c(const float *x, size_t n);
#if PG_ARCH_X86_64
float pg_sum_f32_avx2(const float *x, size_t n);
#endif
#if PG_ARCH_AARCH64
float pg_sum_f32_neon(const float *x, size_t n);
#endif

#endif /* PEREGRINE_TENSOR_KERNELS_SUM_H */
