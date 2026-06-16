/*
 * peregrine - f32 RMSNorm:  out[i] = x[i] * rsqrt(mean(x^2) + eps) * weight[i]
 *
 *   where mean(x^2) = (1/n) * sum_i x[i]^2
 *
 * The canonical transformer normalize. Two passes over x: a sum-of-squares
 * reduction, then a per-element scale-and-weight. Exercises reduction +
 * scalar rsqrt + broadcast + a second input array, all in one kernel.
 */
#ifndef PEREGRINE_TENSOR_KERNELS_RMSNORM_H
#define PEREGRINE_TENSOR_KERNELS_RMSNORM_H

#include <stddef.h>
#include "util/arch.h"

typedef void (*pg_rmsnorm_f32_fn)(float *out, const float *x,
                                  const float *weight, size_t n, float eps);

typedef struct PgRmsnormDSP {
    pg_rmsnorm_f32_fn rmsnorm_f32;
} PgRmsnormDSP;

typedef struct PgRmsnormVariant {
    const char        *name;
    pg_rmsnorm_f32_fn  fn;
    unsigned           req_flags;
} PgRmsnormVariant;

void pg_rmsnorm_dsp_init(PgRmsnormDSP *dsp, unsigned cpu_flags);
const PgRmsnormVariant *pg_rmsnorm_variants(size_t *count);

void pg_rmsnorm_f32_c(float *out, const float *x, const float *weight, size_t n, float eps);
#if PG_ARCH_X86_64
void pg_rmsnorm_f32_avx2(float *out, const float *x, const float *weight, size_t n, float eps);
#endif
#if PG_ARCH_AARCH64
void pg_rmsnorm_f32_neon(float *out, const float *x, const float *weight, size_t n, float eps);
#endif

#endif /* PEREGRINE_TENSOR_KERNELS_RMSNORM_H */
