/*
 * peregrine - f32 max reduction:  max_i x[i]   (requires n >= 1)
 *
 * NaN handling is unspecified (inputs are assumed finite); this matches the
 * `a > m ? a : m` reference and the hardware max instructions on both arches.
 */
#ifndef PEREGRINE_TENSOR_KERNELS_MAX_H
#define PEREGRINE_TENSOR_KERNELS_MAX_H

#include <stddef.h>
#include "util/arch.h"

typedef float (*pg_max_f32_fn)(const float *x, size_t n);

typedef struct PgMaxDSP { pg_max_f32_fn max_f32; } PgMaxDSP;

typedef struct PgMaxVariant {
    const char    *name;
    pg_max_f32_fn  fn;
    unsigned       req_flags;
} PgMaxVariant;

void pg_max_dsp_init(PgMaxDSP *dsp, unsigned cpu_flags);
const PgMaxVariant *pg_max_variants(size_t *count);

float pg_max_f32_c(const float *x, size_t n);
#if PG_ARCH_X86_64
float pg_max_f32_avx2(const float *x, size_t n);
#endif
#if PG_ARCH_AARCH64
float pg_max_f32_neon(const float *x, size_t n);
#endif

#endif /* PEREGRINE_TENSOR_KERNELS_MAX_H */
