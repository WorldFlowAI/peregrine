/*
 * peregrine - f32 SiLU / swish:  out[i] = x[i] * sigmoid(x[i]) = x[i] / (1 + e^-x[i])
 *
 * C reference only for now. Hand-written NEON/AVX2 (built on a vectorised exp)
 * will be added as variants in a follow-up; the dispatch table and checkasm are
 * already in place so it works and is tested today.
 */
#ifndef PEREGRINE_TENSOR_KERNELS_SILU_H
#define PEREGRINE_TENSOR_KERNELS_SILU_H

#include <stddef.h>
#include "util/arch.h"

typedef void (*pg_silu_f32_fn)(const float *in, float *out, size_t n);

typedef struct PgSiluDSP { pg_silu_f32_fn silu_f32; } PgSiluDSP;

typedef struct PgSiluVariant {
    const char     *name;
    pg_silu_f32_fn  fn;
    unsigned        req_flags;
} PgSiluVariant;

void pg_silu_dsp_init(PgSiluDSP *dsp, unsigned cpu_flags);
const PgSiluVariant *pg_silu_variants(size_t *count);

void pg_silu_f32_c(const float *in, float *out, size_t n);

#endif /* PEREGRINE_TENSOR_KERNELS_SILU_H */
