/*
 * peregrine - f32 GELU (tanh approximation):
 *   gelu(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
 *
 * This is the tanh approximation used by GPT-2/most fast kernels (it reduces to
 * a tanh, hence a vectorised exp later). The exact erf form can be added as a
 * separate variant if a model needs it.
 *
 * C reference only for now; NEON/AVX2 follow on top of the vectorised exp.
 */
#ifndef PEREGRINE_TENSOR_KERNELS_GELU_H
#define PEREGRINE_TENSOR_KERNELS_GELU_H

#include <stddef.h>
#include "util/arch.h"

typedef void (*pg_gelu_f32_fn)(const float *in, float *out, size_t n);

typedef struct PgGeluDSP { pg_gelu_f32_fn gelu_f32; } PgGeluDSP;

typedef struct PgGeluVariant {
    const char     *name;
    pg_gelu_f32_fn  fn;
    unsigned        req_flags;
} PgGeluVariant;

void pg_gelu_dsp_init(PgGeluDSP *dsp, unsigned cpu_flags);
const PgGeluVariant *pg_gelu_variants(size_t *count);

void pg_gelu_f32_c(const float *in, float *out, size_t n);

#endif /* PEREGRINE_TENSOR_KERNELS_GELU_H */
