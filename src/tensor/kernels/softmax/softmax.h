/*
 * peregrine - f32 softmax over n elements (numerically stable):
 *   out[i] = e^(x[i] - max) / sum_j e^(x[j] - max)
 *
 * Operates on a single contiguous vector (one softmax row). Higher-level code
 * loops this over rows / attention heads.
 *
 * C reference only for now; a fused NEON/AVX2 version (max + exp + sum + scale,
 * built on the vectorised exp) follows.
 */
#ifndef PEREGRINE_TENSOR_KERNELS_SOFTMAX_H
#define PEREGRINE_TENSOR_KERNELS_SOFTMAX_H

#include <stddef.h>
#include "util/arch.h"

typedef void (*pg_softmax_f32_fn)(const float *in, float *out, size_t n);

typedef struct PgSoftmaxDSP { pg_softmax_f32_fn softmax_f32; } PgSoftmaxDSP;

typedef struct PgSoftmaxVariant {
    const char        *name;
    pg_softmax_f32_fn  fn;
    unsigned           req_flags;
} PgSoftmaxVariant;

void pg_softmax_dsp_init(PgSoftmaxDSP *dsp, unsigned cpu_flags);
const PgSoftmaxVariant *pg_softmax_variants(size_t *count);

void pg_softmax_f32_c(const float *in, float *out, size_t n);
#if PG_ARCH_X86_64
void pg_softmax_f32_avx2(const float *in, float *out, size_t n);
#endif
#if PG_ARCH_AARCH64
void pg_softmax_f32_neon(const float *in, float *out, size_t n);
#endif

#endif /* PEREGRINE_TENSOR_KERNELS_SOFTMAX_H */
