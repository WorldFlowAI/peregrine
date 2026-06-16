/*
 * peregrine - f32 RoPE (rotary position embedding).
 *
 * Deliberately split into two kernels and decoupled from KV-cache storage, so
 * it serves both vanilla decode and arbitrary-position / semantic KV reuse:
 *
 *   pg_rope_cache_f32 - build per-position cos/sin tables ONCE per sequence.
 *                       This is the only part that needs transcendentals
 *                       (a vectorised sincos lands here later).
 *   pg_rope_f32       - apply the rotation. Pure multiply/add, no
 *                       transcendentals. Out-of-place capable (out may alias
 *                       in), so a cached chunk can be roped into a scratch
 *                       buffer without mutating it.
 *
 * Because RoPE is an orthogonal rotation R_p with R_a*R_b = R_{a+b} and
 * R_p^-1 = R_{-p}, you can re-base a cached (already-roped) K block to new
 * positions p' by applying this kernel with a cos/sin cache built for the
 * deltas (p' - p): R_{p'-p} * (R_p * k) = R_{p'} * k, exactly. Un-rope is the
 * same with negated angles. Positions are an explicit per-token array, so
 * reused chunks may land at arbitrary, non-contiguous positions.
 *
 * Tensor layout: x is [n_tokens, n_heads, head_dim] contiguous, head_dim even.
 * cos/sin are [n_tokens, head_dim/2].
 *
 * C reference only for now; NEON/AVX2 (rotation) + the vectorised sincos
 * (cache) follow.
 */
#ifndef PEREGRINE_TENSOR_KERNELS_ROPE_H
#define PEREGRINE_TENSOR_KERNELS_ROPE_H

#include <stddef.h>
#include <stdint.h>
#include "util/arch.h"

typedef enum PgRopeMode {
    PG_ROPE_INTERLEAVED = 0,  /* rotate adjacent pairs (x[2i], x[2i+1])     */
    PG_ROPE_NEOX        = 1,  /* rotate split-half pairs (x[i], x[i+d/2])   */
} PgRopeMode;

typedef void (*pg_rope_cache_f32_fn)(float *cos, float *sin,
                                     const int32_t *positions,
                                     size_t n_tokens, size_t head_dim,
                                     float theta_base);

typedef void (*pg_rope_f32_fn)(float *out, const float *in,
                               const float *cos, const float *sin,
                               size_t n_tokens, size_t n_heads, size_t head_dim,
                               PgRopeMode mode);

typedef struct PgRopeDSP {
    pg_rope_cache_f32_fn cache;
    pg_rope_f32_fn       apply;
} PgRopeDSP;

typedef struct PgRopeVariant {
    const char     *name;
    pg_rope_f32_fn  fn;       /* the rotation (the hot, per-layer part) */
    unsigned        req_flags;
} PgRopeVariant;

void pg_rope_dsp_init(PgRopeDSP *dsp, unsigned cpu_flags);
const PgRopeVariant *pg_rope_variants(size_t *count);

void pg_rope_cache_f32_c(float *cos, float *sin, const int32_t *positions,
                         size_t n_tokens, size_t head_dim, float theta_base);
void pg_rope_f32_c(float *out, const float *in, const float *cos, const float *sin,
                   size_t n_tokens, size_t n_heads, size_t head_dim, PgRopeMode mode);
#if PG_ARCH_X86_64
void pg_rope_f32_avx2(float *out, const float *in, const float *cos, const float *sin,
                      size_t n_tokens, size_t n_heads, size_t head_dim, PgRopeMode mode);
#endif
#if PG_ARCH_AARCH64
void pg_rope_f32_neon(float *out, const float *in, const float *cos, const float *sin,
                      size_t n_tokens, size_t n_heads, size_t head_dim, PgRopeMode mode);
#endif

#endif /* PEREGRINE_TENSOR_KERNELS_ROPE_H */
