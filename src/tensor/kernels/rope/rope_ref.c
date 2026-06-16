/*
 * peregrine - f32 RoPE: portable C references.
 *   cache:  cos/sin[t,i] = cos|sin( positions[t] * theta_base^(-2i/head_dim) )
 *   apply:  rotate each pair (a,b) by its angle:
 *             out0 = a*cos - b*sin
 *             out1 = a*sin + b*cos
 *           reading both lanes before writing, so out may alias in (in place).
 */
#include "tensor/kernels/rope/rope.h"

#include <math.h>

void pg_rope_cache_f32_c(float *cosb, float *sinb, const int32_t *positions,
                         size_t n_tokens, size_t head_dim, float theta_base)
{
    size_t half = head_dim / 2;
    for (size_t t = 0; t < n_tokens; t++) {
        double p = (double)positions[t];
        for (size_t i = 0; i < half; i++) {
            double freq = pow((double)theta_base, -2.0 * (double)i / (double)head_dim);
            double ang = p * freq;
            cosb[t * half + i] = (float)cos(ang);
            sinb[t * half + i] = (float)sin(ang);
        }
    }
}

void pg_rope_f32_c(float *out, const float *in, const float *cosb, const float *sinb,
                   size_t n_tokens, size_t n_heads, size_t head_dim, PgRopeMode mode)
{
    size_t half = head_dim / 2;
    for (size_t t = 0; t < n_tokens; t++) {
        const float *ct = cosb + t * half;
        const float *st = sinb + t * half;
        for (size_t h = 0; h < n_heads; h++) {
            const float *xi = in  + (t * n_heads + h) * head_dim;
            float       *xo = out + (t * n_heads + h) * head_dim;
            for (size_t i = 0; i < half; i++) {
                float c = ct[i], s = st[i];
                size_t i0, i1;
                if (mode == PG_ROPE_INTERLEAVED) { i0 = 2 * i; i1 = 2 * i + 1; }
                else                             { i0 = i;     i1 = i + half;  }
                float a = xi[i0], b = xi[i1];
                xo[i0] = a * c - b * s;
                xo[i1] = a * s + b * c;
            }
        }
    }
}
