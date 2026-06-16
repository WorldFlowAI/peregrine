/*
 * peregrine - f32 SiLU: portable C reference.
 *   silu(x) = x / (1 + e^-x)
 */
#include "tensor/kernels/silu/silu.h"

#include <math.h>

void pg_silu_f32_c(const float *in, float *out, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        float x = in[i];
        out[i] = x / (1.0f + expf(-x));
    }
}
