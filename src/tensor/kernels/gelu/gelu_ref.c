/*
 * peregrine - f32 GELU (tanh approximation): portable C reference.
 */
#include "tensor/kernels/gelu/gelu.h"

#include <math.h>

/* sqrt(2/pi) */
#define PG_GELU_K 0.7978845608028654f

void pg_gelu_f32_c(const float *in, float *out, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        float x = in[i];
        float inner = PG_GELU_K * (x + 0.044715f * x * x * x);
        out[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
}
