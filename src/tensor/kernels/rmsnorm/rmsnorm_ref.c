/*
 * peregrine - f32 RMSNorm: portable C reference (the correctness oracle).
 */
#include "tensor/kernels/rmsnorm/rmsnorm.h"

#include <math.h>

void pg_rmsnorm_f32_c(float *out, const float *x, const float *weight, size_t n, float eps)
{
    if (n == 0)
        return;

    float ss = 0.0f;
    for (size_t i = 0; i < n; i++)
        ss += x[i] * x[i];

    float mean  = ss / (float)n;
    float scale = 1.0f / sqrtf(mean + eps);

    for (size_t i = 0; i < n; i++)
        out[i] = x[i] * scale * weight[i];
}
