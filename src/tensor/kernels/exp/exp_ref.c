/*
 * peregrine - f32 exp: portable C reference (libm expf, the accuracy oracle).
 */
#include "tensor/kernels/exp/exp.h"

#include <math.h>

void pg_exp_f32_c(const float *in, float *out, size_t n)
{
    for (size_t i = 0; i < n; i++)
        out[i] = expf(in[i]);
}
