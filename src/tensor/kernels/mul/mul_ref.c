/*
 * peregrine - f32 elementwise multiply: portable C reference.
 */
#include "tensor/kernels/mul/mul.h"

void pg_mul_f32_c(const float *a, const float *b, float *out, size_t n)
{
    for (size_t i = 0; i < n; i++)
        out[i] = a[i] * b[i];
}
