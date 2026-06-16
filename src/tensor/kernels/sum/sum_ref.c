/*
 * peregrine - f32 sum: portable C reference (the correctness oracle).
 */
#include "tensor/kernels/sum/sum.h"

float pg_sum_f32_c(const float *x, size_t n)
{
    float s = 0.0f;
    for (size_t i = 0; i < n; i++)
        s += x[i];
    return s;
}
