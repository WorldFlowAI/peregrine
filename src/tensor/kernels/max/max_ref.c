/*
 * peregrine - f32 max: portable C reference (the correctness oracle).
 */
#include "tensor/kernels/max/max.h"

#include <math.h>

float pg_max_f32_c(const float *x, size_t n)
{
    if (n == 0)
        return -INFINITY;
    float m = x[0];
    for (size_t i = 1; i < n; i++)
        if (x[i] > m)
            m = x[i];
    return m;
}
