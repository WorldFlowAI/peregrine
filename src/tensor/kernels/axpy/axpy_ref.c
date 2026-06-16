/*
 * peregrine - f32 axpy: portable C reference (the correctness oracle).
 */
#include "tensor/kernels/axpy/axpy.h"

void pg_axpy_f32_c(float alpha, const float *x, float *y, size_t n)
{
    for (size_t i = 0; i < n; i++)
        y[i] += alpha * x[i];
}
