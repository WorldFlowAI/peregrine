/*
 * peregrine - f32 dot product: portable C reference.
 *
 * Deliberately simple and unvectorised. This is the oracle checkasm compares
 * every hand-written variant against; keep it obviously correct, never
 * "optimise" it (that is what the asm is for).
 */
#include "tensor/kernels/dot/dot.h"

float pg_dot_f32_c(const float *a, const float *b, size_t n)
{
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++)
        sum += a[i] * b[i];
    return sum;
}
