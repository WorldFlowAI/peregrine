/*
 * peregrine - f32 elementwise add: portable C reference.
 */
#include "tensor/kernels/add/add.h"

void pg_add_f32_c(const float *a, const float *b, float *out, size_t n)
{
    for (size_t i = 0; i < n; i++)
        out[i] = a[i] + b[i];
}
