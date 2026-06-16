/*
 * peregrine - f32 softmax: portable C reference (numerically stable).
 */
#include "tensor/kernels/softmax/softmax.h"

#include <math.h>

void pg_softmax_f32_c(const float *in, float *out, size_t n)
{
    if (n == 0)
        return;

    float m = in[0];
    for (size_t i = 1; i < n; i++)
        if (in[i] > m)
            m = in[i];

    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float e = expf(in[i] - m);
        out[i] = e;
        sum += e;
    }

    float inv = 1.0f / sum;
    for (size_t i = 0; i < n; i++)
        out[i] *= inv;
}
