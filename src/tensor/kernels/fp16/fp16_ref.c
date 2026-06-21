/*
 * peregrine - fp16/f32 conversion portable references.
 */
#include "tensor/kernels/fp16/fp16.h"

void pg_fp16_to_f32_array_c(float *dst, const pg_fp16 *src, size_t n)
{
    for (size_t i = 0; i < n; i++)
        dst[i] = pg_fp16_to_f32(src[i]);
}

void pg_f32_to_fp16_array_c(pg_fp16 *dst, const float *src, size_t n)
{
    for (size_t i = 0; i < n; i++)
        dst[i] = pg_f32_to_fp16(src[i]);
}
