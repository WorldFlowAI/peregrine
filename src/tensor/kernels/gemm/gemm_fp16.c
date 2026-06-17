/*
 * peregrine - fp16-input GEMM: fp16->f32 callbacks + scalar entry.
 */
#include "tensor/kernels/gemm/gemm_fp16.h"

#include <string.h>

float pg_fp16_to_f32(pg_fp16 h)
{
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (uint32_t)(h >> 10) & 0x1fu;
    uint32_t mant = (uint32_t)h & 0x03ffu;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int e = -14;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                e--;
            }
            mant &= 0x03ffu;
            bits = sign | (uint32_t)(e + 127) << 23 | mant << 13;
        }
    } else if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | mant << 13;
    } else {
        bits = sign | (exp + 112u) << 23 | mant << 13;
    }

    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

void pg_fp16_pack_a(float *restrict dst, const void *srcv, size_t lda,
                    size_t ib, size_t K, size_t mr)
{
    const pg_fp16 *A = srcv;
    for (size_t k = 0; k < K; k++)
        for (size_t r = 0; r < mr; r++)
            dst[k * mr + r] = pg_fp16_to_f32(A[(ib + r) * lda + k]);
}

void pg_fp16_pack_b(float *restrict dst, const void *srcv, size_t ldb,
                    size_t jb, size_t K, size_t nr)
{
    const pg_fp16 *B = srcv;
    for (size_t k = 0; k < K; k++)
        for (size_t c = 0; c < nr; c++)
            dst[k * nr + c] = pg_fp16_to_f32(B[k * ldb + (jb + c)]);
}

void pg_fp16_scalar(size_t i, size_t j, size_t K,
                    const void *Av, size_t lda, const void *Bv, size_t ldb,
                    float *C, size_t ldc)
{
    const pg_fp16 *A = Av, *B = Bv;
    float s = 0.0f;
    for (size_t k = 0; k < K; k++)
        s += pg_fp16_to_f32(A[i * lda + k]) * pg_fp16_to_f32(B[k * ldb + j]);
    C[i * ldc + j] = s;
}

void pg_fp16gemm_c(size_t M, size_t N, size_t K,
                   const pg_fp16 *A, size_t lda,
                   const pg_fp16 *B, size_t ldb,
                   float *C, size_t ldc)
{
    for (size_t i = 0; i < M; i++)
        for (size_t j = 0; j < N; j++)
            pg_fp16_scalar(i, j, K, A, lda, B, ldb, C, ldc);
}
