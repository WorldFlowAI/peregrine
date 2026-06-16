/*
 * peregrine - bf16-input GEMM: bf16->f32 conversion callbacks + scalar entry.
 */
#include "tensor/kernels/gemm/gemm_bf16.h"

#include <string.h>

/* bf16 -> f32 is exact: bf16 occupies the high 16 bits of the f32 layout. */
static inline float bf16_to_f32(pg_bf16 h)
{
    uint32_t bits = (uint32_t)h << 16;
    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

void pg_bf16_pack_a(float *dst, const void *srcv, size_t lda,
                    size_t ib, size_t K, size_t mr)
{
    const pg_bf16 *A = srcv;
    for (size_t k = 0; k < K; k++)
        for (size_t r = 0; r < mr; r++)
            dst[k * mr + r] = bf16_to_f32(A[(ib + r) * lda + k]);
}

void pg_bf16_pack_b(float *dst, const void *srcv, size_t ldb,
                    size_t jb, size_t K, size_t nr)
{
    const pg_bf16 *B = srcv;
    for (size_t k = 0; k < K; k++)
        for (size_t c = 0; c < nr; c++)
            dst[k * nr + c] = bf16_to_f32(B[k * ldb + (jb + c)]);
}

void pg_bf16_scalar(size_t i, size_t j, size_t K,
                    const void *Av, size_t lda, const void *Bv, size_t ldb,
                    float *C, size_t ldc)
{
    const pg_bf16 *A = Av, *B = Bv;
    float s = 0.0f;
    for (size_t k = 0; k < K; k++)
        s += bf16_to_f32(A[i * lda + k]) * bf16_to_f32(B[k * ldb + j]);
    C[i * ldc + j] = s;
}

/* Portable reference: scalar bf16->f32 dot per cell (the correctness oracle's
 * variant). */
void pg_bf16gemm_c(size_t M, size_t N, size_t K,
                   const pg_bf16 *A, size_t lda,
                   const pg_bf16 *B, size_t ldb,
                   float *C, size_t ldc)
{
    for (size_t i = 0; i < M; i++)
        for (size_t j = 0; j < N; j++)
            pg_bf16_scalar(i, j, K, A, lda, B, ldb, C, ldc);
}
