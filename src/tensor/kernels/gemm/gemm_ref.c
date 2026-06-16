/*
 * peregrine - f32 GEMM scalar reference (correctness oracle).
 *
 * ikj loop order so B and C are walked along rows (cache-friendly), but this is
 * still the naive O(MNK) baseline the SIMD variants are measured against.
 */
#include "tensor/kernels/gemm/gemm.h"

void pg_sgemm_c(size_t M, size_t N, size_t K,
                const float *A, size_t lda,
                const float *B, size_t ldb,
                float *C, size_t ldc)
{
    for (size_t i = 0; i < M; i++)
        for (size_t j = 0; j < N; j++)
            C[i * ldc + j] = 0.0f;

    for (size_t i = 0; i < M; i++)
        for (size_t k = 0; k < K; k++) {
            const float a = A[i * lda + k];
            const float *brow = B + k * ldb;
            float *crow = C + i * ldc;
            for (size_t j = 0; j < N; j++)
                crow[j] += a * brow[j];
        }
}
