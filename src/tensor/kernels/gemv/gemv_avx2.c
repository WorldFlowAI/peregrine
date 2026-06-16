/*
 * peregrine - f32 GEMV, x86-64 AVX2 entry point.
 * Threaded dot-per-row over the AVX2 dot kernel.
 */
#include "tensor/kernels/gemv/gemv.h"

void pg_sgemv_avx2(size_t M, size_t K, const float *A, size_t lda,
                   const float *x, float *y)
{
    pg_sgemv_driver(M, K, A, lda, x, y, pg_dot_f32_avx2);
}
