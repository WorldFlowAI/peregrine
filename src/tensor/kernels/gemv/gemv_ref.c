/*
 * peregrine - f32 GEMV portable reference (correctness oracle's variant).
 * Threaded dot-per-row over the scalar dot kernel.
 */
#include "tensor/kernels/gemv/gemv.h"

void pg_sgemv_c(size_t M, size_t K, const float *A, size_t lda,
                const float *x, float *y)
{
    pg_sgemv_driver(M, K, A, lda, x, y, pg_dot_f32_c);
}
