/*
 * peregrine - f32 GEMV, AArch64 NEON entry point.
 * Threaded dot-per-row over the NEON dot kernel.
 */
#include "tensor/kernels/gemv/gemv.h"

void pg_sgemv_neon(size_t M, size_t K, const float *A, size_t lda,
                   const float *x, float *y)
{
    pg_sgemv_driver(M, K, A, lda, x, y, pg_dot_f32_neon);
}
