/*
 * peregrine - f32 GEMM, AArch64 NEON entry point.
 *
 * Wires the 8x8 NEON microkernel into the shared packed + threaded driver
 * (gemm_blocked.c). The microkernel is the only ISA-specific part.
 */
#include "tensor/kernels/gemm/gemm.h"

extern void pg_sgemm_ukernel_8x8_neon(size_t kc, const float *a, const float *b,
                                       float *c, size_t ldc);

void pg_sgemm_neon(size_t M, size_t N, size_t K,
                   const float *A, size_t lda,
                   const float *B, size_t ldb,
                   float *C, size_t ldc)
{
    pg_sgemm_blocked(M, N, K, A, lda, B, ldb, C, ldc,
                     8, 8, pg_sgemm_ukernel_8x8_neon);
}
