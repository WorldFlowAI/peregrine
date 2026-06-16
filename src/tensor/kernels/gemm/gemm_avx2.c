/*
 * peregrine - f32 GEMM, x86-64 AVX2 entry point.
 *
 * Wires the 6x16 AVX2 microkernel into the shared packed + threaded driver
 * (gemm_blocked.c). The microkernel is the only ISA-specific part.
 */
#include "tensor/kernels/gemm/gemm.h"

extern void pg_sgemm_ukernel_6x16_avx2(size_t kc, const float *a, const float *b,
                                        float *c, size_t ldc);

void pg_sgemm_avx2(size_t M, size_t N, size_t K,
                   const float *A, size_t lda,
                   const float *B, size_t ldb,
                   float *C, size_t ldc)
{
    pg_sgemm_blocked(M, N, K, A, lda, B, ldb, C, ldc,
                     6, 16, pg_sgemm_ukernel_6x16_avx2);
}
