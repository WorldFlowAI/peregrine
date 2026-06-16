/*
 * peregrine - bf16-input GEMM, x86-64 AVX2 entry.
 * bf16->f32 packers feeding the f32 6x16 AVX2 microkernel via the generic core.
 */
#include "tensor/kernels/gemm/gemm_bf16.h"

extern void pg_sgemm_ukernel_6x16_avx2(size_t kc, const float *a, const float *b,
                                        float *c, size_t ldc);

void pg_bf16gemm_avx2(size_t M, size_t N, size_t K,
                      const pg_bf16 *A, size_t lda,
                      const pg_bf16 *B, size_t ldb,
                      float *C, size_t ldc)
{
    pg_gemm_blocked_generic(M, N, K, A, lda, B, ldb, C, ldc,
                            6, 16, pg_sgemm_ukernel_6x16_avx2,
                            pg_bf16_pack_a, pg_bf16_pack_b, pg_bf16_scalar);
}
