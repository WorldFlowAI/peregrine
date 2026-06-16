/*
 * peregrine - bf16-input GEMM, AArch64 NEON entry.
 * bf16->f32 packers feeding the f32 8x8 NEON microkernel via the generic core.
 */
#include "tensor/kernels/gemm/gemm_bf16.h"

extern void pg_sgemm_ukernel_8x8_neon(size_t kc, const float *a, const float *b,
                                       float *c, size_t ldc);

void pg_bf16gemm_neon(size_t M, size_t N, size_t K,
                      const pg_bf16 *A, size_t lda,
                      const pg_bf16 *B, size_t ldb,
                      float *C, size_t ldc)
{
    pg_gemm_blocked_generic(M, N, K, A, lda, B, ldb, C, ldc,
                            8, 8, pg_sgemm_ukernel_8x8_neon,
                            pg_bf16_pack_a, pg_bf16_pack_b, pg_bf16_scalar);
}
