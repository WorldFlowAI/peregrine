/*
 * peregrine - fp16-input GEMM, x86-64 AVX2 storage path.
 * fp16->f32 packers feeding the f32 6x16 AVX2 microkernel via the generic core.
 */
#include "tensor/kernels/gemm/gemm_fp16.h"

extern void pg_sgemm_ukernel_6x16_avx2(size_t kc, const float *a, const float *b,
                                        float *c, size_t ldc);

void pg_fp16gemm_avx2(size_t M, size_t N, size_t K,
                      const pg_fp16 *A, size_t lda,
                      const pg_fp16 *B, size_t ldb,
                      float *C, size_t ldc)
{
    pg_gemm_blocked_generic(M, N, K, A, lda, B, ldb, C, ldc,
                            6, 16, pg_sgemm_ukernel_6x16_avx2,
                            pg_fp16_pack_a, pg_fp16_pack_b, pg_fp16_scalar);
}
