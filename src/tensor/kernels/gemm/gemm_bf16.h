/*
 * peregrine - bf16-input GEMM (storage path).
 *
 *   C[M,N] (f32) = A[M,K] (bf16) @ B[K,N] (bf16)
 *
 * bfloat16 is exactly the top 16 bits of an IEEE f32, so bf16 -> f32 is a free
 * bit-extension (no rounding, no ISA feature). Inputs are stored bf16 (half the
 * memory/bandwidth), converted to f32 in the pack step, and accumulated in f32
 * by the SAME register-blocked f32 microkernel via the generic driver. Compute
 * speed therefore matches SGEMM; the win is the halved input footprint and the
 * ability to consume bf16 weights directly.
 *
 * A native bf16-compute path (ARM BFDOT/BFMMLA, x86 AVX512-BF16) is a follow-up;
 * it trades exactness for speed and needs ISA features to detect/validate.
 */
#ifndef PEREGRINE_TENSOR_KERNELS_GEMM_BF16_H
#define PEREGRINE_TENSOR_KERNELS_GEMM_BF16_H

#include <stddef.h>
#include <stdint.h>
#include "util/arch.h"
#include "tensor/kernels/gemm/gemm.h"

typedef uint16_t pg_bf16;

typedef void (*pg_bf16gemm_fn)(size_t M, size_t N, size_t K,
                               const pg_bf16 *A, size_t lda,
                               const pg_bf16 *B, size_t ldb,
                               float *C, size_t ldc);

typedef struct PgBf16GemmDSP {
    pg_bf16gemm_fn gemm;
} PgBf16GemmDSP;

typedef struct PgBf16GemmVariant {
    const char    *name;
    pg_bf16gemm_fn fn;
    unsigned       req_flags;
} PgBf16GemmVariant;

void pg_bf16gemm_dsp_init(PgBf16GemmDSP *dsp, unsigned cpu_flags);
const PgBf16GemmVariant *pg_bf16gemm_variants(size_t *count);

void pg_bf16gemm_c(size_t M, size_t N, size_t K,
                   const pg_bf16 *A, size_t lda,
                   const pg_bf16 *B, size_t ldb,
                   float *C, size_t ldc);
#if PG_ARCH_AARCH64
void pg_bf16gemm_neon(size_t M, size_t N, size_t K,
                      const pg_bf16 *A, size_t lda,
                      const pg_bf16 *B, size_t ldb,
                      float *C, size_t ldc);
/* native bf16-compute path (FEAT_BF16 / BFMMLA) - the optimized tier */
void pg_bf16gemm_bfmmla(size_t M, size_t N, size_t K,
                        const pg_bf16 *A, size_t lda,
                        const pg_bf16 *B, size_t ldb,
                        float *C, size_t ldc);
#endif
#if PG_ARCH_X86_64
void pg_bf16gemm_avx2(size_t M, size_t N, size_t K,
                      const pg_bf16 *A, size_t lda,
                      const pg_bf16 *B, size_t ldb,
                      float *C, size_t ldc);
#endif

/* bf16->f32 pack/scalar callbacks for the generic GEMM core (shared by the
 * arch entries). */
void pg_bf16_pack_a(float *dst, const void *A, size_t lda,
                    size_t ib, size_t K, size_t mr);
void pg_bf16_pack_b(float *dst, const void *B, size_t ldb,
                    size_t jb, size_t K, size_t nr);
void pg_bf16_scalar(size_t i, size_t j, size_t K,
                    const void *A, size_t lda, const void *B, size_t ldb,
                    float *C, size_t ldc);

#endif /* PEREGRINE_TENSOR_KERNELS_GEMM_BF16_H */
