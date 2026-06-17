/*
 * peregrine - fp16-input GEMM.
 *
 *   C[M,N] (f32) = A[M,K] (fp16) @ B[K,N] (fp16)
 *
 * The portable/storage paths convert fp16 inputs to f32 while packing and reuse
 * the f32 microkernel. Native FP16 variants may use fp16 arithmetic internally
 * and convert the final tile to f32 on store.
 */
#ifndef PEREGRINE_TENSOR_KERNELS_GEMM_FP16_H
#define PEREGRINE_TENSOR_KERNELS_GEMM_FP16_H

#include <stddef.h>
#include <stdint.h>
#include "util/arch.h"
#include "tensor/kernels/gemm/gemm.h"

typedef uint16_t pg_fp16;

typedef void (*pg_fp16gemm_fn)(size_t M, size_t N, size_t K,
                               const pg_fp16 *A, size_t lda,
                               const pg_fp16 *B, size_t ldb,
                               float *C, size_t ldc);

typedef struct PgFp16GemmDSP {
    pg_fp16gemm_fn gemm;
} PgFp16GemmDSP;

typedef struct PgFp16GemmVariant {
    const char    *name;
    pg_fp16gemm_fn fn;
    unsigned       req_flags;
} PgFp16GemmVariant;

void pg_fp16gemm_dsp_init(PgFp16GemmDSP *dsp, unsigned cpu_flags);
const PgFp16GemmVariant *pg_fp16gemm_variants(size_t *count);

float pg_fp16_to_f32(pg_fp16 h);

void pg_fp16gemm_c(size_t M, size_t N, size_t K,
                   const pg_fp16 *A, size_t lda,
                   const pg_fp16 *B, size_t ldb,
                   float *C, size_t ldc);
#if PG_ARCH_AARCH64
void pg_fp16gemm_neon(size_t M, size_t N, size_t K,
                      const pg_fp16 *A, size_t lda,
                      const pg_fp16 *B, size_t ldb,
                      float *C, size_t ldc);
void pg_fp16gemm_f16(size_t M, size_t N, size_t K,
                     const pg_fp16 *A, size_t lda,
                     const pg_fp16 *B, size_t ldb,
                     float *C, size_t ldc);
#endif
#if PG_ARCH_X86_64
void pg_fp16gemm_avx2(size_t M, size_t N, size_t K,
                      const pg_fp16 *A, size_t lda,
                      const pg_fp16 *B, size_t ldb,
                      float *C, size_t ldc);
#endif

void pg_fp16_pack_a(float *dst, const void *A, size_t lda,
                    size_t ib, size_t K, size_t mr);
void pg_fp16_pack_b(float *dst, const void *B, size_t ldb,
                    size_t jb, size_t K, size_t nr);
void pg_fp16_scalar(size_t i, size_t j, size_t K,
                    const void *A, size_t lda, const void *B, size_t ldb,
                    float *C, size_t ldc);

#endif /* PEREGRINE_TENSOR_KERNELS_GEMM_FP16_H */
