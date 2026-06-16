/*
 * peregrine - single-precision GEMM (matrix multiply).
 *
 *   C[M,N] = A[M,K] @ B[K,N]      (row-major, C is overwritten)
 *
 * Leading dimensions (row strides, in elements) are explicit: lda >= K,
 * ldb >= N, ldc >= N. This lets callers operate on sub-tiles of a larger
 * matrix without copying. v1 is fixed at this shape; transpose flags and
 * alpha/beta scaling are mechanical follow-ups layered on the same kernels.
 *
 * The optimised variants follow Goto's structure: pack panels of A and B into
 * contiguous, microkernel-friendly buffers, then drive a register-blocked
 * microkernel (the hot loop, hand-written per ISA) over the packed panels.
 * Edges that don't fill a full register tile fall back to the scalar path, so
 * arbitrary M/N/K are correct. C reference + NEON microkernel now; AVX2
 * microkernel, bf16/fp16, and a tiled threadpool driver follow.
 */
#ifndef PEREGRINE_TENSOR_KERNELS_GEMM_H
#define PEREGRINE_TENSOR_KERNELS_GEMM_H

#include <stddef.h>
#include "util/arch.h"

typedef void (*pg_sgemm_fn)(size_t M, size_t N, size_t K,
                            const float *A, size_t lda,
                            const float *B, size_t ldb,
                            float *C, size_t ldc);

typedef struct PgGemmDSP {
    pg_sgemm_fn sgemm;
} PgGemmDSP;

typedef struct PgGemmVariant {
    const char  *name;
    pg_sgemm_fn  fn;
    unsigned     req_flags;
} PgGemmVariant;

void pg_gemm_dsp_init(PgGemmDSP *dsp, unsigned cpu_flags);
const PgGemmVariant *pg_gemm_variants(size_t *count);

void pg_sgemm_c(size_t M, size_t N, size_t K,
                const float *A, size_t lda,
                const float *B, size_t ldb,
                float *C, size_t ldc);
#if PG_ARCH_AARCH64
void pg_sgemm_neon(size_t M, size_t N, size_t K,
                   const float *A, size_t lda,
                   const float *B, size_t ldb,
                   float *C, size_t ldc);
#endif
#if PG_ARCH_X86_64
void pg_sgemm_avx2(size_t M, size_t N, size_t K,
                   const float *A, size_t lda,
                   const float *B, size_t ldb,
                   float *C, size_t ldc);
#endif

/*
 * Internal: the shared packed + threaded driver. Each ISA's public entry
 * (pg_sgemm_neon / pg_sgemm_avx2) calls this with its microkernel and tile
 * dimensions (mr x nr). The microkernel computes an mr x nr block of C as
 * C[mr x nr] = sum_{k<kc} a[k][0..mr) (x) b[k][0..nr), reading the pre-packed
 * a/b panels contiguously and overwriting C (one call spans all of K).
 */
typedef void (*pg_sgemm_ukernel_fn)(size_t kc, const float *a, const float *b,
                                    float *c, size_t ldc);
void pg_sgemm_blocked(size_t M, size_t N, size_t K,
                      const float *A, size_t lda,
                      const float *B, size_t ldb,
                      float *C, size_t ldc,
                      size_t mr, size_t nr, pg_sgemm_ukernel_fn ukernel);

#endif /* PEREGRINE_TENSOR_KERNELS_GEMM_H */
