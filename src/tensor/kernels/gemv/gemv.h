/*
 * peregrine - single-precision GEMV (matrix-vector multiply).
 *
 *   y[M] = A[M,K] @ x[K]      (row-major, y is overwritten)
 *
 * lda >= K is the row stride (elements). A GEMV row is exactly a length-K dot
 * product, so rather than a separate microkernel GEMV reuses the per-ISA dot
 * kernels (already hand-written and checked) and parallelises the rows across
 * the thread pool. GEMV is memory-bound (A is streamed once, x stays hot), so
 * the win over scalar is mostly the vectorised reduction plus the cores.
 *
 * Transpose (A^T @ x) and alpha/beta are mechanical follow-ups.
 */
#ifndef PEREGRINE_TENSOR_KERNELS_GEMV_H
#define PEREGRINE_TENSOR_KERNELS_GEMV_H

#include <stddef.h>
#include "util/arch.h"
#include "tensor/kernels/dot/dot.h"

typedef void (*pg_sgemv_f32_fn)(size_t M, size_t K, const float *A, size_t lda,
                                const float *x, float *y);

typedef struct PgGemvDSP {
    pg_sgemv_f32_fn sgemv;
} PgGemvDSP;

typedef struct PgGemvVariant {
    const char     *name;
    pg_sgemv_f32_fn fn;
    unsigned        req_flags;
} PgGemvVariant;

void pg_gemv_dsp_init(PgGemvDSP *dsp, unsigned cpu_flags);
const PgGemvVariant *pg_gemv_variants(size_t *count);

void pg_sgemv_c(size_t M, size_t K, const float *A, size_t lda,
                const float *x, float *y);
#if PG_ARCH_AARCH64
void pg_sgemv_neon(size_t M, size_t K, const float *A, size_t lda,
                   const float *x, float *y);
#endif
#if PG_ARCH_X86_64
void pg_sgemv_avx2(size_t M, size_t K, const float *A, size_t lda,
                   const float *x, float *y);
#endif

/* Internal: threaded dot-per-row driver, shared by every variant (each passes
 * its ISA's dot kernel). */
void pg_sgemv_driver(size_t M, size_t K, const float *A, size_t lda,
                     const float *x, float *y, pg_dot_f32_fn dot);

#endif /* PEREGRINE_TENSOR_KERNELS_GEMV_H */
