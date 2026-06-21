/*
 * peregrine - quantized-weight GEMV for GGUF block formats.
 *
 *   y[M] = dequant(A[M,K]) @ x[K]
 *
 * The matrix is row-major in GGUF's on-disk block layout. The row stride is
 * bytes, because Q8_0/Q4_K rows are packed block streams rather than scalar
 * arrays. Dot functions consume one packed row.
 */
#ifndef PEREGRINE_TENSOR_KERNELS_QGEMV_H
#define PEREGRINE_TENSOR_KERNELS_QGEMV_H

#include <stddef.h>
#include "util/arch.h"

#define PG_Q8_0_BLOCK 32u
#define PG_Q8_0_BLOCK_BYTES 34u
#define PG_Q4_K_BLOCK 256u
#define PG_Q4_K_BLOCK_BYTES 144u

typedef float (*pg_qdot_f32_fn)(const void *row, const float *x, size_t K);
typedef void (*pg_qgemv_f32_fn)(size_t M, size_t K, const void *A,
                                size_t row_bytes, const float *x, float *y);

typedef struct PgQgemvDSP {
    pg_qgemv_f32_fn q8_0;
    pg_qgemv_f32_fn q4_k;
    pg_qdot_f32_fn q8_0_dot;
    pg_qdot_f32_fn q4_k_dot;
} PgQgemvDSP;

typedef struct PgQgemvVariant {
    const char      *name;
    pg_qgemv_f32_fn fn;
    unsigned         req_flags;
} PgQgemvVariant;

void pg_qgemv_dsp_init(PgQgemvDSP *dsp, unsigned cpu_flags);
const PgQgemvVariant *pg_q8_0_gemv_variants(size_t *count);
const PgQgemvVariant *pg_q4_k_gemv_variants(size_t *count);

size_t pg_q8_0_row_bytes(size_t K);
size_t pg_q4_k_row_bytes(size_t K);

void pg_q8_0_dequant_row(const void *row, float *dst, size_t K);
void pg_q4_k_dequant_row(const void *row, float *dst, size_t K);

float pg_q8_0_dot_f32_c(const void *row, const float *x, size_t K);
float pg_q4_k_dot_f32_c(const void *row, const float *x, size_t K);

void pg_q8_0_gemv_c(size_t M, size_t K, const void *A, size_t row_bytes,
                    const float *x, float *y);
void pg_q4_k_gemv_c(size_t M, size_t K, const void *A, size_t row_bytes,
                    const float *x, float *y);

#if PG_ARCH_AARCH64
float pg_q8_0_dot_f32_neon(const void *row, const float *x, size_t K);
void pg_q8_0_gemv_2x_neon(const void *row0, const void *row1,
                          const float *x, size_t K, float *y);
void pg_q8_0_gemv_neon(size_t M, size_t K, const void *A, size_t row_bytes,
                       const float *x, float *y);
#endif

void pg_qgemv_driver(size_t M, size_t K, const void *A, size_t row_bytes,
                     const float *x, float *y, pg_qdot_f32_fn dot);

#endif /* PEREGRINE_TENSOR_KERNELS_QGEMV_H */
