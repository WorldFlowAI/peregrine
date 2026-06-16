/*
 * peregrine - f32 GEMM driver for the AArch64 NEON microkernel.
 *
 * Packs A into 8-row panels and B into 8-column panels (contiguous, so the
 * microkernel reads each depth step as two aligned 4-float loads), then drives
 * the 8x8 register-blocked microkernel over the full-tile region. The L-shaped
 * border that doesn't fill an 8x8 tile falls back to a scalar dot, so arbitrary
 * M/N/K are correct.
 *
 * v1 uses a single K pass (the microkernel overwrites C). Cache tiling over
 * K/M/N (with a C-accumulating microkernel) and a threaded driver are
 * follow-ups; this already turns the naive strided B access into contiguous
 * panel reads, which is the bulk of the win over the scalar reference.
 */
#include "tensor/kernels/gemm/gemm.h"

#include <stdlib.h>

#define MR 8
#define NR 8

extern void pg_sgemm_ukernel_8x8_neon(size_t kc, const float *a, const float *b,
                                       float *c, size_t ldc);

/* dst[k*MR + r] = A[(ib+r)*lda + k] : 8 row-elements contiguous per depth k */
static void pack_a(float *restrict dst, const float *A, size_t lda,
                   size_t ib, size_t K)
{
    for (size_t k = 0; k < K; k++)
        for (size_t r = 0; r < MR; r++)
            dst[k * MR + r] = A[(ib + r) * lda + k];
}

/* dst[k*NR + c] = B[k*ldb + (jb+c)] : 8 col-elements contiguous per depth k */
static void pack_b(float *restrict dst, const float *B, size_t ldb,
                   size_t jb, size_t K)
{
    for (size_t k = 0; k < K; k++)
        for (size_t c = 0; c < NR; c++)
            dst[k * NR + c] = B[k * ldb + (jb + c)];
}

static void scalar_cell(size_t i, size_t j, size_t K,
                        const float *A, size_t lda, const float *B, size_t ldb,
                        float *C, size_t ldc)
{
    float s = 0.0f;
    for (size_t k = 0; k < K; k++)
        s += A[i * lda + k] * B[k * ldb + j];
    C[i * ldc + j] = s;
}

void pg_sgemm_neon(size_t M, size_t N, size_t K,
                   const float *A, size_t lda,
                   const float *B, size_t ldb,
                   float *C, size_t ldc)
{
    size_t M8 = M - (M % MR);
    size_t N8 = N - (N % NR);

    if (M8 && N8 && K) {
        size_t nblk = N8 / NR;
        float *pb = malloc(nblk * K * (size_t)NR * sizeof(float));
        float *pa = malloc(K * (size_t)MR * sizeof(float));
        if (!pb || !pa) {
            free(pb);
            free(pa);
            pg_sgemm_c(M, N, K, A, lda, B, ldb, C, ldc);
            return;
        }
        for (size_t jb = 0; jb < N8; jb += NR)
            pack_b(pb + (jb / NR) * K * NR, B, ldb, jb, K);
        for (size_t ib = 0; ib < M8; ib += MR) {
            pack_a(pa, A, lda, ib, K);
            for (size_t jb = 0; jb < N8; jb += NR)
                pg_sgemm_ukernel_8x8_neon(K, pa, pb + (jb / NR) * K * NR,
                                          C + ib * ldc + jb, ldc);
        }
        free(pb);
        free(pa);
    } else {
        /* no full 8x8 tile fits; the whole product is "border" */
        M8 = 0;
        N8 = 0;
    }

    /* L-shaped border: rows [M8,M) over all columns, then the right strip */
    for (size_t i = M8; i < M; i++)
        for (size_t j = 0; j < N; j++)
            scalar_cell(i, j, K, A, lda, B, ldb, C, ldc);
    for (size_t i = 0; i < M8; i++)
        for (size_t j = N8; j < N; j++)
            scalar_cell(i, j, K, A, lda, B, ldb, C, ldc);
}
