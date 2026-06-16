/*
 * peregrine - f32 GEMM driver for the AArch64 NEON microkernel.
 *
 * Packs A into 8-row panels and B into 8-column panels (contiguous, so the
 * microkernel reads each depth step as two aligned 4-float loads), then drives
 * the 8x8 register-blocked microkernel over the full-tile region. The L-shaped
 * border that doesn't fill an 8x8 tile falls back to a scalar dot, so arbitrary
 * M/N/K are correct.
 *
 * Parallelised over row-blocks: B is packed once (shared, read-only) and the
 * row-blocks of C are partitioned across the thread pool. Each chunk owns a
 * disjoint stripe of C and its own packed-A panel, so the worker needs no
 * locking. On this hardware the single-thread microkernel already sits near one
 * core's peak (~105 GFLOP/s), so the threads are where the remaining scaling is.
 *
 * v1 uses a single K pass (the microkernel overwrites C). Cache tiling over K
 * was measured to give nothing here (the kernel stays compute-bound through
 * 2048^3) and is left for hardware where it pays. x86 AVX2 microkernel +
 * bf16/fp16 paths are follow-ups.
 */
#include "tensor/kernels/gemm/gemm.h"

#include "util/thread.h"

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

typedef struct {
    size_t       N8, K, lda, ldc;
    const float *A;
    const float *pb;   /* packed B, shared read-only */
    float       *C;
} GemmRowJob;

/* B element at depth k, global column j, recovered from the packed panel. */
static inline float pb_at(const float *pb, size_t K, size_t k, size_t j)
{
    return pb[(j / NR) * K * NR + k * NR + (j % NR)];
}

/* Compute the row-blocks [begin, end) (each MR rows) over all full column
 * tiles, using a private packed-A panel. */
static void gemm_row_task(void *vctx, size_t begin, size_t end)
{
    GemmRowJob *t = vctx;
    float *pa = malloc(t->K * (size_t)MR * sizeof(float));

    for (size_t ibx = begin; ibx < end; ibx++) {
        size_t ib = ibx * MR;
        if (pa) {
            pack_a(pa, t->A, t->lda, ib, t->K);
            for (size_t jb = 0; jb < t->N8; jb += NR)
                pg_sgemm_ukernel_8x8_neon(t->K, pa, t->pb + (jb / NR) * t->K * NR,
                                          t->C + ib * t->ldc + jb, t->ldc);
        } else {
            /* allocation failed: still correct, just unvectorised, straight off
             * the already-packed B */
            for (size_t i = ib; i < ib + MR; i++)
                for (size_t j = 0; j < t->N8; j++) {
                    float s = 0.0f;
                    for (size_t k = 0; k < t->K; k++)
                        s += t->A[i * t->lda + k] * pb_at(t->pb, t->K, k, j);
                    t->C[i * t->ldc + j] = s;
                }
        }
    }
    free(pa);
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
        if (!pb) {
            pg_sgemm_c(M, N, K, A, lda, B, ldb, C, ldc);
            return;
        }
        for (size_t jb = 0; jb < N8; jb += NR)
            pack_b(pb + (jb / NR) * K * NR, B, ldb, jb, K);

        GemmRowJob job = { N8, K, lda, ldc, A, pb, C };
        pg_parallel_for(pg_global_threadpool(), M8 / MR, 1, gemm_row_task, &job);
        free(pb);
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
