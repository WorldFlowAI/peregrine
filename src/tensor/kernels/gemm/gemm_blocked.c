/*
 * peregrine - shared packed + threaded GEMM driver.
 *
 * ISA-independent orchestration around a register-blocked microkernel: pack A
 * into mr-row panels and B into nr-column panels (contiguous, so the microkernel
 * reads each depth step as aligned vector loads), drive the microkernel over the
 * full-tile region, and scalar-fill the L-shaped border so arbitrary M/N/K are
 * correct. Parallelised over row-blocks: B is packed once (shared, read-only)
 * and each chunk owns a disjoint stripe of C plus its own packed-A panel, so the
 * worker needs no locking.
 *
 * One K pass: the microkernel overwrites C. Cache tiling over K was measured to
 * give nothing on Apple Silicon (compute-bound through 2048^3); it belongs on
 * hardware where the working set spills, behind a C-accumulating microkernel.
 */
#include "tensor/kernels/gemm/gemm.h"

#include "util/thread.h"

#include <stdlib.h>

/* dst[k*mr + r] = A[(ib+r)*lda + k] : mr row-elements contiguous per depth k */
static void pack_a(float *restrict dst, const float *A, size_t lda,
                   size_t ib, size_t K, size_t mr)
{
    for (size_t k = 0; k < K; k++)
        for (size_t r = 0; r < mr; r++)
            dst[k * mr + r] = A[(ib + r) * lda + k];
}

/* dst[k*nr + c] = B[k*ldb + (jb+c)] : nr col-elements contiguous per depth k */
static void pack_b(float *restrict dst, const float *B, size_t ldb,
                   size_t jb, size_t K, size_t nr)
{
    for (size_t k = 0; k < K; k++)
        for (size_t c = 0; c < nr; c++)
            dst[k * nr + c] = B[k * ldb + (jb + c)];
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
    size_t              N8, K, lda, ldc, mr, nr;
    const float        *A;
    const float        *pb;   /* packed B, shared read-only */
    float              *C;
    pg_sgemm_ukernel_fn uk;
} GemmRowJob;

/* B element at depth k, global column j, recovered from the packed panel. */
static inline float pb_at(const float *pb, size_t K, size_t nr, size_t k, size_t j)
{
    return pb[(j / nr) * K * nr + k * nr + (j % nr)];
}

/* Compute row-blocks [begin, end) (each mr rows) over all full column tiles,
 * using a private packed-A panel. */
static void gemm_row_task(void *vctx, size_t begin, size_t end)
{
    GemmRowJob *t = vctx;
    float *pa = malloc(t->K * t->mr * sizeof(float));

    for (size_t ibx = begin; ibx < end; ibx++) {
        size_t ib = ibx * t->mr;
        if (pa) {
            pack_a(pa, t->A, t->lda, ib, t->K, t->mr);
            for (size_t jb = 0; jb < t->N8; jb += t->nr)
                t->uk(t->K, pa, t->pb + (jb / t->nr) * t->K * t->nr,
                      t->C + ib * t->ldc + jb, t->ldc);
        } else {
            /* allocation failed: still correct, just unvectorised, off the
             * already-packed B */
            for (size_t i = ib; i < ib + t->mr; i++)
                for (size_t j = 0; j < t->N8; j++) {
                    float s = 0.0f;
                    for (size_t k = 0; k < t->K; k++)
                        s += t->A[i * t->lda + k] * pb_at(t->pb, t->K, t->nr, k, j);
                    t->C[i * t->ldc + j] = s;
                }
        }
    }
    free(pa);
}

void pg_sgemm_blocked(size_t M, size_t N, size_t K,
                      const float *A, size_t lda,
                      const float *B, size_t ldb,
                      float *C, size_t ldc,
                      size_t mr, size_t nr, pg_sgemm_ukernel_fn ukernel)
{
    size_t M8 = M - (M % mr);
    size_t N8 = N - (N % nr);

    if (M8 && N8 && K) {
        size_t nblk = N8 / nr;
        float *pb = malloc(nblk * K * nr * sizeof(float));
        if (!pb) {
            pg_sgemm_c(M, N, K, A, lda, B, ldb, C, ldc);
            return;
        }
        for (size_t jb = 0; jb < N8; jb += nr)
            pack_b(pb + (jb / nr) * K * nr, B, ldb, jb, K, nr);

        GemmRowJob job = { N8, K, lda, ldc, mr, nr, A, pb, C, ukernel };
        pg_parallel_for(pg_global_threadpool(), M8 / mr, 1, gemm_row_task, &job);
        free(pb);
    } else {
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
