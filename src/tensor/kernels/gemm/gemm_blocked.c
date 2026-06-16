/*
 * peregrine - shared packed + threaded GEMM driver.
 *
 * ISA- and dtype-independent orchestration around a register-blocked f32
 * microkernel: pack A into mr-row panels and B into nr-column panels
 * (contiguous f32, so the microkernel reads aligned vector loads), drive the
 * microkernel over the full-tile region, and scalar-fill the L-shaped border so
 * arbitrary M/N/K are correct. Parallelised over row-blocks: B is packed once
 * (shared, read-only) and each chunk owns a disjoint stripe of C plus its own
 * packed-A panel, so the worker needs no locking.
 *
 * The input matrix type is abstracted behind pack/scalar callbacks (see
 * pg_gemm_blocked_generic), so the f32 and bf16/fp16 GEMMs reuse this one driver
 * and the one f32 microkernel; the callback converts into the f32 panels.
 *
 * One K pass: the microkernel overwrites C. Cache tiling over K was measured to
 * give nothing on Apple Silicon (compute-bound through 2048^3); it belongs on
 * hardware where the working set spills, behind a C-accumulating microkernel.
 */
#include "tensor/kernels/gemm/gemm.h"

#include "util/thread.h"

#include <stdlib.h>

typedef struct {
    size_t              N8, K, lda, ldb, ldc, mr, nr;
    const void         *A;
    const void         *B;
    const float        *pb;   /* packed B, shared read-only */
    float              *C;
    pg_sgemm_ukernel_fn uk;
    pg_gemm_pack_fn     pack_a;
    pg_gemm_scalar_fn   scalar;
} GemmRowJob;

/* Compute row-blocks [begin, end) (each mr rows) over all full column tiles,
 * using a private packed-A panel. */
static void gemm_row_task(void *vctx, size_t begin, size_t end)
{
    GemmRowJob *t = vctx;
    float *pa = malloc(t->K * t->mr * sizeof(float));

    for (size_t ibx = begin; ibx < end; ibx++) {
        size_t ib = ibx * t->mr;
        if (pa) {
            t->pack_a(pa, t->A, t->lda, ib, t->K, t->mr);
            for (size_t jb = 0; jb < t->N8; jb += t->nr)
                t->uk(t->K, pa, t->pb + (jb / t->nr) * t->K * t->nr,
                      t->C + ib * t->ldc + jb, t->ldc);
        } else {
            /* allocation failed: fall back to the scalar path for these rows */
            for (size_t i = ib; i < ib + t->mr; i++)
                for (size_t j = 0; j < t->N8; j++)
                    t->scalar(i, j, t->K, t->A, t->lda, t->B, t->ldb, t->C, t->ldc);
        }
    }
    free(pa);
}

void pg_gemm_blocked_generic(size_t M, size_t N, size_t K,
                             const void *A, size_t lda,
                             const void *B, size_t ldb,
                             float *C, size_t ldc,
                             size_t mr, size_t nr, pg_sgemm_ukernel_fn ukernel,
                             pg_gemm_pack_fn pack_a, pg_gemm_pack_fn pack_b,
                             pg_gemm_scalar_fn scalar)
{
    size_t M8 = M - (M % mr);
    size_t N8 = N - (N % nr);

    if (M8 && N8 && K) {
        size_t nblk = N8 / nr;
        float *pb = malloc(nblk * K * nr * sizeof(float));
        if (pb) {
            for (size_t jb = 0; jb < N8; jb += nr)
                pack_b(pb + (jb / nr) * K * nr, B, ldb, jb, K, nr);

            GemmRowJob job = { N8, K, lda, ldb, ldc, mr, nr, A, B, pb, C,
                               ukernel, pack_a, scalar };
            pg_parallel_for(pg_global_threadpool(), M8 / mr, 1, gemm_row_task, &job);
            free(pb);
        } else {
            M8 = 0;   /* allocation failed: full scalar path below */
            N8 = 0;
        }
    } else {
        M8 = 0;
        N8 = 0;
    }

    /* L-shaped border (or everything, if no full tile / pack failed): rows
     * [M8,M) over all columns, then the right strip */
    for (size_t i = M8; i < M; i++)
        for (size_t j = 0; j < N; j++)
            scalar(i, j, K, A, lda, B, ldb, C, ldc);
    for (size_t i = 0; i < M8; i++)
        for (size_t j = N8; j < N; j++)
            scalar(i, j, K, A, lda, B, ldb, C, ldc);
}

/* ---- f32 callbacks + public f32 entry ----------------------------------- */

static void f32_pack_a(float *restrict dst, const void *srcv, size_t lda,
                       size_t ib, size_t K, size_t mr)
{
    const float *A = srcv;
    for (size_t k = 0; k < K; k++)
        for (size_t r = 0; r < mr; r++)
            dst[k * mr + r] = A[(ib + r) * lda + k];
}

static void f32_pack_b(float *restrict dst, const void *srcv, size_t ldb,
                       size_t jb, size_t K, size_t nr)
{
    const float *B = srcv;
    for (size_t k = 0; k < K; k++)
        for (size_t c = 0; c < nr; c++)
            dst[k * nr + c] = B[k * ldb + (jb + c)];
}

static void f32_scalar(size_t i, size_t j, size_t K,
                       const void *Av, size_t lda, const void *Bv, size_t ldb,
                       float *C, size_t ldc)
{
    const float *A = Av, *B = Bv;
    float s = 0.0f;
    for (size_t k = 0; k < K; k++)
        s += A[i * lda + k] * B[k * ldb + j];
    C[i * ldc + j] = s;
}

void pg_sgemm_blocked(size_t M, size_t N, size_t K,
                      const float *A, size_t lda,
                      const float *B, size_t ldb,
                      float *C, size_t ldc,
                      size_t mr, size_t nr, pg_sgemm_ukernel_fn ukernel)
{
    pg_gemm_blocked_generic(M, N, K, A, lda, B, ldb, C, ldc, mr, nr, ukernel,
                            f32_pack_a, f32_pack_b, f32_scalar);
}
