/*
 * peregrine - shared threaded GEMV driver.
 *
 * y[i] = dot(A row i, x). Rows are independent and write disjoint y entries, so
 * they partition cleanly across the thread pool with no locking. The grain is a
 * batch of rows (so claiming the shared cursor is cheap relative to the work).
 */
#include "tensor/kernels/gemv/gemv.h"

#include "util/thread.h"

#define PG_GEMV_PAR_MIN_M 1024
#define PG_GEMV_PAR_GRAIN 128

typedef struct {
    size_t        K, lda;
    const float  *A;
    const float  *x;
    float        *y;
    pg_dot_f32_fn dot;
} GemvJob;

static void gemv_task(void *vctx, size_t begin, size_t end)
{
    GemvJob *t = vctx;
    for (size_t i = begin; i < end; i++)
        t->y[i] = t->dot(t->A + i * t->lda, t->x, t->K);
}

void pg_sgemv_driver(size_t M, size_t K, const float *A, size_t lda,
                     const float *x, float *y, pg_dot_f32_fn dot)
{
    GemvJob job = { K, lda, A, x, y, dot };
    if (M < PG_GEMV_PAR_MIN_M) {
        gemv_task(&job, 0, M);
        return;
    }
    pg_parallel_for(pg_global_threadpool(), M, PG_GEMV_PAR_GRAIN, gemv_task, &job);
}
