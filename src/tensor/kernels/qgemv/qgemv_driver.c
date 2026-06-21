/*
 * peregrine - shared threaded quantized GEMV driver.
 */
#include "tensor/kernels/qgemv/qgemv.h"

#include "util/thread.h"

#define PG_QGEMV_PAR_MIN_M 1024
#define PG_QGEMV_PAR_GRAIN 256

typedef struct QgemvJob {
    size_t K;
    size_t row_bytes;
    const unsigned char *A;
    const float *x;
    float *y;
    pg_qdot_f32_fn dot;
} QgemvJob;

static void qgemv_task(void *vctx, size_t begin, size_t end)
{
    QgemvJob *t = vctx;

    for (size_t i = begin; i < end; i++)
        t->y[i] = t->dot(t->A + i * t->row_bytes, t->x, t->K);
}

void pg_qgemv_driver(size_t M, size_t K, const void *A, size_t row_bytes,
                     const float *x, float *y, pg_qdot_f32_fn dot)
{
    QgemvJob job = { K, row_bytes, A, x, y, dot };

    if (M < PG_QGEMV_PAR_MIN_M) {
        qgemv_task(&job, 0, M);
        return;
    }
    pg_parallel_for(pg_global_threadpool(), M, PG_QGEMV_PAR_GRAIN,
                    qgemv_task, &job);
}
