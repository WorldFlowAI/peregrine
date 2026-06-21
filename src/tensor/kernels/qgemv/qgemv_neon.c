/*
 * peregrine - Q8_0 GEMV, AArch64 NEON entry point.
 * Threaded row-block driver over hand-written fused dequant-dot kernels.
 */
#include "tensor/kernels/qgemv/qgemv.h"

#include "util/thread.h"

#define PG_Q8_0_GEMV_PAR_MIN_M 1024
#define PG_Q8_0_GEMV_PAR_GRAIN 256

typedef struct Q8GemvJob {
    size_t K;
    size_t row_bytes;
    const unsigned char *A;
    const float *x;
    float *y;
} Q8GemvJob;

static void q8_0_gemv_task(void *vctx, size_t begin, size_t end)
{
    Q8GemvJob *t = vctx;
    size_t i = begin;

    for (; i + 1 < end; i += 2) {
        const unsigned char *row0 = t->A + i * t->row_bytes;
        const unsigned char *row1 = row0 + t->row_bytes;

        pg_q8_0_gemv_2x_neon(row0, row1, t->x, t->K, t->y + i);
    }
    if (i < end)
        t->y[i] = pg_q8_0_dot_f32_neon(t->A + i * t->row_bytes,
                                        t->x, t->K);
}

void pg_q8_0_gemv_neon(size_t M, size_t K, const void *A, size_t row_bytes,
                       const float *x, float *y)
{
    Q8GemvJob job = { K, row_bytes, A, x, y };

    if (M < PG_Q8_0_GEMV_PAR_MIN_M) {
        q8_0_gemv_task(&job, 0, M);
        return;
    }
    pg_parallel_for(pg_global_threadpool(), M, PG_Q8_0_GEMV_PAR_GRAIN,
                    q8_0_gemv_task, &job);
}
