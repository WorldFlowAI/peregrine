/*
 * peregrine - native fp16 GEMM driver (AArch64 FEAT_FP16).
 *
 * Packs A/B as fp16 panels and drives the 8x8 FP16 FMLA microkernel. The
 * microkernel accumulates in fp16 lanes and converts the final tile to f32.
 */
#include "tensor/kernels/gemm/gemm_fp16.h"

#include "util/thread.h"

#include <stdlib.h>

#define MR 8
#define NR 8

extern void pg_fp16gemm_ukernel_8x8_f16(size_t kc, const pg_fp16 *a,
                                        const pg_fp16 *b, float *c,
                                        size_t ldc);

static void pack_a_f16(pg_fp16 *restrict dst, const pg_fp16 *A, size_t lda,
                       size_t ib, size_t K)
{
    for (size_t k = 0; k < K; k++)
        for (size_t r = 0; r < MR; r++)
            dst[k * MR + r] = A[(ib + r) * lda + k];
}

static void pack_b_f16(pg_fp16 *restrict dst, const pg_fp16 *B, size_t ldb,
                       size_t jb, size_t K)
{
    for (size_t k = 0; k < K; k++)
        for (size_t c = 0; c < NR; c++)
            dst[k * NR + c] = B[k * ldb + (jb + c)];
}

typedef struct {
    size_t          N8, K, lda, ldb, ldc;
    const pg_fp16  *A, *B;
    const pg_fp16  *pb;
    float          *C;
} Fp16Job;

static void fp16_row_task(void *vctx, size_t begin, size_t end)
{
    Fp16Job *t = vctx;
    pg_fp16 *pa = malloc(t->K * MR * sizeof(pg_fp16));

    for (size_t ibx = begin; ibx < end; ibx++) {
        size_t ib = ibx * MR;
        if (pa) {
            pack_a_f16(pa, t->A, t->lda, ib, t->K);
            for (size_t jb = 0; jb < t->N8; jb += NR)
                pg_fp16gemm_ukernel_8x8_f16(t->K, pa,
                    t->pb + (jb / NR) * t->K * NR, t->C + ib * t->ldc + jb, t->ldc);
        } else {
            for (size_t i = ib; i < ib + MR; i++)
                for (size_t j = 0; j < t->N8; j++)
                    pg_fp16_scalar(i, j, t->K, t->A, t->lda, t->B, t->ldb,
                                   t->C, t->ldc);
        }
    }
    free(pa);
}

void pg_fp16gemm_f16(size_t M, size_t N, size_t K,
                     const pg_fp16 *A, size_t lda,
                     const pg_fp16 *B, size_t ldb,
                     float *C, size_t ldc)
{
    size_t M8 = M - (M % MR);
    size_t N8 = N - (N % NR);

    if (M8 && N8 && K) {
        size_t nblk = N8 / NR;
        pg_fp16 *pb = malloc(nblk * K * NR * sizeof(pg_fp16));
        if (pb) {
            for (size_t jb = 0; jb < N8; jb += NR)
                pack_b_f16(pb + (jb / NR) * K * NR, B, ldb, jb, K);

            Fp16Job job = { N8, K, lda, ldb, ldc, A, B, pb, C };
            pg_parallel_for(pg_global_threadpool(), M8 / MR, 1, fp16_row_task, &job);
            free(pb);
        } else {
            M8 = 0;
            N8 = 0;
        }
    } else {
        M8 = 0;
        N8 = 0;
    }

    for (size_t i = M8; i < M; i++)
        for (size_t j = 0; j < N; j++)
            pg_fp16_scalar(i, j, K, A, lda, B, ldb, C, ldc);
    for (size_t i = 0; i < M8; i++)
        for (size_t j = N8; j < N; j++)
            pg_fp16_scalar(i, j, K, A, lda, B, ldb, C, ldc);
}
