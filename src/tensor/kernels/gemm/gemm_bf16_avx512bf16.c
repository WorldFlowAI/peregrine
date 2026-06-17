/*
 * peregrine - native bf16 GEMM driver, x86-64 AVX512-BF16.
 *
 * Packs A/B into K-pair bf16 panels and drives a 6x16 VDPBF16PS microkernel.
 * Each dword lane holds two bf16 values for one column/row pair.
 */
#include "tensor/kernels/gemm/gemm_bf16.h"

#include "util/thread.h"

#include <stdlib.h>

#define MR 6
#define NR 16

extern void pg_bf16gemm_ukernel_6x16_avx512bf16(size_t kg, const pg_bf16 *a,
                                                const pg_bf16 *b, float *c,
                                                size_t ldc);

static void pack_a_avx512bf16(pg_bf16 *restrict dst, const pg_bf16 *A,
                              size_t lda, size_t ib, size_t K, size_t KG)
{
    for (size_t kg = 0; kg < KG; kg++)
        for (size_t r = 0; r < MR; r++)
            for (size_t kk = 0; kk < 2; kk++) {
                size_t k = kg * 2 + kk;
                dst[kg * MR * 2 + r * 2 + kk] =
                    (k < K) ? A[(ib + r) * lda + k] : 0;
            }
}

static void pack_b_avx512bf16(pg_bf16 *restrict dst, const pg_bf16 *B,
                              size_t ldb, size_t jb, size_t K, size_t KG)
{
    for (size_t kg = 0; kg < KG; kg++)
        for (size_t c = 0; c < NR; c++)
            for (size_t kk = 0; kk < 2; kk++) {
                size_t k = kg * 2 + kk;
                dst[kg * NR * 2 + c * 2 + kk] =
                    (k < K) ? B[k * ldb + (jb + c)] : 0;
            }
}

typedef struct {
    size_t         N16, K, KG, lda, ldb, ldc;
    const pg_bf16 *A, *B;
    const pg_bf16 *pb;
    float         *C;
} Bf16Avx512Job;

static void bf16_avx512_row_task(void *vctx, size_t begin, size_t end)
{
    Bf16Avx512Job *t = vctx;
    pg_bf16 *pa = malloc(t->KG * MR * 2 * sizeof(pg_bf16));

    for (size_t ibx = begin; ibx < end; ibx++) {
        size_t ib = ibx * MR;
        if (pa) {
            pack_a_avx512bf16(pa, t->A, t->lda, ib, t->K, t->KG);
            for (size_t jb = 0; jb < t->N16; jb += NR)
                pg_bf16gemm_ukernel_6x16_avx512bf16(t->KG, pa,
                    t->pb + (jb / NR) * t->KG * NR * 2,
                    t->C + ib * t->ldc + jb, t->ldc);
        } else {
            for (size_t i = ib; i < ib + MR; i++)
                for (size_t j = 0; j < t->N16; j++)
                    pg_bf16_scalar(i, j, t->K, t->A, t->lda, t->B, t->ldb,
                                   t->C, t->ldc);
        }
    }
    free(pa);
}

void pg_bf16gemm_avx512bf16(size_t M, size_t N, size_t K,
                            const pg_bf16 *A, size_t lda,
                            const pg_bf16 *B, size_t ldb,
                            float *C, size_t ldc)
{
    size_t M6 = M - (M % MR);
    size_t N16 = N - (N % NR);
    size_t KG = (K + 1) / 2;

    if (M6 && N16 && K) {
        size_t nblk = N16 / NR;
        pg_bf16 *pb = malloc(nblk * KG * NR * 2 * sizeof(pg_bf16));
        if (pb) {
            for (size_t jb = 0; jb < N16; jb += NR)
                pack_b_avx512bf16(pb + (jb / NR) * KG * NR * 2, B, ldb, jb, K, KG);

            Bf16Avx512Job job = { N16, K, KG, lda, ldb, ldc, A, B, pb, C };
            pg_parallel_for(pg_global_threadpool(), M6 / MR, 1,
                            bf16_avx512_row_task, &job);
            free(pb);
        } else {
            M6 = 0;
            N16 = 0;
        }
    } else {
        M6 = 0;
        N16 = 0;
    }

    for (size_t i = M6; i < M; i++)
        for (size_t j = 0; j < N; j++)
            pg_bf16_scalar(i, j, K, A, lda, B, ldb, C, ldc);
    for (size_t i = 0; i < M6; i++)
        for (size_t j = N16; j < N; j++)
            pg_bf16_scalar(i, j, K, A, lda, B, ldb, C, ldc);
}
