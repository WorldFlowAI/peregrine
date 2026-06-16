/*
 * peregrine - native bf16 GEMM driver (AArch64 BFMMLA), the optimized tier.
 *
 * Packs A/B into the BFMMLA block layout (bf16, K zero-padded to a multiple of
 * 4) and drives the 8x8 BFMMLA microkernel, threaded over row-blocks. The
 * L-shaped border falls back to the scalar bf16 path. No locking: each chunk
 * owns a disjoint stripe of C and its own packed-A panel.
 *
 * Layout matches the microkernel:
 *   a[kg]: 4 row-pairs, each 8 bf16 = [r0k0..r0k3, r1k0..r1k3]   (Vn, 2x4)
 *   b[kg]: 4 col-pairs, each 8 bf16 = [k0c0,k0c1, k1c0,k1c1, ...] (Vm, 4x2)
 */
#include "tensor/kernels/gemm/gemm_bf16.h"

#include "util/thread.h"

#include <stdlib.h>

#define MR 8
#define NR 8

extern void pg_bf16gemm_ukernel_8x8_bfmmla(size_t kg, const pg_bf16 *a,
                                           const pg_bf16 *b, float *c,
                                           size_t ldc);

/* a[kg*32 + rp*8 + r*4 + kk] = A[(ib + 2*rp + r)*lda + (kg*4 + kk)] (0-padded) */
static void pack_a_bfmmla(pg_bf16 *restrict dst, const pg_bf16 *A, size_t lda,
                          size_t ib, size_t K, size_t KG)
{
    for (size_t kg = 0; kg < KG; kg++)
        for (size_t rp = 0; rp < 4; rp++)
            for (size_t r = 0; r < 2; r++)
                for (size_t kk = 0; kk < 4; kk++) {
                    size_t k = kg * 4 + kk;
                    dst[kg * 32 + rp * 8 + r * 4 + kk] =
                        (k < K) ? A[(ib + 2 * rp + r) * lda + k] : 0;
                }
}

/* BFMMLA reads Vm as a 2x4 block (col-of-pair c outer, k inner), computing
 * A . B^T: b[kg*32 + cp*8 + c*4 + kk] = B[(kg*4 + kk)*ldb + (jb + 2*cp + c)]. */
static void pack_b_bfmmla(pg_bf16 *restrict dst, const pg_bf16 *B, size_t ldb,
                          size_t jb, size_t K, size_t KG)
{
    for (size_t kg = 0; kg < KG; kg++)
        for (size_t cp = 0; cp < 4; cp++)
            for (size_t c = 0; c < 2; c++)
                for (size_t kk = 0; kk < 4; kk++) {
                    size_t k = kg * 4 + kk;
                    dst[kg * 32 + cp * 8 + c * 4 + kk] =
                        (k < K) ? B[k * ldb + (jb + 2 * cp + c)] : 0;
                }
}

typedef struct {
    size_t         N8, K, KG, lda, ldb, ldc;
    const pg_bf16 *A, *B;
    const pg_bf16 *pb;   /* packed B, shared read-only */
    float         *C;
} Bf16Job;

static void bf16_row_task(void *vctx, size_t begin, size_t end)
{
    Bf16Job *t = vctx;
    pg_bf16 *pa = malloc(t->KG * 32 * sizeof(pg_bf16));

    for (size_t ibx = begin; ibx < end; ibx++) {
        size_t ib = ibx * MR;
        if (pa) {
            pack_a_bfmmla(pa, t->A, t->lda, ib, t->K, t->KG);
            for (size_t jb = 0; jb < t->N8; jb += NR)
                pg_bf16gemm_ukernel_8x8_bfmmla(t->KG, pa,
                    t->pb + (jb / NR) * t->KG * 32, t->C + ib * t->ldc + jb, t->ldc);
        } else {
            for (size_t i = ib; i < ib + MR; i++)
                for (size_t j = 0; j < t->N8; j++)
                    pg_bf16_scalar(i, j, t->K, t->A, t->lda, t->B, t->ldb,
                                   t->C, t->ldc);
        }
    }
    free(pa);
}

void pg_bf16gemm_bfmmla(size_t M, size_t N, size_t K,
                        const pg_bf16 *A, size_t lda,
                        const pg_bf16 *B, size_t ldb,
                        float *C, size_t ldc)
{
    size_t M8 = M - (M % MR);
    size_t N8 = N - (N % NR);
    size_t KG = (K + 3) / 4;

    if (M8 && N8 && K) {
        size_t nblk = N8 / NR;
        pg_bf16 *pb = malloc(nblk * KG * 32 * sizeof(pg_bf16));
        if (pb) {
            for (size_t jb = 0; jb < N8; jb += NR)
                pack_b_bfmmla(pb + (jb / NR) * KG * 32, B, ldb, jb, K, KG);

            Bf16Job job = { N8, K, KG, lda, ldb, ldc, A, B, pb, C };
            pg_parallel_for(pg_global_threadpool(), M8 / MR, 1, bf16_row_task, &job);
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
            pg_bf16_scalar(i, j, K, A, lda, B, ldb, C, ldc);
    for (size_t i = 0; i < M8; i++)
        for (size_t j = N8; j < N; j++)
            pg_bf16_scalar(i, j, K, A, lda, B, ldb, C, ldc);
}
