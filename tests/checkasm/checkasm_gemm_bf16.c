/*
 * peregrine - checkasm module for bf16-input GEMM (C f32 = A bf16 @ B bf16).
 *
 * Inputs are random f32 rounded to bf16 (round-to-nearest-even). The oracle
 * computes in double from the EXACT f32 value of each stored bf16, so the only
 * slack vs the kernel is f32 accumulation order -> a reduction-aware tolerance
 * (scales with summation work). Padded leading dims; 8-arg signature bypasses
 * the clobber trampoline so the variant is called directly.
 */
#include "checkasm.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util/cpu.h"
#include "util/mem.h"
#include "tensor/kernels/gemm/gemm_bf16.h"

#define MAXM 96
#define MAXN 96
#define MAXK 96
#define LDA  (MAXK + 3)
#define LDB  (MAXN + 5)
#define LDC  (MAXN + 7)

static float bf16_to_f32(pg_bf16 h)
{
    uint32_t bits = (uint32_t)h << 16;
    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

/* round-to-nearest-even f32 -> bf16 (finite test data only) */
static pg_bf16 f32_to_bf16(float f)
{
    uint32_t x;
    memcpy(&x, &f, sizeof x);
    uint32_t rounding = 0x7fffu + ((x >> 16) & 1u);
    return (pg_bf16)((x + rounding) >> 16);
}

static pg_bf16 *bbuf(size_t n)
{
    pg_bf16 *p = pg_aligned_alloc(PG_ALIGN, n * sizeof(pg_bf16));
    if (p) memset(p, 0, n * sizeof(pg_bf16));
    return p;
}

static void fuzz_variant(const PgBf16GemmVariant *v, pg_bf16 *A, pg_bf16 *B,
                         float *C, double *ref, double *work)
{
    static const size_t edges[] = { 0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33 };
    const size_t ne = sizeof edges / sizeof edges[0];
    int ok = 1;

    for (int t = 0; t < 300 && ok; t++) {
        size_t M = (t < (int)ne) ? edges[t] : checkasm_rand_range(1, MAXM);
        size_t N = (t < (int)ne) ? edges[t] : checkasm_rand_range(1, MAXN);
        size_t K = (t < (int)ne) ? edges[t] : checkasm_rand_range(1, MAXK);

        for (size_t i = 0; i < M; i++)
            for (size_t k = 0; k < K; k++) A[i * LDA + k] = f32_to_bf16(checkasm_randf(2.0f));
        for (size_t k = 0; k < K; k++)
            for (size_t j = 0; j < N; j++) B[k * LDB + j] = f32_to_bf16(checkasm_randf(2.0f));

        for (size_t i = 0; i < M; i++)
            for (size_t j = 0; j < N; j++) {
                double s = 0.0, w = 0.0;
                for (size_t k = 0; k < K; k++) {
                    double p = (double)bf16_to_f32(A[i * LDA + k]) *
                               (double)bf16_to_f32(B[k * LDB + j]);
                    s += p;
                    w += fabs(p);
                }
                ref[i * MAXN + j] = s;
                work[i * MAXN + j] = w;
            }

        v->fn(M, N, K, A, LDA, B, LDB, C, LDC);

        for (size_t i = 0; i < M && ok; i++)
            for (size_t j = 0; j < N; j++) {
                double tol = 16.0 * FLT_EPSILON * work[i * MAXN + j] + 1e-6;
                if (fabs((double)C[i * LDC + j] - ref[i * MAXN + j]) > tol) {
                    checkasm_fail("bf16gemm.%s M=%zu N=%zu K=%zu i=%zu j=%zu ref=%g got=%g",
                                  v->name, M, N, K, i, j, ref[i * MAXN + j], C[i * LDC + j]);
                    ok = 0;
                    break;
                }
            }
    }
    checkasm_report("bf16gemm", v->name, ok);
}

static double bench_variant(pg_bf16gemm_fn fn, const pg_bf16 *A, const pg_bf16 *B,
                            float *C, size_t M, size_t N, size_t K, double *gflops)
{
    int iters = 1;
    double el = 0.0;
    fn(M, N, K, A, K, B, N, C, N);
    do {
        double t0 = checkasm_now();
        for (int i = 0; i < iters; i++) fn(M, N, K, A, K, B, N, C, N);
        el = checkasm_now() - t0;
        if (el < 0.05) iters *= 2;
    } while (el < 0.05 && iters < (1 << 20));
    double per = el / iters;
    *gflops = 2.0 * (double)M * (double)N * (double)K / per / 1e9;
    return per;
}

void checkasm_check_gemm_bf16(void)
{
    size_t nv;
    const PgBf16GemmVariant *v = pg_bf16gemm_variants(&nv);
    unsigned flags = pg_get_cpu_flags();

    pg_bf16 *A   = bbuf(MAXM * LDA), *B = bbuf(MAXK * LDB);
    float   *C   = pg_aligned_alloc(PG_ALIGN, MAXM * LDC * sizeof(float));
    double  *ref = malloc(MAXM * MAXN * sizeof(double));
    double  *work = malloc(MAXM * MAXN * sizeof(double));

    for (size_t i = 0; i < nv; i++)
        if ((flags & v[i].req_flags) == v[i].req_flags)
            fuzz_variant(&v[i], A, B, C, ref, work);

    if (checkasm_bench_enabled()) {
        static const size_t dims[] = { 512, 1024 };
        for (size_t s = 0; s < sizeof dims / sizeof dims[0]; s++) {
            size_t n = dims[s];
            pg_bf16 *bA = bbuf(n * n), *bB = bbuf(n * n);
            float *bC = pg_aligned_alloc(PG_ALIGN, n * n * sizeof(float));
            for (size_t k = 0; k < n * n; k++) {
                bA[k] = f32_to_bf16(checkasm_randf(1.0f));
                bB[k] = f32_to_bf16(checkasm_randf(1.0f));
            }
            char title[48];
            snprintf(title, sizeof title, "bf16gemm  %zux%zux%zu", n, n, n);
            checkasm_bench_begin(title, "GFLOP/s");
            for (size_t i = 0; i < nv; i++) {
                if ((flags & v[i].req_flags) != v[i].req_flags) continue;
                double g, per = bench_variant(v[i].fn, bA, bB, bC, n, n, n, &g);
                checkasm_bench_row(v[i].name, per, g);
            }
            checkasm_bench_end();
            pg_aligned_free(bA); pg_aligned_free(bB); pg_aligned_free(bC);
        }
    }

    free(ref);
    free(work);
    pg_aligned_free(A);
    pg_aligned_free(B);
    pg_aligned_free(C);
}
