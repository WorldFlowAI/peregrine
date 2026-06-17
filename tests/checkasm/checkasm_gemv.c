/*
 * peregrine - checkasm module for f32 GEMV (y[M] = A[M,K] @ x[K]).
 *
 * Each output is a length-K dot, validated against a double oracle with a
 * per-row reduction-aware tolerance (scales with sum |a*x|, not |y|). lda is
 * padded so the row stride is exercised. GEMV's 6-arg signature bypasses the
 * clobber trampoline, so the variant is called directly.
 */
#include "checkasm.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util/cpu.h"
#include "util/mem.h"
#include "tensor/kernels/gemv/gemv.h"

#define MAXM 512
#define MAXK 1376
#define LDA  (MAXK + 5)

static float *fbuf(size_t n)
{
    float *p = pg_aligned_alloc(PG_ALIGN, n * sizeof(float));
    if (p) memset(p, 0, n * sizeof(float));
    return p;
}

static void fuzz_variant(const PgGemvVariant *v, float *A, float *x, float *y)
{
    static const struct {
        size_t M, K;
    } edges[] = {
        { 0, 0 }, { 1, 1 }, { 2, 2 }, { 3, 3 }, { 7, 7 },
        { 8, 8 }, { 9, 9 }, { 15, 15 }, { 16, 16 }, { 17, 17 },
        { 31, 31 }, { 33, 33 }, { 64, 64 }, { 512, 512 },
        { 512, 1376 },
    };
    const size_t ne = sizeof edges / sizeof edges[0];
    int ok = 1;

    for (int t = 0; t < 300 && ok; t++) {
        size_t M = (t < (int)ne) ? edges[t].M : checkasm_rand_range(1, MAXM);
        size_t K = (t < (int)ne) ? edges[t].K : checkasm_rand_range(1, MAXK);

        for (size_t i = 0; i < M; i++)
            for (size_t k = 0; k < K; k++) A[i * LDA + k] = checkasm_randf(2.0f);
        for (size_t k = 0; k < K; k++) x[k] = checkasm_randf(2.0f);

        v->fn(M, K, A, LDA, x, y);

        for (size_t i = 0; i < M && ok; i++) {
            double s = 0.0, w = 0.0;
            for (size_t k = 0; k < K; k++) {
                double p = (double)A[i * LDA + k] * (double)x[k];
                s += p;
                w += fabs(p);
            }
            double tol = 16.0 * FLT_EPSILON * w + 1e-6;
            if (fabs((double)y[i] - s) > tol) {
                checkasm_fail("gemv.%s M=%zu K=%zu i=%zu ref=%g got=%g",
                              v->name, M, K, i, s, y[i]);
                ok = 0;
            }
        }
    }
    checkasm_report("sgemv_f32", v->name, ok);
}

static double bench_variant(pg_sgemv_f32_fn fn, const float *A, const float *x,
                            float *y, size_t M, size_t K, double *gflops)
{
    volatile float sink = 0;
    int iters = 1;
    double el = 0.0;
    fn(M, K, A, K, x, y);
    do {
        double t0 = checkasm_now();
        for (int i = 0; i < iters; i++) fn(M, K, A, K, x, y);
        el = checkasm_now() - t0;
        if (el < 0.05) iters *= 2;
    } while (el < 0.05 && iters < (1 << 22));
    sink += y[0];
    (void)sink;
    double per = el / iters;
    *gflops = 2.0 * (double)M * (double)K / per / 1e9;
    return per;
}

void checkasm_check_gemv(void)
{
    size_t nv;
    const PgGemvVariant *v = pg_gemv_variants(&nv);
    unsigned flags = pg_get_cpu_flags();

    float *A = fbuf(MAXM * LDA), *x = fbuf(MAXK), *y = fbuf(MAXM);

    for (size_t i = 0; i < nv; i++)
        if ((flags & v[i].req_flags) == v[i].req_flags)
            fuzz_variant(&v[i], A, x, y);

    if (checkasm_bench_enabled()) {
        static const struct {
            size_t M, K;
        } dims[] = {
            { 512, 512 },
            { 512, 1376 },
            { 1376, 512 },
            { 32000, 512 },
            { 4096, 4096 },
            { 8192, 8192 },
        };
        for (size_t s = 0; s < sizeof dims / sizeof dims[0]; s++) {
            size_t M = dims[s].M;
            size_t K = dims[s].K;
            float *bA = fbuf(M * K), *bx = fbuf(K), *by = fbuf(M);
            for (size_t k = 0; k < M * K; k++) bA[k] = checkasm_randf(1.0f);
            for (size_t k = 0; k < K; k++) bx[k] = checkasm_randf(1.0f);
            char title[48];
            snprintf(title, sizeof title, "sgemv_f32  %zux%zu", M, K);
            checkasm_bench_begin(title, "GFLOP/s");
            for (size_t i = 0; i < nv; i++) {
                if ((flags & v[i].req_flags) != v[i].req_flags) continue;
                double g, per = bench_variant(v[i].fn, bA, bx, by, M, K, &g);
                checkasm_bench_row(v[i].name, per, g);
            }
            checkasm_bench_end();
            pg_aligned_free(bA); pg_aligned_free(bx); pg_aligned_free(by);
        }
    }

    pg_aligned_free(A);
    pg_aligned_free(x);
    pg_aligned_free(y);
}
