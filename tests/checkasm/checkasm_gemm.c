/*
 * peregrine - checkasm module for f32 GEMM (C[M,N] = A[M,K] @ B[K,N]).
 *
 * Validates each variant against a double-precision oracle. The per-cell
 * tolerance scales with the summation WORK (sum of |a*b| over k), not the
 * result magnitude: a GEMM cell is a length-K dot that can cancel heavily, so
 * the SIMD-vs-scalar difference is ~eps*work, which a result-relative bound
 * would undershoot.
 *
 * Leading dimensions are deliberately padded (lda != K, ldb/ldc != N) so stride
 * handling is exercised. GEMM's 9-arg signature bypasses the clobber
 * trampoline, so (like rope) the variant is called directly.
 */
#include "checkasm.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util/cpu.h"
#include "util/mem.h"
#include "tensor/kernels/gemm/gemm.h"

#define MAXM 96
#define MAXN 96
#define MAXK 96
#define LDA  (MAXK + 3)
#define LDB  (MAXN + 5)
#define LDC  (MAXN + 7)

static float *fbuf(size_t n)
{
    float *p = pg_aligned_alloc(PG_ALIGN, n * sizeof(float));
    if (p) memset(p, 0, n * sizeof(float));
    return p;
}

static void fuzz_variant(const PgGemmVariant *v, float *A, float *B, float *C,
                         double *ref, double *work)
{
    static const size_t edges[] = { 0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33 };
    const size_t ne = sizeof edges / sizeof edges[0];
    int ok = 1;

    for (int t = 0; t < 300 && ok; t++) {
        size_t M = (t < (int)ne) ? edges[t] : checkasm_rand_range(1, MAXM);
        size_t N = (t < (int)ne) ? edges[t] : checkasm_rand_range(1, MAXN);
        size_t K = (t < (int)ne) ? edges[t] : checkasm_rand_range(1, MAXK);

        for (size_t i = 0; i < M; i++)
            for (size_t k = 0; k < K; k++) A[i * LDA + k] = checkasm_randf(2.0f);
        for (size_t k = 0; k < K; k++)
            for (size_t j = 0; j < N; j++) B[k * LDB + j] = checkasm_randf(2.0f);

        for (size_t i = 0; i < M; i++)
            for (size_t j = 0; j < N; j++) {
                double s = 0.0, w = 0.0;
                for (size_t k = 0; k < K; k++) {
                    double p = (double)A[i * LDA + k] * (double)B[k * LDB + j];
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
                double got = C[i * LDC + j];
                if (fabs(got - ref[i * MAXN + j]) > tol) {
                    checkasm_fail("gemm.%s M=%zu N=%zu K=%zu i=%zu j=%zu ref=%g got=%g",
                                  v->name, M, N, K, i, j, ref[i * MAXN + j], got);
                    ok = 0;
                    break;
                }
            }
    }
    checkasm_report("sgemm_f32", v->name, ok);
}

static double bench_variant(pg_sgemm_fn fn, const float *A, const float *B,
                            float *C, size_t M, size_t N, size_t K, double *gflops)
{
    volatile float sink = 0;
    int iters = 1;
    double el = 0.0;
    fn(M, N, K, A, K, B, N, C, N);
    do {
        double t0 = checkasm_now();
        for (int i = 0; i < iters; i++) fn(M, N, K, A, K, B, N, C, N);
        el = checkasm_now() - t0;
        if (el < 0.05) iters *= 2;
    } while (el < 0.05 && iters < (1 << 20));
    sink += C[0];
    (void)sink;
    double per = el / iters;
    *gflops = 2.0 * (double)M * (double)N * (double)K / per / 1e9;
    return per;
}

void checkasm_check_gemm(void)
{
    size_t nv;
    const PgGemmVariant *v = pg_gemm_variants(&nv);
    unsigned flags = pg_get_cpu_flags();

    float  *A    = fbuf(MAXM * LDA), *B = fbuf(MAXK * LDB), *C = fbuf(MAXM * LDC);
    double *ref  = malloc(MAXM * MAXN * sizeof(double));
    double *work = malloc(MAXM * MAXN * sizeof(double));

    for (size_t i = 0; i < nv; i++)
        if ((flags & v[i].req_flags) == v[i].req_flags)
            fuzz_variant(&v[i], A, B, C, ref, work);

    if (checkasm_bench_enabled()) {
        static const size_t dims[] = { 256, 512, 1024, 2048 };
        for (size_t s = 0; s < sizeof dims / sizeof dims[0]; s++) {
            size_t n = dims[s];
            float *bA = fbuf(n * n), *bB = fbuf(n * n), *bC = fbuf(n * n);
            for (size_t k = 0; k < n * n; k++) { bA[k] = checkasm_randf(1.0f); bB[k] = checkasm_randf(1.0f); }
            char title[48];
            snprintf(title, sizeof title, "sgemm_f32  %zux%zux%zu", n, n, n);
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
