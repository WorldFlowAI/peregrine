/*
 * peregrine - checkasm module for fp16-input GEMM (C f32 = A fp16 @ B fp16).
 *
 * The oracle computes in double from the exact value of each stored fp16. The
 * storage paths accumulate in f32; the native ARM FP16 path accumulates in fp16
 * lanes, so its tolerance scales with both K and sum(abs(term)).
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
#include "tensor/kernels/gemm/gemm_fp16.h"

#define MAXM 96
#define MAXN 96
#define MAXK 96
#define LDA  (MAXK + 3)
#define LDB  (MAXN + 5)
#define LDC  (MAXN + 7)

/* round-to-nearest-even f32 -> fp16 (finite test data only) */
static pg_fp16 f32_to_fp16(float f)
{
    uint32_t x;
    memcpy(&x, &f, sizeof x);

    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t absx = x & 0x7fffffffu;
    if (absx >= 0x7f800000u)
        return (pg_fp16)(sign | 0x7c00u | ((absx & 0x007fffffu) ? 0x0200u : 0));

    int exp = (int)(absx >> 23) - 127;
    uint32_t mant = absx & 0x007fffffu;

    if (exp > 15)
        return (pg_fp16)(sign | 0x7c00u);

    if (exp >= -14) {
        uint32_t he = (uint32_t)(exp + 15) << 10;
        uint32_t hm = mant >> 13;
        uint32_t rem = mant & 0x1fffu;
        if (rem > 0x1000u || (rem == 0x1000u && (hm & 1u))) {
            hm++;
            if (hm == 0x0400u) {
                hm = 0;
                he += 0x0400u;
            }
        }
        return (pg_fp16)(sign | he | hm);
    }

    if (exp < -25)
        return (pg_fp16)sign;

    mant |= 0x00800000u;
    uint32_t rshift = (uint32_t)(13 + (-exp - 14));
    uint32_t hm = mant >> rshift;
    uint32_t rem_mask = (1u << rshift) - 1u;
    uint32_t rem = mant & rem_mask;
    uint32_t half = 1u << (rshift - 1);
    if (rem > half || (rem == half && (hm & 1u)))
        hm++;
    return (pg_fp16)(sign | hm);
}

static pg_fp16 *hbuf(size_t n)
{
    pg_fp16 *p = pg_aligned_alloc(PG_ALIGN, n * sizeof(pg_fp16));
    if (p) memset(p, 0, n * sizeof(pg_fp16));
    return p;
}

static double fp16_tol(const PgFp16GemmVariant *v, size_t K, double work)
{
    if (v->req_flags & PG_CPU_FP16) {
        double ku = (double)K * 0x1p-11;
        double gamma = ku < 0.9 ? ku / (1.0 - ku) : 9.0;
        return 1.5 * gamma * work + 2e-3;
    }
    return 16.0 * FLT_EPSILON * work + 1e-6;
}

static void fuzz_variant(const PgFp16GemmVariant *v, pg_fp16 *A, pg_fp16 *B,
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
            for (size_t k = 0; k < K; k++) A[i * LDA + k] = f32_to_fp16(checkasm_randf(2.0f));
        for (size_t k = 0; k < K; k++)
            for (size_t j = 0; j < N; j++) B[k * LDB + j] = f32_to_fp16(checkasm_randf(2.0f));

        for (size_t i = 0; i < M; i++)
            for (size_t j = 0; j < N; j++) {
                double s = 0.0, w = 0.0;
                for (size_t k = 0; k < K; k++) {
                    double p = (double)pg_fp16_to_f32(A[i * LDA + k]) *
                               (double)pg_fp16_to_f32(B[k * LDB + j]);
                    s += p;
                    w += fabs(p);
                }
                ref[i * MAXN + j] = s;
                work[i * MAXN + j] = w;
            }

        v->fn(M, N, K, A, LDA, B, LDB, C, LDC);

        for (size_t i = 0; i < M && ok; i++)
            for (size_t j = 0; j < N; j++) {
                double tol = fp16_tol(v, K, work[i * MAXN + j]);
                if (fabs((double)C[i * LDC + j] - ref[i * MAXN + j]) > tol) {
                    checkasm_fail("fp16gemm.%s M=%zu N=%zu K=%zu i=%zu j=%zu ref=%g got=%g tol=%g work=%g",
                                  v->name, M, N, K, i, j, ref[i * MAXN + j],
                                  C[i * LDC + j], tol, work[i * MAXN + j]);
                    ok = 0;
                    break;
                }
            }
    }
    checkasm_report("fp16gemm", v->name, ok);
}

static double bench_variant(pg_fp16gemm_fn fn, const pg_fp16 *A, const pg_fp16 *B,
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

void checkasm_check_gemm_fp16(void)
{
    size_t nv;
    const PgFp16GemmVariant *v = pg_fp16gemm_variants(&nv);
    unsigned flags = pg_get_cpu_flags();

    pg_fp16 *A = hbuf(MAXM * LDA), *B = hbuf(MAXK * LDB);
    float *C = pg_aligned_alloc(PG_ALIGN, MAXM * LDC * sizeof(float));
    double *ref = malloc(MAXM * MAXN * sizeof(double));
    double *work = malloc(MAXM * MAXN * sizeof(double));

    for (size_t i = 0; i < nv; i++)
        if ((flags & v[i].req_flags) == v[i].req_flags)
            fuzz_variant(&v[i], A, B, C, ref, work);

    if (checkasm_bench_enabled()) {
        static const size_t dims[] = { 512, 1024 };
        for (size_t s = 0; s < sizeof dims / sizeof dims[0]; s++) {
            size_t n = dims[s];
            pg_fp16 *bA = hbuf(n * n), *bB = hbuf(n * n);
            float *bC = pg_aligned_alloc(PG_ALIGN, n * n * sizeof(float));
            for (size_t k = 0; k < n * n; k++) {
                bA[k] = f32_to_fp16(checkasm_randf(1.0f));
                bB[k] = f32_to_fp16(checkasm_randf(1.0f));
            }
            char title[48];
            snprintf(title, sizeof title, "fp16gemm  %zux%zux%zu", n, n, n);
            checkasm_bench_begin(title, "GFLOP/s");
            for (size_t i = 0; i < nv; i++) {
                if ((flags & v[i].req_flags) != v[i].req_flags) continue;
                double g, per = bench_variant(v[i].fn, bA, bB, bC, n, n, n, &g);
                checkasm_bench_row(v[i].name, per, g);
            }
            checkasm_bench_end();
            pg_aligned_free(bA);
            pg_aligned_free(bB);
            pg_aligned_free(bC);
        }
    }

    free(ref);
    free(work);
    pg_aligned_free(A);
    pg_aligned_free(B);
    pg_aligned_free(C);
}
