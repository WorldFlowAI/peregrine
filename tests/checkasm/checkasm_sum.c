/*
 * peregrine - checkasm module for the f32 sum reduction.
 */
#include "checkasm.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "util/cpu.h"
#include "util/mem.h"
#include "tensor/kernels/sum/sum.h"

#define MAXN  (1u << 16)
#define PAD   16

static float *alloc_buf(size_t n)
{
    float *p = pg_aligned_alloc(PG_ALIGN, n * sizeof(float));
    if (p) memset(p, 0, n * sizeof(float));
    return p;
}

/* Sum of signed values cancels, so tolerance scales with the work, not |ref|. */
static double sum_work(const float *x, size_t n)
{
    double s = 0.0;
    for (size_t i = 0; i < n; i++)
        s += fabs((double)x[i]);
    return s;
}

static int close_enough(float ref, float got, double work)
{
    float tol = 1e-4f + (float)(16.0 * (double)FLT_EPSILON * work);
    return fabsf(ref - got) <= tol;
}

static void fuzz_variant(const PgSumVariant *v, float *x)
{
    static const size_t edges[] = {
        0, 1, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 257,
    };
    const float mags[] = { 0.5f, 1.0f, 10.0f, 100.0f };
    int ok = 1;

    for (int t = 0; t < 420 && ok; t++) {
        int edge   = t < (int)(sizeof edges / sizeof edges[0]);
        size_t n   = edge ? edges[t] : checkasm_rand_range(0, MAXN);
        size_t off = edge ? 0 : checkasm_rand_range(0, PAD - 1);
        float  mag = mags[checkasm_rng() & 3];
        float *p = x + off;

        for (size_t i = 0; i < n; i++) p[i] = checkasm_randf(mag);
        float ref = pg_sum_f32_c(p, n);
        float got = CK_CALL_REDUCE(v->fn, p, n);

        const char *reg;
        if (checkasm_clobbered(&reg)) {
            checkasm_fail("sum.%s n=%zu off=%zu clobbered callee-saved %s", v->name, n, off, reg);
            ok = 0;
        } else if (!close_enough(ref, got, sum_work(p, n))) {
            checkasm_fail("sum.%s n=%zu off=%zu mag=%g ref=%.6g got=%.6g", v->name, n, off, mag, ref, got);
            ok = 0;
        }
    }
    checkasm_report("sum_f32", v->name, ok);
}

static double bench_variant(pg_sum_f32_fn fn, const float *x, size_t n, double *gflops)
{
    volatile float sink = 0;
    int iters = 1;
    double el = 0.0;
    sink += fn(x, n);
    do {
        double t0 = checkasm_now();
        for (int i = 0; i < iters; i++) sink += fn(x, n);
        el = checkasm_now() - t0;
        if (el < 0.05) iters *= 2;
    } while (el < 0.05 && iters < (1 << 28));
    (void)sink;
    double per = el / iters;
    *gflops = (double)n / per / 1e9;
    return per;
}

void checkasm_check_sum(void)
{
    size_t nv;
    const PgSumVariant *v = pg_sum_variants(&nv);
    unsigned flags = pg_get_cpu_flags();
    float *x = alloc_buf(MAXN + PAD);

    for (size_t i = 0; i < nv; i++)
        if ((flags & v[i].req_flags) == v[i].req_flags)
            fuzz_variant(&v[i], x);

    if (checkasm_bench_enabled()) {
        static const size_t sizes[] = { 4096, 65536, 1u << 20 };
        for (size_t s = 0; s < sizeof sizes / sizeof sizes[0]; s++) {
            size_t n = sizes[s];
            float *bx = alloc_buf(n);
            for (size_t k = 0; k < n; k++) bx[k] = checkasm_randf(1.0f);
            char title[48];
            snprintf(title, sizeof title, "sum_f32  n=%zu", n);
            checkasm_bench_begin(title, "GFLOP/s");
            for (size_t i = 0; i < nv; i++) {
                if ((flags & v[i].req_flags) != v[i].req_flags) continue;
                double g, per = bench_variant(v[i].fn, bx, n, &g);
                checkasm_bench_row(v[i].name, per, g);
            }
            checkasm_bench_end();
            pg_aligned_free(bx);
        }
    }
    pg_aligned_free(x);
}
