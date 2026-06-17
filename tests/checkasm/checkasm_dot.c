/*
 * peregrine - checkasm test module for the f32 dot kernel.
 *
 * Walks the dot variant table and, for every implementation the host CPU can
 * run, fuzzes it against the C reference (random sizes, start alignments and
 * magnitudes) plus a fixed list of edge sizes. Under --bench it also times
 * each variant at several working-set sizes and prints a speedup table.
 */
#include "checkasm.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "util/cpu.h"
#include "util/mem.h"
#include "tensor/kernels/dot/dot.h"

#define MAXN  (1u << 16)   /* correctness working set */
#define PAD   16           /* slack for start-offset misalignment */

static float *alloc_buf(size_t n)
{
    float *p = pg_aligned_alloc(PG_ALIGN, n * sizeof(float));
    if (p) memset(p, 0, n * sizeof(float));
    return p;
}

static int supported(const PgDotVariant *v, unsigned flags)
{
    return (flags & v->req_flags) == v->req_flags;
}

/* Work = sum of |a_i*b_i|. A dot of signed inputs can cancel almost completely
 * (small result, huge intermediate magnitudes), so the meaningful error scale
 * is the standard floating-summation bound ~ eps * work, NOT eps * |result|. */
static double dot_work(const float *a, const float *b, size_t n)
{
    double s = 0.0;
    for (size_t i = 0; i < n; i++)
        s += fabs((double)a[i] * (double)b[i]);
    return s;
}

static int close_enough(float ref, float got, double work)
{
    /* Inherent reordering error of two summation orders is bounded by a small
     * multiple of eps*work; a real kernel bug is off by O(work) or O(result),
     * far above this floor. */
    float tol = 1e-4f + (float)(16.0 * (double)FLT_EPSILON * work);
    return fabsf(ref - got) <= tol;
}

static void fuzz_variant(const PgDotVariant *v, float *a, float *b)
{
    static const size_t edges[] = {
        0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33,
        63, 64, 65, 127, 128, 129, 255, 256, 257,
        512, 1376,
    };
    const float mags[] = { 0.5f, 1.0f, 10.0f, 100.0f };
    int ok = 1;

    /* deterministic edge sizes (offset 0) */
    for (size_t e = 0; e < sizeof edges / sizeof edges[0] && ok; e++) {
        size_t n = edges[e];
        for (size_t i = 0; i < n; i++) { a[i] = checkasm_randf(1.0f); b[i] = checkasm_randf(1.0f); }
        float ref = pg_dot_f32_c(a, b, n);
        float got = CK_CALL_DOT(v->fn, a, b, n);
        const char *reg;
        if (checkasm_clobbered(&reg)) {
            checkasm_fail("dot.%s edge n=%zu clobbered callee-saved %s", v->name, n, reg);
            ok = 0;
        } else if (!close_enough(ref, got, dot_work(a, b, n))) {
            checkasm_fail("dot.%s edge n=%zu off=0 ref=%.6g got=%.6g", v->name, n, ref, got);
            ok = 0;
        }
    }

    /* randomised: size, start offset (alignment), and magnitude */
    for (int t = 0; t < 400 && ok; t++) {
        size_t n   = checkasm_rand_range(0, MAXN);
        size_t off = checkasm_rand_range(0, PAD - 1);
        float  mag = mags[checkasm_rng() & 3];
        float *pa = a + off, *pb = b + off;
        for (size_t i = 0; i < n; i++) { pa[i] = checkasm_randf(mag); pb[i] = checkasm_randf(mag); }
        float ref = pg_dot_f32_c(pa, pb, n);
        float got = CK_CALL_DOT(v->fn, pa, pb, n);
        const char *reg;
        if (checkasm_clobbered(&reg)) {
            checkasm_fail("dot.%s rand n=%zu off=%zu clobbered callee-saved %s",
                          v->name, n, off, reg);
            ok = 0;
        } else if (!close_enough(ref, got, dot_work(pa, pb, n))) {
            checkasm_fail("dot.%s rand n=%zu off=%zu mag=%g ref=%.6g got=%.6g",
                          v->name, n, off, mag, ref, got);
            ok = 0;
        }
    }

    checkasm_report("dot_f32", v->name, ok);
}

static double bench_variant(pg_dot_f32_fn fn, const float *a, const float *b,
                            size_t n, double *gflops)
{
    volatile float sink = 0;
    int iters = 1;
    double el = 0.0;

    sink += fn(a, b, n);  /* warm up caches/branch predictors */
    do {
        double t0 = checkasm_now();
        for (int i = 0; i < iters; i++)
            sink += fn(a, b, n);
        el = checkasm_now() - t0;
        if (el < 0.05) iters *= 2;
    } while (el < 0.05 && iters < (1 << 28));
    (void)sink;

    double per = el / iters;
    *gflops = (2.0 * (double)n) / per / 1e9;  /* one mul + one add per element */
    return per;
}

void checkasm_check_dot(void)
{
    size_t nv;
    const PgDotVariant *v = pg_dot_variants(&nv);
    unsigned flags = pg_get_cpu_flags();

    float *a = alloc_buf(MAXN + PAD);
    float *b = alloc_buf(MAXN + PAD);

    for (size_t i = 0; i < nv; i++)
        if (supported(&v[i], flags))
            fuzz_variant(&v[i], a, b);

    if (checkasm_bench_enabled()) {
        static const size_t sizes[] = { 512, 1376, 4096, 65536, 1u << 20 };
        for (size_t s = 0; s < sizeof sizes / sizeof sizes[0]; s++) {
            size_t n = sizes[s];
            float *ba = alloc_buf(n);
            float *bb = alloc_buf(n);
            for (size_t k = 0; k < n; k++) { ba[k] = checkasm_randf(1.0f); bb[k] = checkasm_randf(1.0f); }

            char title[48];
            snprintf(title, sizeof title, "dot_f32  n=%zu", n);
            checkasm_bench_begin(title, "GFLOP/s");
            for (size_t i = 0; i < nv; i++) {
                if (!supported(&v[i], flags)) continue;
                double gflops;
                double per = bench_variant(v[i].fn, ba, bb, n, &gflops);
                checkasm_bench_row(v[i].name, per, gflops);
            }
            checkasm_bench_end();

            pg_aligned_free(ba);
            pg_aligned_free(bb);
        }
    }

    pg_aligned_free(a);
    pg_aligned_free(b);
}
