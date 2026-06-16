/*
 * peregrine - checkasm module for the f32 axpy kernel (y += alpha*x).
 *
 * In-place output, so each call runs on a fresh copy of y and we compare the
 * resulting arrays. Same fuzz dimensions as dot: size, start alignment,
 * magnitude (here also a random alpha), plus a fixed edge-size list.
 */
#include "checkasm.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "util/cpu.h"
#include "util/mem.h"
#include "tensor/kernels/axpy/axpy.h"

#define MAXN  (1u << 16)
#define PAD   16

static float *alloc_buf(size_t n)
{
    float *p = pg_aligned_alloc(PG_ALIGN, n * sizeof(float));
    if (p) memset(p, 0, n * sizeof(float));
    return p;
}

static int arrays_close(const float *ref, const float *got, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        float tol = 2e-3f * (fabsf(ref[i]) + 1.0f);
        if (fabsf(ref[i] - got[i]) > tol)
            return 0;
    }
    return 1;
}

static void fuzz_variant(const PgAxpyVariant *v, float *x, float *y0,
                         float *yr, float *yt)
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
        float  alpha = checkasm_randf(mag);

        for (size_t i = 0; i < n; i++) {
            x[off + i]  = checkasm_randf(mag);
            y0[off + i] = checkasm_randf(mag);
        }
        memcpy(yr + off, y0 + off, n * sizeof(float));
        memcpy(yt + off, y0 + off, n * sizeof(float));

        pg_axpy_f32_c(alpha, x + off, yr + off, n);
        CK_CALL_AXPY(v->fn, alpha, x + off, yt + off, n);

        const char *reg;
        if (checkasm_clobbered(&reg)) {
            checkasm_fail("axpy.%s n=%zu off=%zu clobbered callee-saved %s",
                          v->name, n, off, reg);
            ok = 0;
        } else if (!arrays_close(yr + off, yt + off, n)) {
            checkasm_fail("axpy.%s n=%zu off=%zu alpha=%g mismatch",
                          v->name, n, off, alpha);
            ok = 0;
        }
    }
    checkasm_report("axpy_f32", v->name, ok);
}

static double bench_variant(pg_axpy_f32_fn fn, const float *x, float *y,
                            size_t n, double *gflops)
{
    volatile int sink = 0;
    int iters = 1;
    double el = 0.0;

    fn(0.0f, x, y, n);
    do {
        double t0 = checkasm_now();
        for (int i = 0; i < iters; i++)
            fn(1e-6f, x, y, n);
        el = checkasm_now() - t0;
        if (el < 0.05) iters *= 2;
    } while (el < 0.05 && iters < (1 << 28));
    sink += (int)y[0];
    (void)sink;

    double per = el / iters;
    *gflops = (2.0 * (double)n) / per / 1e9;  /* one mul + one add per element */
    return per;
}

void checkasm_check_axpy(void)
{
    size_t nv;
    const PgAxpyVariant *v = pg_axpy_variants(&nv);
    unsigned flags = pg_get_cpu_flags();

    float *x  = alloc_buf(MAXN + PAD);
    float *y0 = alloc_buf(MAXN + PAD);
    float *yr = alloc_buf(MAXN + PAD);
    float *yt = alloc_buf(MAXN + PAD);

    for (size_t i = 0; i < nv; i++)
        if ((flags & v[i].req_flags) == v[i].req_flags)
            fuzz_variant(&v[i], x, y0, yr, yt);

    if (checkasm_bench_enabled()) {
        static const size_t sizes[] = { 4096, 65536, 1u << 20 };
        for (size_t s = 0; s < sizeof sizes / sizeof sizes[0]; s++) {
            size_t n = sizes[s];
            float *bx = alloc_buf(n), *by = alloc_buf(n);
            for (size_t k = 0; k < n; k++) { bx[k] = checkasm_randf(1.0f); by[k] = checkasm_randf(1.0f); }
            char title[48];
            snprintf(title, sizeof title, "axpy_f32  n=%zu", n);
            checkasm_bench_begin(title, "GFLOP/s");
            for (size_t i = 0; i < nv; i++) {
                if ((flags & v[i].req_flags) != v[i].req_flags) continue;
                double g, per = bench_variant(v[i].fn, bx, by, n, &g);
                checkasm_bench_row(v[i].name, per, g);
            }
            checkasm_bench_end();
            pg_aligned_free(bx); pg_aligned_free(by);
        }
    }

    pg_aligned_free(x);  pg_aligned_free(y0);
    pg_aligned_free(yr); pg_aligned_free(yt);
}
