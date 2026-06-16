/*
 * peregrine - checkasm module for f32 softmax.
 * Validates against a double-precision softmax of the same row, and checks the
 * outputs sum to 1.
 */
#include "checkasm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util/cpu.h"
#include "util/mem.h"
#include "tensor/kernels/softmax/softmax.h"

#define MAXN  (1u << 14)
#define PAD   16

static float *alloc_buf(size_t n)
{
    float *p = pg_aligned_alloc(PG_ALIGN, n * sizeof(float));
    if (p) memset(p, 0, n * sizeof(float));
    return p;
}

static void fuzz_variant(const PgSoftmaxVariant *v, float *in, float *out, double *ref)
{
    static const size_t edges[] = { 1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 63, 64, 257 };
    const float mags[] = { 0.5f, 1.0f, 10.0f, 50.0f };
    int ok = 1;

    for (int t = 0; t < 300 && ok; t++) {
        int edge   = t < (int)(sizeof edges / sizeof edges[0]);
        size_t n   = edge ? edges[t] : checkasm_rand_range(1, MAXN);
        size_t off = edge ? 0 : checkasm_rand_range(0, PAD - 1);
        float  mag = mags[checkasm_rng() & 3];
        for (size_t i = 0; i < n; i++) in[off + i] = checkasm_randf(mag);

        /* double-precision oracle */
        double dmax = in[off];
        for (size_t i = 1; i < n; i++) if (in[off + i] > dmax) dmax = in[off + i];
        double dsum = 0.0;
        for (size_t i = 0; i < n; i++) { ref[i] = exp((double)in[off + i] - dmax); dsum += ref[i]; }
        for (size_t i = 0; i < n; i++) ref[i] /= dsum;

        CK_CALL_UNARY(v->fn, in + off, out + off, n);

        const char *reg;
        if (checkasm_clobbered(&reg)) {
            checkasm_fail("softmax.%s n=%zu off=%zu clobbered callee-saved %s", v->name, n, off, reg);
            ok = 0;
            break;
        }
        double gsum = 0.0;
        for (size_t i = 0; i < n; i++) {
            double tol = 1e-4 * ref[i] + 1e-7;
            if (fabs((double)out[off + i] - ref[i]) > tol) {
                checkasm_fail("softmax.%s n=%zu i=%zu ref=%g got=%g", v->name, n, i, ref[i], out[off + i]);
                ok = 0;
                break;
            }
            gsum += out[off + i];
        }
        if (ok && fabs(gsum - 1.0) > 1e-4) {
            checkasm_fail("softmax.%s n=%zu outputs sum to %g, not 1", v->name, n, gsum);
            ok = 0;
        }
    }
    checkasm_report("softmax_f32", v->name, ok);
}

static double bench_variant(pg_softmax_f32_fn fn, const float *in, float *out, size_t n, double *melems)
{
    volatile float sink = 0;
    int iters = 1;
    double el = 0.0;
    fn(in, out, n);
    do {
        double t0 = checkasm_now();
        for (int i = 0; i < iters; i++) fn(in, out, n);
        el = checkasm_now() - t0;
        if (el < 0.05) iters *= 2;
    } while (el < 0.05 && iters < (1 << 28));
    sink += out[0];
    (void)sink;
    double per = el / iters;
    *melems = (double)n / per / 1e6;
    return per;
}

void checkasm_check_softmax(void)
{
    size_t nv;
    const PgSoftmaxVariant *v = pg_softmax_variants(&nv);
    unsigned flags = pg_get_cpu_flags();
    float  *in  = alloc_buf(MAXN + PAD), *out = alloc_buf(MAXN + PAD);
    double *ref = malloc(MAXN * sizeof(double));

    for (size_t i = 0; i < nv; i++)
        if ((flags & v[i].req_flags) == v[i].req_flags)
            fuzz_variant(&v[i], in, out, ref);

    if (checkasm_bench_enabled()) {
        static const size_t sizes[] = { 1024, 16384, 1u << 18 };
        for (size_t s = 0; s < sizeof sizes / sizeof sizes[0]; s++) {
            size_t n = sizes[s];
            float *bi = alloc_buf(n), *bo = alloc_buf(n);
            for (size_t k = 0; k < n; k++) bi[k] = checkasm_randf(4.0f);
            char title[48];
            snprintf(title, sizeof title, "softmax_f32  n=%zu", n);
            checkasm_bench_begin(title, "Melem/s");
            for (size_t i = 0; i < nv; i++) {
                if ((flags & v[i].req_flags) != v[i].req_flags) continue;
                double m, per = bench_variant(v[i].fn, bi, bo, n, &m);
                checkasm_bench_row(v[i].name, per, m);
            }
            checkasm_bench_end();
            pg_aligned_free(bi); pg_aligned_free(bo);
        }
    }

    free(ref);
    pg_aligned_free(in);
    pg_aligned_free(out);
}
