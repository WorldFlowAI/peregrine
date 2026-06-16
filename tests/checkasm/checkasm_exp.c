/*
 * peregrine - checkasm module for f32 exp (vectorised approximation).
 * Validates each variant against double-precision exp() and reports the actual
 * worst-case relative error.
 */
#include "checkasm.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "util/cpu.h"
#include "util/mem.h"
#include "tensor/kernels/exp/exp.h"

#define MAXN  (1u << 14)
#define PAD   16
#define TOL   1e-6   /* asm poly is ~1e-7; expf oracle adds <1 ulp */

static float *alloc_buf(size_t n)
{
    float *p = pg_aligned_alloc(PG_ALIGN, n * sizeof(float));
    if (p) memset(p, 0, n * sizeof(float));
    return p;
}

static void fuzz_variant(const PgExpVariant *v, float *in, float *out)
{
    static const size_t edges[] = { 0, 1, 3, 7, 8, 15, 16, 17, 31, 32, 63, 64, 257 };
    int ok = 1;
    double maxrel = 0.0;

    for (int t = 0; t < 300 && ok; t++) {
        int edge   = t < (int)(sizeof edges / sizeof edges[0]);
        size_t n   = edge ? edges[t] : checkasm_rand_range(0, MAXN);
        size_t off = edge ? 0 : checkasm_rand_range(0, PAD - 1);
        /* accurate range; beyond +/-88 expf overflows/underflows */
        for (size_t i = 0; i < n; i++) in[off + i] = checkasm_randf(40.0f);

        CK_CALL_UNARY(v->fn, in + off, out + off, n);

        const char *reg;
        if (checkasm_clobbered(&reg)) {
            checkasm_fail("exp.%s n=%zu clobbered callee-saved %s", v->name, n, reg);
            ok = 0;
            break;
        }
        for (size_t i = 0; i < n; i++) {
            double ref = exp((double)in[off + i]);
            double rel = fabs((double)out[off + i] - ref) / ref;
            if (rel > maxrel) maxrel = rel;
            if (rel > TOL) {
                checkasm_fail("exp.%s x=%g ref=%g got=%g rel=%.2e",
                              v->name, in[off + i], ref, out[off + i], rel);
                ok = 0;
                break;
            }
        }
    }
    if (ok)
        printf("      exp.%s max relative error: %.2e\n", v->name, maxrel);
    checkasm_report("exp_f32", v->name, ok);
}

static double bench_variant(pg_exp_f32_fn fn, const float *in, float *out, size_t n, double *melems)
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

void checkasm_check_exp(void)
{
    size_t nv;
    const PgExpVariant *v = pg_exp_variants(&nv);
    unsigned flags = pg_get_cpu_flags();
    float *in = alloc_buf(MAXN + PAD), *out = alloc_buf(MAXN + PAD);

    for (size_t i = 0; i < nv; i++)
        if ((flags & v[i].req_flags) == v[i].req_flags)
            fuzz_variant(&v[i], in, out);

    if (checkasm_bench_enabled()) {
        static const size_t sizes[] = { 4096, 65536, 1u << 20 };
        for (size_t s = 0; s < sizeof sizes / sizeof sizes[0]; s++) {
            size_t n = sizes[s];
            float *bi = alloc_buf(n), *bo = alloc_buf(n);
            for (size_t k = 0; k < n; k++) bi[k] = checkasm_randf(40.0f);
            char title[48];
            snprintf(title, sizeof title, "exp_f32  n=%zu", n);
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

    pg_aligned_free(in);
    pg_aligned_free(out);
}
