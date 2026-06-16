/*
 * peregrine - checkasm module for f32 SiLU.
 * Validates each variant against a double-precision oracle (so the C reference
 * is proven now, and future NEON/AVX2 variants are checked the same way).
 */
#include "checkasm.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "util/cpu.h"
#include "util/mem.h"
#include "tensor/kernels/silu/silu.h"

#define MAXN  (1u << 14)
#define PAD   16

static float *alloc_buf(size_t n)
{
    float *p = pg_aligned_alloc(PG_ALIGN, n * sizeof(float));
    if (p) memset(p, 0, n * sizeof(float));
    return p;
}

static int close_rel(float got, double ref)
{
    double tol = 1e-4 * fabs(ref) + 1e-6;
    return fabs((double)got - ref) <= tol;
}

static void fuzz_variant(const PgSiluVariant *v, float *in, float *out)
{
    static const size_t edges[] = { 0, 1, 3, 7, 8, 15, 16, 17, 31, 32, 63, 64, 257 };
    const float mags[] = { 0.5f, 1.0f, 5.0f, 10.0f };
    int ok = 1;

    for (int t = 0; t < 300 && ok; t++) {
        int edge   = t < (int)(sizeof edges / sizeof edges[0]);
        size_t n   = edge ? edges[t] : checkasm_rand_range(0, MAXN);
        size_t off = edge ? 0 : checkasm_rand_range(0, PAD - 1);
        float  mag = mags[checkasm_rng() & 3];
        for (size_t i = 0; i < n; i++) in[off + i] = checkasm_randf(mag);

        CK_CALL_UNARY(v->fn, in + off, out + off, n);

        const char *reg;
        if (checkasm_clobbered(&reg)) {
            checkasm_fail("silu.%s n=%zu off=%zu clobbered callee-saved %s", v->name, n, off, reg);
            ok = 0;
            break;
        }
        for (size_t i = 0; i < n; i++) {
            double x = in[off + i];
            double ref = x / (1.0 + exp(-x));
            if (!close_rel(out[off + i], ref)) {
                checkasm_fail("silu.%s n=%zu i=%zu x=%g ref=%g got=%g", v->name, n, i, x, ref, out[off + i]);
                ok = 0;
                break;
            }
        }
    }
    checkasm_report("silu_f32", v->name, ok);
}

static double bench_variant(pg_silu_f32_fn fn, const float *in, float *out, size_t n, double *melems)
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

void checkasm_check_silu(void)
{
    size_t nv;
    const PgSiluVariant *v = pg_silu_variants(&nv);
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
            for (size_t k = 0; k < n; k++) bi[k] = checkasm_randf(8.0f);
            char title[48];
            snprintf(title, sizeof title, "silu_f32  n=%zu", n);
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
