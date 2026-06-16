/*
 * peregrine - checkasm module for the f32 elementwise add.
 * Each output is a single rounded sum, identical in C and SIMD: exact.
 */
#include "checkasm.h"

#include <stdio.h>
#include <string.h>

#include "util/cpu.h"
#include "util/mem.h"
#include "tensor/kernels/add/add.h"

#define MAXN  (1u << 16)
#define PAD   16

static float *alloc_buf(size_t n)
{
    float *p = pg_aligned_alloc(PG_ALIGN, n * sizeof(float));
    if (p) memset(p, 0, n * sizeof(float));
    return p;
}

static void fuzz_variant(const PgAddVariant *v, float *a, float *b, float *r, float *t)
{
    static const size_t edges[] = {
        0, 1, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 257,
    };
    const float mags[] = { 0.5f, 1.0f, 10.0f, 100.0f };
    int ok = 1;

    for (int it = 0; it < 420 && ok; it++) {
        int edge   = it < (int)(sizeof edges / sizeof edges[0]);
        size_t n   = edge ? edges[it] : checkasm_rand_range(0, MAXN);
        size_t off = edge ? 0 : checkasm_rand_range(0, PAD - 1);
        float  mag = mags[checkasm_rng() & 3];
        for (size_t i = 0; i < n; i++) { a[off + i] = checkasm_randf(mag); b[off + i] = checkasm_randf(mag); }

        pg_add_f32_c(a + off, b + off, r + off, n);
        CK_CALL_BINOP(v->fn, a + off, b + off, t + off, n);

        const char *reg;
        if (checkasm_clobbered(&reg)) {
            checkasm_fail("add.%s n=%zu off=%zu clobbered callee-saved %s", v->name, n, off, reg);
            ok = 0;
        } else if (memcmp(r + off, t + off, n * sizeof(float)) != 0) {
            checkasm_fail("add.%s n=%zu off=%zu mag=%g mismatch", v->name, n, off, mag);
            ok = 0;
        }
    }
    checkasm_report("add_f32", v->name, ok);
}

static double bench_variant(pg_add_f32_fn fn, const float *a, const float *b, float *o, size_t n, double *gbps)
{
    int iters = 1;
    double el = 0.0;
    fn(a, b, o, n);
    do {
        double t0 = checkasm_now();
        for (int i = 0; i < iters; i++) fn(a, b, o, n);
        el = checkasm_now() - t0;
        if (el < 0.05) iters *= 2;
    } while (el < 0.05 && iters < (1 << 28));
    double per = el / iters;
    *gbps = 3.0 * (double)n * sizeof(float) / per / 1e9;  /* read a,b + write o */
    return per;
}

void checkasm_check_add(void)
{
    size_t nv;
    const PgAddVariant *v = pg_add_variants(&nv);
    unsigned flags = pg_get_cpu_flags();
    float *a = alloc_buf(MAXN + PAD), *b = alloc_buf(MAXN + PAD);
    float *r = alloc_buf(MAXN + PAD), *t = alloc_buf(MAXN + PAD);

    for (size_t i = 0; i < nv; i++)
        if ((flags & v[i].req_flags) == v[i].req_flags)
            fuzz_variant(&v[i], a, b, r, t);

    if (checkasm_bench_enabled()) {
        static const size_t sizes[] = { 4096, 65536, 1u << 20 };
        for (size_t s = 0; s < sizeof sizes / sizeof sizes[0]; s++) {
            size_t n = sizes[s];
            float *ba = alloc_buf(n), *bb = alloc_buf(n), *bo = alloc_buf(n);
            for (size_t k = 0; k < n; k++) { ba[k] = checkasm_randf(1.0f); bb[k] = checkasm_randf(1.0f); }
            char title[48];
            snprintf(title, sizeof title, "add_f32  n=%zu", n);
            checkasm_bench_begin(title, "GB/s");
            for (size_t i = 0; i < nv; i++) {
                if ((flags & v[i].req_flags) != v[i].req_flags) continue;
                double g, per = bench_variant(v[i].fn, ba, bb, bo, n, &g);
                checkasm_bench_row(v[i].name, per, g);
            }
            checkasm_bench_end();
            pg_aligned_free(ba); pg_aligned_free(bb); pg_aligned_free(bo);
        }
    }
    pg_aligned_free(a); pg_aligned_free(b); pg_aligned_free(r); pg_aligned_free(t);
}
