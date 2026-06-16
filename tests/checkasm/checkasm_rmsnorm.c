/*
 * peregrine - checkasm module for the f32 RMSNorm kernel.
 *
 * Writes to a separate out[] so ref and variant each render into their own
 * buffer and we compare. Fuzzes size, start alignment, magnitude and eps.
 */
#include "checkasm.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "util/cpu.h"
#include "util/mem.h"
#include "tensor/kernels/rmsnorm/rmsnorm.h"

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
        float tol = 3e-3f * (fabsf(ref[i]) + 1e-3f);
        if (fabsf(ref[i] - got[i]) > tol)
            return 0;
    }
    return 1;
}

static void fuzz_variant(const PgRmsnormVariant *v, float *x, float *w,
                         float *outr, float *outt)
{
    static const size_t edges[] = {
        1, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 257,
    };
    const float mags[] = { 0.5f, 1.0f, 10.0f };
    const float epss[] = { 1e-6f, 1e-5f, 1e-3f };
    int ok = 1;

    for (int t = 0; t < 420 && ok; t++) {
        int edge   = t < (int)(sizeof edges / sizeof edges[0]);
        size_t n   = edge ? edges[t] : checkasm_rand_range(1, MAXN);
        size_t off = edge ? 0 : checkasm_rand_range(0, PAD - 1);
        float  mag = mags[checkasm_rng() % 3];
        float  eps = epss[checkasm_rng() % 3];

        for (size_t i = 0; i < n; i++) {
            x[off + i] = checkasm_randf(mag);
            w[off + i] = checkasm_randf(1.0f);
        }

        pg_rmsnorm_f32_c(outr + off, x + off, w + off, n, eps);
        CK_CALL_RMSNORM(v->fn, outt + off, x + off, w + off, n, eps);

        const char *reg;
        if (checkasm_clobbered(&reg)) {
            checkasm_fail("rmsnorm.%s n=%zu off=%zu clobbered callee-saved %s",
                          v->name, n, off, reg);
            ok = 0;
        } else if (!arrays_close(outr + off, outt + off, n)) {
            checkasm_fail("rmsnorm.%s n=%zu off=%zu mag=%g eps=%g mismatch",
                          v->name, n, off, mag, eps);
            ok = 0;
        }
    }
    checkasm_report("rmsnorm_f32", v->name, ok);
}

static double bench_variant(pg_rmsnorm_f32_fn fn, float *out, const float *x,
                            const float *w, size_t n, double *gbps)
{
    volatile int sink = 0;
    int iters = 1;
    double el = 0.0;

    fn(out, x, w, n, 1e-5f);
    do {
        double t0 = checkasm_now();
        for (int i = 0; i < iters; i++)
            fn(out, x, w, n, 1e-5f);
        el = checkasm_now() - t0;
        if (el < 0.05) iters *= 2;
    } while (el < 0.05 && iters < (1 << 28));
    sink += (int)out[0];
    (void)sink;

    double per = el / iters;
    /* memory-bound: x read twice (2 passes) + w + out = 4 streams */
    *gbps = (4.0 * (double)n * sizeof(float)) / per / 1e9;
    return per;
}

void checkasm_check_rmsnorm(void)
{
    size_t nv;
    const PgRmsnormVariant *v = pg_rmsnorm_variants(&nv);
    unsigned flags = pg_get_cpu_flags();

    float *x    = alloc_buf(MAXN + PAD);
    float *w    = alloc_buf(MAXN + PAD);
    float *outr = alloc_buf(MAXN + PAD);
    float *outt = alloc_buf(MAXN + PAD);

    for (size_t i = 0; i < nv; i++)
        if ((flags & v[i].req_flags) == v[i].req_flags)
            fuzz_variant(&v[i], x, w, outr, outt);

    if (checkasm_bench_enabled()) {
        static const size_t sizes[] = { 4096, 65536, 1u << 20 };
        for (size_t s = 0; s < sizeof sizes / sizeof sizes[0]; s++) {
            size_t n = sizes[s];
            float *bx = alloc_buf(n), *bw = alloc_buf(n), *bo = alloc_buf(n);
            for (size_t k = 0; k < n; k++) { bx[k] = checkasm_randf(1.0f); bw[k] = checkasm_randf(1.0f); }
            char title[48];
            snprintf(title, sizeof title, "rmsnorm_f32  n=%zu", n);
            checkasm_bench_begin(title, "GB/s");
            for (size_t i = 0; i < nv; i++) {
                if ((flags & v[i].req_flags) != v[i].req_flags) continue;
                double g, per = bench_variant(v[i].fn, bo, bx, bw, n, &g);
                checkasm_bench_row(v[i].name, per, g);
            }
            checkasm_bench_end();
            pg_aligned_free(bx); pg_aligned_free(bw); pg_aligned_free(bo);
        }
    }

    pg_aligned_free(x);    pg_aligned_free(w);
    pg_aligned_free(outr); pg_aligned_free(outt);
}
