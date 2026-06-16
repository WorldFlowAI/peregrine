/*
 * peregrine - checkasm module for f32 GELU (tanh approximation).
 * Validated against a double-precision evaluation of the same tanh formula.
 */
#include "checkasm.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "util/cpu.h"
#include "util/mem.h"
#include "tensor/kernels/gelu/gelu.h"

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

static double ref_gelu(double x)
{
    const double k = 0.7978845608028654;  /* sqrt(2/pi) */
    return 0.5 * x * (1.0 + tanh(k * (x + 0.044715 * x * x * x)));
}

static void fuzz_variant(const PgGeluVariant *v, float *in, float *out)
{
    static const size_t edges[] = { 0, 1, 3, 7, 8, 15, 16, 17, 31, 32, 63, 64, 257 };
    const float mags[] = { 0.5f, 1.0f, 3.0f, 8.0f };
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
            checkasm_fail("gelu.%s n=%zu off=%zu clobbered callee-saved %s", v->name, n, off, reg);
            ok = 0;
            break;
        }
        for (size_t i = 0; i < n; i++) {
            double ref = ref_gelu(in[off + i]);
            if (!close_rel(out[off + i], ref)) {
                checkasm_fail("gelu.%s n=%zu i=%zu x=%g ref=%g got=%g", v->name, n, i, in[off + i], ref, out[off + i]);
                ok = 0;
                break;
            }
        }
    }
    checkasm_report("gelu_f32", v->name, ok);
}

void checkasm_check_gelu(void)
{
    size_t nv;
    const PgGeluVariant *v = pg_gelu_variants(&nv);
    unsigned flags = pg_get_cpu_flags();
    float *in = alloc_buf(MAXN + PAD), *out = alloc_buf(MAXN + PAD);

    for (size_t i = 0; i < nv; i++)
        if ((flags & v[i].req_flags) == v[i].req_flags)
            fuzz_variant(&v[i], in, out);

    pg_aligned_free(in);
    pg_aligned_free(out);
}
