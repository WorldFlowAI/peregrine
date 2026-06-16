/*
 * peregrine - checkasm module for f32 RoPE.
 *
 * Validates: (1) the cos/sin cache against a double-precision evaluation;
 * (2) the rotation against a double-precision rotation using that cache, for
 * both pair conventions; (3) that in-place (out aliases in) matches
 * out-of-place exactly.
 *
 * rope's apply has 8 args, more than the clobber-check trampoline forwards, so
 * (being C only for now) we call it directly. When the asm variant lands the
 * trampoline will be extended (or mode split into per-convention entry points).
 */
#include "checkasm.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "util/cpu.h"
#include "util/mem.h"
#include "tensor/kernels/rope/rope.h"

#define MAX_TOK   32
#define MAX_HEAD  8
#define MAX_DIM   128
#define MAX_X     (MAX_TOK * MAX_HEAD * MAX_DIM)
#define MAX_C     (MAX_TOK * (MAX_DIM / 2))

static float *fbuf(size_t n)
{
    float *p = pg_aligned_alloc(PG_ALIGN, n * sizeof(float));
    if (p) memset(p, 0, n * sizeof(float));
    return p;
}

static int close_rel(float got, double ref)
{
    double tol = 2e-4 * (fabs(ref) + 1.0);
    return fabs((double)got - ref) <= tol;
}

static void fuzz_variant(const PgRopeVariant *v, float *x, float *out, float *tmp,
                         float *cosb, float *sinb, int32_t *pos)
{
    static const size_t dims[] = { 2, 4, 8, 16, 32, 64, 128 };
    const float theta = 10000.0f;
    int ok = 1;

    for (int t = 0; t < 300 && ok; t++) {
        size_t head_dim = dims[checkasm_rng() % (sizeof dims / sizeof dims[0])];
        size_t half     = head_dim / 2;
        size_t n_tok    = checkasm_rand_range(1, MAX_TOK);
        size_t n_head   = checkasm_rand_range(1, MAX_HEAD);
        PgRopeMode mode = (checkasm_rng() & 1) ? PG_ROPE_NEOX : PG_ROPE_INTERLEAVED;

        for (size_t i = 0; i < n_tok; i++) pos[i] = (int32_t)checkasm_rand_range(0, 100000);
        for (size_t i = 0; i < n_tok * n_head * head_dim; i++) x[i] = checkasm_randf(4.0f);

        pg_rope_cache_f32_c(cosb, sinb, pos, n_tok, head_dim, theta);

        /* (1) cache vs double */
        for (size_t tk = 0; tk < n_tok && ok; tk++)
            for (size_t i = 0; i < half; i++) {
                double freq = pow((double)theta, -2.0 * (double)i / (double)head_dim);
                double ang  = (double)pos[tk] * freq;
                if (!close_rel(cosb[tk * half + i], cos(ang)) ||
                    !close_rel(sinb[tk * half + i], sin(ang))) {
                    checkasm_fail("rope.%s cache mismatch dim=%zu tok=%zu i=%zu", v->name, head_dim, tk, i);
                    ok = 0;
                }
            }

        /* (2) out-of-place apply vs double rotation using the (float) cache */
        v->fn(out, x, cosb, sinb, n_tok, n_head, head_dim, mode);
        for (size_t tk = 0; tk < n_tok && ok; tk++)
            for (size_t h = 0; h < n_head && ok; h++)
                for (size_t i = 0; i < half; i++) {
                    double c = cosb[tk * half + i], s = sinb[tk * half + i];
                    size_t i0 = (mode == PG_ROPE_INTERLEAVED) ? 2 * i : i;
                    size_t i1 = (mode == PG_ROPE_INTERLEAVED) ? 2 * i + 1 : i + half;
                    const float *xi = x + (tk * n_head + h) * head_dim;
                    double e0 = (double)xi[i0] * c - (double)xi[i1] * s;
                    double e1 = (double)xi[i0] * s + (double)xi[i1] * c;
                    const float *xo = out + (tk * n_head + h) * head_dim;
                    if (!close_rel(xo[i0], e0) || !close_rel(xo[i1], e1)) {
                        checkasm_fail("rope.%s apply mismatch dim=%zu mode=%d tok=%zu", v->name, head_dim, mode, tk);
                        ok = 0;
                    }
                }

        /* (3) in-place (out aliases in) must match out-of-place exactly */
        size_t total = n_tok * n_head * head_dim;
        memcpy(tmp, x, total * sizeof(float));
        v->fn(tmp, tmp, cosb, sinb, n_tok, n_head, head_dim, mode);
        if (ok && memcmp(tmp, out, total * sizeof(float)) != 0) {
            checkasm_fail("rope.%s in-place != out-of-place dim=%zu mode=%d", v->name, head_dim, mode);
            ok = 0;
        }
    }
    checkasm_report("rope_f32", v->name, ok);
}

void checkasm_check_rope(void)
{
    size_t nv;
    const PgRopeVariant *v = pg_rope_variants(&nv);
    unsigned flags = pg_get_cpu_flags();

    float *x = fbuf(MAX_X), *out = fbuf(MAX_X), *tmp = fbuf(MAX_X);
    float *cosb = fbuf(MAX_C), *sinb = fbuf(MAX_C);
    int32_t *pos = pg_aligned_alloc(PG_ALIGN, MAX_TOK * sizeof(int32_t));

    for (size_t i = 0; i < nv; i++)
        if ((flags & v[i].req_flags) == v[i].req_flags)
            fuzz_variant(&v[i], x, out, tmp, cosb, sinb, pos);

    pg_aligned_free(x); pg_aligned_free(out); pg_aligned_free(tmp);
    pg_aligned_free(cosb); pg_aligned_free(sinb); pg_aligned_free(pos);
}
