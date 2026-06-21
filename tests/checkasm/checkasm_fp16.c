/*
 * peregrine - checkasm module for fp16/f32 conversion kernels.
 */
#include "checkasm.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tensor/kernels/fp16/fp16.h"
#include "util/cpu.h"
#include "util/mem.h"

#define MAXN 4096

static uint16_t finite_half_bits(void)
{
    uint16_t h;

    do {
        h = (uint16_t)checkasm_rng();
    } while (((h >> 10) & 31u) == 31u);
    return h;
}

static void fill_inputs(pg_fp16 *src_h, float *src_f, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        src_h[i] = finite_half_bits();
        src_f[i] = pg_fp16_to_f32(finite_half_bits());
    }
}

static void fuzz_variant(const PgFp16ConvertVariant *v,
                         pg_fp16 *src_h, pg_fp16 *out_h, pg_fp16 *ref_h,
                         float *src_f, float *out_f, float *ref_f)
{
    static const size_t edges[] = {
        0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33,
        63, 64, 65, 255, 256, 257, 1024, 4096,
    };
    const size_t ne = sizeof edges / sizeof edges[0];
    int ok = 1;

    for (int t = 0; t < 250 && ok; t++) {
        size_t n = (t < (int)ne) ? edges[t] : checkasm_rand_range(0, MAXN);

        fill_inputs(src_h, src_f, n);
        memset(out_f, 0, MAXN * sizeof(*out_f));
        memset(ref_f, 0, MAXN * sizeof(*ref_f));
        memset(out_h, 0, MAXN * sizeof(*out_h));
        memset(ref_h, 0, MAXN * sizeof(*ref_h));

        pg_fp16_to_f32_array_c(ref_f, src_h, n);
        v->to_f32(out_f, src_h, n);
        if (memcmp(out_f, ref_f, n * sizeof(*out_f)) != 0) {
            checkasm_fail("fp16_to_f32.%s n=%zu", v->name, n);
            ok = 0;
            break;
        }

        pg_f32_to_fp16_array_c(ref_h, src_f, n);
        v->from_f32(out_h, src_f, n);
        if (memcmp(out_h, ref_h, n * sizeof(*out_h)) != 0) {
            checkasm_fail("f32_to_fp16.%s n=%zu", v->name, n);
            ok = 0;
        }
    }
    checkasm_report("fp16_convert", v->name, ok);
}

static double bench_to_f32(pg_fp16_to_f32_array_fn fn, float *dst,
                           const pg_fp16 *src, size_t n, double *gbps)
{
    int iters = 1;
    double el = 0.0;

    fn(dst, src, n);
    do {
        double t0 = checkasm_now();
        for (int i = 0; i < iters; i++)
            fn(dst, src, n);
        el = checkasm_now() - t0;
        if (el < 0.05)
            iters *= 2;
    } while (el < 0.05 && iters < (1 << 22));
    double per = el / iters;
    *gbps = (double)n * (sizeof(pg_fp16) + sizeof(float)) / per / 1e9;
    return per;
}

static double bench_from_f32(pg_f32_to_fp16_array_fn fn, pg_fp16 *dst,
                             const float *src, size_t n, double *gbps)
{
    int iters = 1;
    double el = 0.0;

    fn(dst, src, n);
    do {
        double t0 = checkasm_now();
        for (int i = 0; i < iters; i++)
            fn(dst, src, n);
        el = checkasm_now() - t0;
        if (el < 0.05)
            iters *= 2;
    } while (el < 0.05 && iters < (1 << 22));
    double per = el / iters;
    *gbps = (double)n * (sizeof(float) + sizeof(pg_fp16)) / per / 1e9;
    return per;
}

void checkasm_check_fp16(void)
{
    size_t nv;
    const PgFp16ConvertVariant *v = pg_fp16_convert_variants(&nv);
    unsigned flags = pg_get_cpu_flags();
    pg_fp16 *src_h = pg_aligned_alloc(PG_ALIGN, MAXN * sizeof(*src_h));
    pg_fp16 *out_h = pg_aligned_alloc(PG_ALIGN, MAXN * sizeof(*out_h));
    pg_fp16 *ref_h = pg_aligned_alloc(PG_ALIGN, MAXN * sizeof(*ref_h));
    float *src_f = pg_aligned_alloc(PG_ALIGN, MAXN * sizeof(*src_f));
    float *out_f = pg_aligned_alloc(PG_ALIGN, MAXN * sizeof(*out_f));
    float *ref_f = pg_aligned_alloc(PG_ALIGN, MAXN * sizeof(*ref_f));

    for (size_t i = 0; i < nv; i++)
        if ((flags & v[i].req_flags) == v[i].req_flags)
            fuzz_variant(&v[i], src_h, out_h, ref_h, src_f, out_f, ref_f);

    if (checkasm_bench_enabled()) {
        static const size_t dims[] = { 64, 512, 4096, 32768 };

        for (size_t d = 0; d < sizeof dims / sizeof dims[0]; d++) {
            size_t n = dims[d];
            pg_fp16 *bh = pg_aligned_alloc(PG_ALIGN, n * sizeof(*bh));
            pg_fp16 *boh = pg_aligned_alloc(PG_ALIGN, n * sizeof(*boh));
            float *bf = pg_aligned_alloc(PG_ALIGN, n * sizeof(*bf));
            float *bof = pg_aligned_alloc(PG_ALIGN, n * sizeof(*bof));
            char title[48];

            fill_inputs(bh, bf, n);
            snprintf(title, sizeof title, "fp16_to_f32  %zu", n);
            checkasm_bench_begin(title, "GB/s");
            for (size_t i = 0; i < nv; i++) {
                if ((flags & v[i].req_flags) != v[i].req_flags)
                    continue;
                double g, per = bench_to_f32(v[i].to_f32, bof, bh, n, &g);
                checkasm_bench_row(v[i].name, per, g);
            }
            checkasm_bench_end();

            snprintf(title, sizeof title, "f32_to_fp16  %zu", n);
            checkasm_bench_begin(title, "GB/s");
            for (size_t i = 0; i < nv; i++) {
                if ((flags & v[i].req_flags) != v[i].req_flags)
                    continue;
                double g, per = bench_from_f32(v[i].from_f32, boh, bf, n, &g);
                checkasm_bench_row(v[i].name, per, g);
            }
            checkasm_bench_end();

            pg_aligned_free(bh);
            pg_aligned_free(boh);
            pg_aligned_free(bf);
            pg_aligned_free(bof);
        }
    }

    pg_aligned_free(src_h);
    pg_aligned_free(out_h);
    pg_aligned_free(ref_h);
    pg_aligned_free(src_f);
    pg_aligned_free(out_f);
    pg_aligned_free(ref_f);
}
