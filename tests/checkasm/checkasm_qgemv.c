/*
 * peregrine - checkasm module for GGUF quantized GEMV.
 *
 * Each output row is validated against a double-precision dequant-dot oracle.
 * The tolerance scales with sum(abs(dequant*x)), not result magnitude, because
 * signed random products cancel heavily.
 */
#include "checkasm.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tensor/kernels/gemm/gemm_fp16.h"
#include "tensor/kernels/qgemv/qgemv.h"
#include "util/cpu.h"
#include "util/mem.h"

#define MAXM_Q8 512
#define MAXK_Q8 1376
#define MAXM_Q4 512
#define MAXK_Q4 2048

typedef double (*oracle_row_fn)(const void *row, const float *x,
                                size_t K, double *work);
typedef void (*fill_matrix_fn)(unsigned char *A, size_t M, size_t K,
                               size_t stride);

static float *fbuf(size_t n)
{
    float *p = pg_aligned_alloc(PG_ALIGN, n * sizeof(float));
    if (p)
        memset(p, 0, n * sizeof(float));
    return p;
}

static unsigned char *qbuf(size_t n)
{
    unsigned char *p = pg_aligned_alloc(PG_ALIGN, n ? n : 1);
    if (p)
        memset(p, 0, n ? n : 1);
    return p;
}

static void wr_le16(void *p, uint16_t v)
{
    unsigned char *b = p;
    b[0] = (unsigned char)v;
    b[1] = (unsigned char)(v >> 8);
}

static uint16_t rd_le16(const void *p)
{
    const unsigned char *b = p;
    return (uint16_t)b[0] | (uint16_t)((uint16_t)b[1] << 8);
}

static float rand_scale(float mag)
{
    float d = checkasm_randf(mag);

    if (fabsf(d) < 0.001f)
        d = d < 0.0f ? -0.001f : 0.001f;
    return pg_fp16_to_f32(pg_f32_to_fp16(d));
}

static void fill_q8_0(unsigned char *A, size_t M, size_t K, size_t stride)
{
    size_t nb = K / PG_Q8_0_BLOCK;

    for (size_t r = 0; r < M; r++) {
        unsigned char *row = A + r * stride;

        for (size_t b = 0; b < nb; b++) {
            unsigned char *blk = row + b * PG_Q8_0_BLOCK_BYTES;
            float d = rand_scale(0.08f);

            wr_le16(blk, pg_f32_to_fp16(d));
            for (size_t i = 0; i < PG_Q8_0_BLOCK; i++)
                blk[2 + i] = (unsigned char)((int)(checkasm_rng() % 255u) - 127);
        }
    }
}

static void fill_q4_k(unsigned char *A, size_t M, size_t K, size_t stride)
{
    size_t nb = K / PG_Q4_K_BLOCK;

    for (size_t r = 0; r < M; r++) {
        unsigned char *row = A + r * stride;

        for (size_t b = 0; b < nb; b++) {
            unsigned char *blk = row + b * PG_Q4_K_BLOCK_BYTES;

            wr_le16(blk, pg_f32_to_fp16(fabsf(rand_scale(0.01f))));
            wr_le16(blk + 2, pg_f32_to_fp16(fabsf(rand_scale(0.01f))));
            for (size_t i = 0; i < 12; i++)
                blk[4 + i] = (unsigned char)checkasm_rng();
            for (size_t i = 0; i < PG_Q4_K_BLOCK / 2; i++)
                blk[16 + i] = (unsigned char)checkasm_rng();
        }
    }
}

static double oracle_q8_0_row(const void *rowv, const float *x,
                              size_t K, double *work)
{
    const unsigned char *row = rowv;
    size_t nb = K / PG_Q8_0_BLOCK;
    double sum = 0.0;
    double w = 0.0;

    for (size_t b = 0; b < nb; b++) {
        const unsigned char *blk = row + b * PG_Q8_0_BLOCK_BYTES;
        const int8_t *q = (const int8_t *)(const void *)(blk + 2);
        double d = (double)pg_fp16_to_f32(rd_le16(blk));

        for (size_t i = 0; i < PG_Q8_0_BLOCK; i++) {
            double p = d * (double)q[i] *
                       (double)x[b * PG_Q8_0_BLOCK + i];
            sum += p;
            w += fabs(p);
        }
    }
    *work = w;
    return sum;
}

static void get_scale_min_k4(int j, const unsigned char *q,
                             unsigned char *d, unsigned char *m)
{
    if (j < 4) {
        *d = q[j] & 63u;
        *m = q[j + 4] & 63u;
    } else {
        *d = (q[j + 4] & 0x0fu) | (unsigned char)((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | (unsigned char)((q[j] >> 6) << 4);
    }
}

static double oracle_q4_k_row(const void *rowv, const float *x,
                              size_t K, double *work)
{
    const unsigned char *row = rowv;
    size_t nb = K / PG_Q4_K_BLOCK;
    double sum = 0.0;
    double w = 0.0;

    for (size_t b = 0; b < nb; b++) {
        const unsigned char *blk = row + b * PG_Q4_K_BLOCK_BYTES;
        const unsigned char *scales = blk + 4;
        const unsigned char *q = blk + 16;
        const float *xb = x + b * PG_Q4_K_BLOCK;
        double d = (double)pg_fp16_to_f32(rd_le16(blk));
        double dmin = (double)pg_fp16_to_f32(rd_le16(blk + 2));
        int is = 0;

        for (size_t j = 0; j < PG_Q4_K_BLOCK; j += 64) {
            unsigned char sc, m;
            double d1, m1, d2, m2;

            get_scale_min_k4(is + 0, scales, &sc, &m);
            d1 = d * (double)sc;
            m1 = dmin * (double)m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            d2 = d * (double)sc;
            m2 = dmin * (double)m;

            for (size_t l = 0; l < 32; l++) {
                double p = (d1 * (double)(q[l] & 0x0f) - m1) *
                           (double)xb[j + l];
                sum += p;
                w += fabs(p);
            }
            for (size_t l = 0; l < 32; l++) {
                double p = (d2 * (double)(q[l] >> 4) - m2) *
                           (double)xb[j + 32 + l];
                sum += p;
                w += fabs(p);
            }
            q += 32;
            is += 2;
        }
    }
    *work = w;
    return sum;
}

static void fuzz_family(const char *kernel, const PgQgemvVariant *v,
                        fill_matrix_fn fill, oracle_row_fn oracle,
                        size_t (*row_bytes_fn)(size_t),
                        const size_t (*edges)[2], size_t ne,
                        size_t maxM, size_t maxK,
                        unsigned char *A, float *x, float *y)
{
    int ok = 1;

    for (int t = 0; t < 250 && ok; t++) {
        size_t M = (t < (int)ne) ? edges[t][0] :
                   checkasm_rand_range(1, maxM);
        size_t K = (t < (int)ne) ? edges[t][1] :
                   checkasm_rand_range(1, maxK);
        size_t row_bytes;
        size_t stride;

        if (row_bytes_fn == pg_q8_0_row_bytes)
            K = (K + PG_Q8_0_BLOCK - 1) & ~(size_t)(PG_Q8_0_BLOCK - 1);
        else
            K = (K + PG_Q4_K_BLOCK - 1) & ~(size_t)(PG_Q4_K_BLOCK - 1);
        if (K > maxK)
            K = maxK;
        row_bytes = row_bytes_fn(K);
        stride = row_bytes + 13;

        fill(A, M, K, stride);
        for (size_t k = 0; k < K; k++)
            x[k] = checkasm_randf(2.0f);

        v->fn(M, K, A, stride, x, y);
        for (size_t r = 0; r < M && ok; r++) {
            double work = 0.0;
            double ref = oracle(A + r * stride, x, K, &work);
            double tol = 96.0 * FLT_EPSILON * work + 2.0e-5;

            if (fabs((double)y[r] - ref) > tol) {
                checkasm_fail("%s.%s M=%zu K=%zu row=%zu ref=%g got=%g work=%g tol=%g",
                              kernel, v->name, M, K, r, ref, y[r], work, tol);
                ok = 0;
            }
        }
    }
    checkasm_report(kernel, v->name, ok);
}

static double bench_variant(pg_qgemv_f32_fn fn, const unsigned char *A,
                            const float *x, float *y,
                            size_t M, size_t K, size_t row_bytes,
                            double *gflops)
{
    volatile float sink = 0.0f;
    int iters = 1;
    double el = 0.0;

    fn(M, K, A, row_bytes, x, y);
    do {
        double t0 = checkasm_now();

        for (int i = 0; i < iters; i++)
            fn(M, K, A, row_bytes, x, y);
        el = checkasm_now() - t0;
        if (el < 0.05)
            iters *= 2;
    } while (el < 0.05 && iters < (1 << 22));
    sink += y[0];
    (void)sink;

    double per = el / iters;
    *gflops = 2.0 * (double)M * (double)K / per / 1e9;
    return per;
}

static void bench_family(const char *title_prefix, const PgQgemvVariant *v,
                         size_t nv, unsigned flags, fill_matrix_fn fill,
                         size_t (*row_bytes_fn)(size_t),
                         const size_t (*dims)[2], size_t nd)
{
    for (size_t s = 0; s < nd; s++) {
        size_t M = dims[s][0];
        size_t K = dims[s][1];
        size_t row_bytes = row_bytes_fn(K);
        unsigned char *bA = qbuf(M * row_bytes);
        float *bx = fbuf(K), *by = fbuf(M);
        char title[64];

        fill(bA, M, K, row_bytes);
        for (size_t k = 0; k < K; k++)
            bx[k] = checkasm_randf(1.0f);

        snprintf(title, sizeof title, "%s  %zux%zu", title_prefix, M, K);
        checkasm_bench_begin(title, "GFLOP/s");
        for (size_t i = 0; i < nv; i++) {
            if ((flags & v[i].req_flags) != v[i].req_flags)
                continue;
            double g, per = bench_variant(v[i].fn, bA, bx, by,
                                          M, K, row_bytes, &g);
            checkasm_bench_row(v[i].name, per, g);
        }
        checkasm_bench_end();

        pg_aligned_free(bA);
        pg_aligned_free(bx);
        pg_aligned_free(by);
    }
}

void checkasm_check_qgemv(void)
{
    static const size_t q8_edges[][2] = {
        { 0, 0 }, { 1, 32 }, { 2, 64 }, { 3, 96 }, { 7, 128 },
        { 8, 256 }, { 17, 512 }, { 64, 512 }, { 512, 1376 },
    };
    static const size_t q4_edges[][2] = {
        { 0, 0 }, { 1, 256 }, { 2, 512 }, { 7, 768 },
        { 17, 1024 }, { 64, 512 }, { 512, 2048 },
    };
    size_t nq8, nq4;
    const PgQgemvVariant *q8 = pg_q8_0_gemv_variants(&nq8);
    const PgQgemvVariant *q4 = pg_q4_k_gemv_variants(&nq4);
    unsigned flags = pg_get_cpu_flags();
    size_t max_q8_stride = pg_q8_0_row_bytes(MAXK_Q8) + 13;
    size_t max_q4_stride = pg_q4_k_row_bytes(MAXK_Q4) + 13;
    size_t max_stride = max_q8_stride > max_q4_stride ?
                        max_q8_stride : max_q4_stride;
    unsigned char *A = qbuf(MAXM_Q4 * max_stride);
    float *x = fbuf(MAXK_Q4);
    float *y = fbuf(MAXM_Q4);

    for (size_t i = 0; i < nq8; i++)
        if ((flags & q8[i].req_flags) == q8[i].req_flags)
            fuzz_family("q8_0_gemv", &q8[i], fill_q8_0, oracle_q8_0_row,
                        pg_q8_0_row_bytes,
                        q8_edges, sizeof q8_edges / sizeof q8_edges[0],
                        MAXM_Q8, MAXK_Q8, A, x, y);

    for (size_t i = 0; i < nq4; i++)
        if ((flags & q4[i].req_flags) == q4[i].req_flags)
            fuzz_family("q4_k_gemv", &q4[i], fill_q4_k, oracle_q4_k_row,
                        pg_q4_k_row_bytes,
                        q4_edges, sizeof q4_edges / sizeof q4_edges[0],
                        MAXM_Q4, MAXK_Q4, A, x, y);

    if (checkasm_bench_enabled()) {
        static const size_t q8_dims[][2] = {
            { 512, 512 }, { 512, 1376 }, { 1376, 512 },
            { 32000, 512 }, { 4096, 4096 },
        };
        static const size_t q4_dims[][2] = {
            { 512, 512 }, { 1376, 512 }, { 32000, 512 },
            { 4096, 4096 },
        };

        bench_family("q8_0_gemv", q8, nq8, flags, fill_q8_0,
                     pg_q8_0_row_bytes,
                     q8_dims, sizeof q8_dims / sizeof q8_dims[0]);
        bench_family("q4_k_gemv", q4, nq4, flags, fill_q4_k,
                     pg_q4_k_row_bytes,
                     q4_dims, sizeof q4_dims / sizeof q4_dims[0]);
    }

    pg_aligned_free(A);
    pg_aligned_free(x);
    pg_aligned_free(y);
}
