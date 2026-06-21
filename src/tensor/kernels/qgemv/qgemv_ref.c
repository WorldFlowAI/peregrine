/*
 * peregrine - quantized-weight GEMV portable references.
 */
#include "tensor/kernels/qgemv/qgemv.h"

#include "tensor/kernels/gemm/gemm_fp16.h"

#include <stdint.h>

static uint16_t rd_le16(const void *p)
{
    const unsigned char *b = p;
    return (uint16_t)b[0] | (uint16_t)((uint16_t)b[1] << 8);
}

size_t pg_q8_0_row_bytes(size_t K)
{
    if (K == 0 || K % PG_Q8_0_BLOCK != 0)
        return 0;
    return (K / PG_Q8_0_BLOCK) * PG_Q8_0_BLOCK_BYTES;
}

size_t pg_q4_k_row_bytes(size_t K)
{
    if (K == 0 || K % PG_Q4_K_BLOCK != 0)
        return 0;
    return (K / PG_Q4_K_BLOCK) * PG_Q4_K_BLOCK_BYTES;
}

void pg_q8_0_dequant_row(const void *row, float *dst, size_t K)
{
    const unsigned char *p = row;
    size_t nb = K / PG_Q8_0_BLOCK;

    for (size_t b = 0; b < nb; b++) {
        float d = pg_fp16_to_f32(rd_le16(p));
        const int8_t *q = (const int8_t *)(const void *)(p + 2);

        for (size_t i = 0; i < PG_Q8_0_BLOCK; i++)
            dst[b * PG_Q8_0_BLOCK + i] = (float)q[i] * d;
        p += PG_Q8_0_BLOCK_BYTES;
    }
}

float pg_q8_0_dot_f32_c(const void *row, const float *x, size_t K)
{
    const unsigned char *p = row;
    size_t nb = K / PG_Q8_0_BLOCK;
    float sum = 0.0f;

    for (size_t b = 0; b < nb; b++) {
        float d = pg_fp16_to_f32(rd_le16(p));
        const int8_t *q = (const int8_t *)(const void *)(p + 2);
        const float *xb = x + b * PG_Q8_0_BLOCK;

        for (size_t i = 0; i < PG_Q8_0_BLOCK; i++)
            sum += ((float)q[i] * d) * xb[i];
        p += PG_Q8_0_BLOCK_BYTES;
    }
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

void pg_q4_k_dequant_row(const void *row, float *dst, size_t K)
{
    const unsigned char *p = row;
    size_t nb = K / PG_Q4_K_BLOCK;

    for (size_t b = 0; b < nb; b++) {
        const unsigned char *scales = p + 4;
        const unsigned char *q = p + 16;
        float d = pg_fp16_to_f32(rd_le16(p));
        float dmin = pg_fp16_to_f32(rd_le16(p + 2));
        int is = 0;

        for (size_t j = 0; j < PG_Q4_K_BLOCK; j += 64) {
            unsigned char sc, m;
            float d1, m1, d2, m2;

            get_scale_min_k4(is + 0, scales, &sc, &m);
            d1 = d * (float)sc;
            m1 = dmin * (float)m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            d2 = d * (float)sc;
            m2 = dmin * (float)m;

            for (size_t l = 0; l < 32; l++)
                dst[b * PG_Q4_K_BLOCK + j + l] = d1 * (float)(q[l] & 0x0f) - m1;
            for (size_t l = 0; l < 32; l++)
                dst[b * PG_Q4_K_BLOCK + j + 32 + l] = d2 * (float)(q[l] >> 4) - m2;
            q += 32;
            is += 2;
        }
        p += PG_Q4_K_BLOCK_BYTES;
    }
}

float pg_q4_k_dot_f32_c(const void *row, const float *x, size_t K)
{
    const unsigned char *p = row;
    size_t nb = K / PG_Q4_K_BLOCK;
    float sum = 0.0f;

    for (size_t b = 0; b < nb; b++) {
        const unsigned char *scales = p + 4;
        const unsigned char *q = p + 16;
        const float *xb = x + b * PG_Q4_K_BLOCK;
        float d = pg_fp16_to_f32(rd_le16(p));
        float dmin = pg_fp16_to_f32(rd_le16(p + 2));
        int is = 0;

        for (size_t j = 0; j < PG_Q4_K_BLOCK; j += 64) {
            unsigned char sc, m;
            float d1, m1, d2, m2;

            get_scale_min_k4(is + 0, scales, &sc, &m);
            d1 = d * (float)sc;
            m1 = dmin * (float)m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            d2 = d * (float)sc;
            m2 = dmin * (float)m;

            for (size_t l = 0; l < 32; l++)
                sum += (d1 * (float)(q[l] & 0x0f) - m1) * xb[j + l];
            for (size_t l = 0; l < 32; l++)
                sum += (d2 * (float)(q[l] >> 4) - m2) * xb[j + 32 + l];
            q += 32;
            is += 2;
        }
        p += PG_Q4_K_BLOCK_BYTES;
    }
    return sum;
}

void pg_q8_0_gemv_c(size_t M, size_t K, const void *A, size_t row_bytes,
                    const float *x, float *y)
{
    pg_qgemv_driver(M, K, A, row_bytes, x, y, pg_q8_0_dot_f32_c);
}

void pg_q4_k_gemv_c(size_t M, size_t K, const void *A, size_t row_bytes,
                    const float *x, float *y)
{
    pg_qgemv_driver(M, K, A, row_bytes, x, y, pg_q4_k_dot_f32_c);
}
