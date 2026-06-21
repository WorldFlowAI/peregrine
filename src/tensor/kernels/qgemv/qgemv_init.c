/*
 * peregrine - quantized GEMV runtime dispatch + variant tables.
 */
#include "tensor/kernels/qgemv/qgemv.h"
#include "util/cpu.h"

static const PgQgemvVariant g_q8_0_variants[] = {
#if PG_ARCH_AARCH64
    { "neon", pg_q8_0_gemv_neon, PG_CPU_NEON | PG_CPU_FP16 },
#endif
    { "c",    pg_q8_0_gemv_c,    0 },
};

static const PgQgemvVariant g_q4_k_variants[] = {
    { "c", pg_q4_k_gemv_c, 0 },
};

const PgQgemvVariant *pg_q8_0_gemv_variants(size_t *count)
{
    *count = sizeof(g_q8_0_variants) / sizeof(g_q8_0_variants[0]);
    return g_q8_0_variants;
}

const PgQgemvVariant *pg_q4_k_gemv_variants(size_t *count)
{
    *count = sizeof(g_q4_k_variants) / sizeof(g_q4_k_variants[0]);
    return g_q4_k_variants;
}

void pg_qgemv_dsp_init(PgQgemvDSP *dsp, unsigned cpu_flags)
{
    dsp->q8_0 = pg_q8_0_gemv_c;
    dsp->q8_0_dot = pg_q8_0_dot_f32_c;
    dsp->q4_k = pg_q4_k_gemv_c;
    dsp->q4_k_dot = pg_q4_k_dot_f32_c;

#if PG_ARCH_AARCH64
    if ((cpu_flags & (PG_CPU_NEON | PG_CPU_FP16)) == (PG_CPU_NEON | PG_CPU_FP16)) {
        dsp->q8_0 = pg_q8_0_gemv_neon;
        dsp->q8_0_dot = pg_q8_0_dot_f32_neon;
    }
#endif
    (void)cpu_flags;
}
