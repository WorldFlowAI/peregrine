/*
 * peregrine - f32 GEMV: runtime dispatch + variant table.
 */
#include "tensor/kernels/gemv/gemv.h"
#include "util/cpu.h"

static const PgGemvVariant g_variants[] = {
#if PG_ARCH_AARCH64
    { "neon", pg_sgemv_neon, PG_CPU_NEON },
#elif PG_ARCH_X86_64
    { "avx2", pg_sgemv_avx2, PG_CPU_AVX2 | PG_CPU_FMA },
#endif
    { "c",    pg_sgemv_c,    0 },
};

const PgGemvVariant *pg_gemv_variants(size_t *count)
{
    *count = sizeof(g_variants) / sizeof(g_variants[0]);
    return g_variants;
}

void pg_gemv_dsp_init(PgGemvDSP *dsp, unsigned cpu_flags)
{
#if PG_ARCH_AARCH64
    if (cpu_flags & PG_CPU_NEON) {
        dsp->sgemv = pg_sgemv_neon;
        return;
    }
#elif PG_ARCH_X86_64
    if ((cpu_flags & (PG_CPU_AVX2 | PG_CPU_FMA)) == (PG_CPU_AVX2 | PG_CPU_FMA)) {
        dsp->sgemv = pg_sgemv_avx2;
        return;
    }
#endif
    (void)cpu_flags;
    dsp->sgemv = pg_sgemv_c;
}
