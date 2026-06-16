/*
 * peregrine - f32 GEMM: runtime dispatch + variant table.
 */
#include "tensor/kernels/gemm/gemm.h"
#include "util/cpu.h"

static const PgGemmVariant g_variants[] = {
#if PG_ARCH_AARCH64
    { "neon", pg_sgemm_neon, PG_CPU_NEON },
#elif PG_ARCH_X86_64
    { "avx2", pg_sgemm_avx2, PG_CPU_AVX2 | PG_CPU_FMA },
#endif
    /* bf16/fp16 paths are follow-ups. */
    { "c",    pg_sgemm_c,    0 },
};

const PgGemmVariant *pg_gemm_variants(size_t *count)
{
    *count = sizeof(g_variants) / sizeof(g_variants[0]);
    return g_variants;
}

void pg_gemm_dsp_init(PgGemmDSP *dsp, unsigned cpu_flags)
{
#if PG_ARCH_AARCH64
    if (cpu_flags & PG_CPU_NEON) {
        dsp->sgemm = pg_sgemm_neon;
        return;
    }
#elif PG_ARCH_X86_64
    if ((cpu_flags & (PG_CPU_AVX2 | PG_CPU_FMA)) == (PG_CPU_AVX2 | PG_CPU_FMA)) {
        dsp->sgemm = pg_sgemm_avx2;
        return;
    }
#endif
    (void)cpu_flags;
    dsp->sgemm = pg_sgemm_c;
}
