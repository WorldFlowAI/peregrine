/*
 * peregrine - fp16-input GEMM: runtime dispatch + variant table.
 */
#include "tensor/kernels/gemm/gemm_fp16.h"
#include "util/cpu.h"

static const PgFp16GemmVariant g_variants[] = {
#if PG_ARCH_AARCH64
    { "f16",  pg_fp16gemm_f16,  PG_CPU_NEON | PG_CPU_FP16 },
    { "neon", pg_fp16gemm_neon, PG_CPU_NEON },
#elif PG_ARCH_X86_64
    { "avx2", pg_fp16gemm_avx2, PG_CPU_AVX2 | PG_CPU_FMA },
#endif
    { "c",    pg_fp16gemm_c,    0 },
};

const PgFp16GemmVariant *pg_fp16gemm_variants(size_t *count)
{
    *count = sizeof(g_variants) / sizeof(g_variants[0]);
    return g_variants;
}

void pg_fp16gemm_dsp_init(PgFp16GemmDSP *dsp, unsigned cpu_flags)
{
#if PG_ARCH_AARCH64
    if ((cpu_flags & (PG_CPU_NEON | PG_CPU_FP16)) == (PG_CPU_NEON | PG_CPU_FP16)) {
        dsp->gemm = pg_fp16gemm_f16;
        return;
    }
    if (cpu_flags & PG_CPU_NEON) {
        dsp->gemm = pg_fp16gemm_neon;
        return;
    }
#elif PG_ARCH_X86_64
    if ((cpu_flags & (PG_CPU_AVX2 | PG_CPU_FMA)) == (PG_CPU_AVX2 | PG_CPU_FMA)) {
        dsp->gemm = pg_fp16gemm_avx2;
        return;
    }
#endif
    (void)cpu_flags;
    dsp->gemm = pg_fp16gemm_c;
}
