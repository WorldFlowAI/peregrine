/*
 * peregrine - f32 GEMM: runtime dispatch + variant table.
 */
#include "tensor/kernels/gemm/gemm.h"
#include "util/cpu.h"

static const PgGemmVariant g_variants[] = {
#if PG_ARCH_AARCH64
    { "neon", pg_sgemm_neon, PG_CPU_NEON },
#endif
    /* x86-64 AVX2 microkernel + bf16/fp16 paths are follow-ups. */
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
#endif
    (void)cpu_flags;
    dsp->sgemm = pg_sgemm_c;
}
