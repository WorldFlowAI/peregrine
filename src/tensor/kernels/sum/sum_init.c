/*
 * peregrine - f32 sum: runtime dispatch + variant table.
 */
#include "tensor/kernels/sum/sum.h"
#include "util/cpu.h"

static const PgSumVariant g_variants[] = {
#if PG_ARCH_X86_64
    { "avx2", pg_sum_f32_avx2, PG_CPU_AVX2 | PG_CPU_FMA },
#elif PG_ARCH_AARCH64
    { "neon", pg_sum_f32_neon, PG_CPU_NEON },
#endif
    { "c",    pg_sum_f32_c,    0 },
};

const PgSumVariant *pg_sum_variants(size_t *count)
{
    *count = sizeof(g_variants) / sizeof(g_variants[0]);
    return g_variants;
}

void pg_sum_dsp_init(PgSumDSP *dsp, unsigned cpu_flags)
{
    size_t n;
    const PgSumVariant *v = pg_sum_variants(&n);
    for (size_t i = 0; i < n; i++)
        if ((cpu_flags & v[i].req_flags) == v[i].req_flags) {
            dsp->sum_f32 = v[i].fn;
            return;
        }
    dsp->sum_f32 = pg_sum_f32_c;
}
