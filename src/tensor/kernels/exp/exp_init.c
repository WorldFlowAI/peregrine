/*
 * peregrine - f32 exp: runtime dispatch + variant table.
 */
#include "tensor/kernels/exp/exp.h"
#include "util/cpu.h"

static const PgExpVariant g_variants[] = {
#if PG_ARCH_X86_64
    { "avx2", pg_exp_f32_avx2, PG_CPU_AVX2 | PG_CPU_FMA },
#elif PG_ARCH_AARCH64
    { "neon", pg_exp_f32_neon, PG_CPU_NEON },
#endif
    { "c",    pg_exp_f32_c,    0 },
};

const PgExpVariant *pg_exp_variants(size_t *count)
{
    *count = sizeof(g_variants) / sizeof(g_variants[0]);
    return g_variants;
}

void pg_exp_dsp_init(PgExpDSP *dsp, unsigned cpu_flags)
{
    size_t n;
    const PgExpVariant *v = pg_exp_variants(&n);
    for (size_t i = 0; i < n; i++)
        if ((cpu_flags & v[i].req_flags) == v[i].req_flags) {
            dsp->exp_f32 = v[i].fn;
            return;
        }
    dsp->exp_f32 = pg_exp_f32_c;
}
