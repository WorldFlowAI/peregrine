/*
 * peregrine - f32 SiLU: runtime dispatch + variant table.
 */
#include "tensor/kernels/silu/silu.h"
#include "util/cpu.h"

static const PgSiluVariant g_variants[] = {
#if PG_ARCH_X86_64
    { "avx2", pg_silu_f32_avx2, PG_CPU_AVX2 | PG_CPU_FMA },
#elif PG_ARCH_AARCH64
    { "neon", pg_silu_f32_neon, PG_CPU_NEON },
#endif
    { "c",    pg_silu_f32_c,    0 },
};

const PgSiluVariant *pg_silu_variants(size_t *count)
{
    *count = sizeof(g_variants) / sizeof(g_variants[0]);
    return g_variants;
}

void pg_silu_dsp_init(PgSiluDSP *dsp, unsigned cpu_flags)
{
    size_t n;
    const PgSiluVariant *v = pg_silu_variants(&n);
    for (size_t i = 0; i < n; i++)
        if ((cpu_flags & v[i].req_flags) == v[i].req_flags) {
            dsp->silu_f32 = v[i].fn;
            return;
        }
    dsp->silu_f32 = pg_silu_f32_c;
}
