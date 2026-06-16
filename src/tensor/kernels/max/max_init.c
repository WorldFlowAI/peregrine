/*
 * peregrine - f32 max: runtime dispatch + variant table.
 */
#include "tensor/kernels/max/max.h"
#include "util/cpu.h"

static const PgMaxVariant g_variants[] = {
#if PG_ARCH_X86_64
    { "avx2", pg_max_f32_avx2, PG_CPU_AVX2 | PG_CPU_FMA },
#elif PG_ARCH_AARCH64
    { "neon", pg_max_f32_neon, PG_CPU_NEON },
#endif
    { "c",    pg_max_f32_c,    0 },
};

const PgMaxVariant *pg_max_variants(size_t *count)
{
    *count = sizeof(g_variants) / sizeof(g_variants[0]);
    return g_variants;
}

void pg_max_dsp_init(PgMaxDSP *dsp, unsigned cpu_flags)
{
    size_t n;
    const PgMaxVariant *v = pg_max_variants(&n);
    for (size_t i = 0; i < n; i++)
        if ((cpu_flags & v[i].req_flags) == v[i].req_flags) {
            dsp->max_f32 = v[i].fn;
            return;
        }
    dsp->max_f32 = pg_max_f32_c;
}
