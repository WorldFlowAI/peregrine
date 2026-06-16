/*
 * peregrine - f32 GELU: runtime dispatch + variant table.
 */
#include "tensor/kernels/gelu/gelu.h"
#include "util/cpu.h"

static const PgGeluVariant g_variants[] = {
#if PG_ARCH_X86_64
    { "avx2", pg_gelu_f32_avx2, PG_CPU_AVX2 | PG_CPU_FMA },
#elif PG_ARCH_AARCH64
    { "neon", pg_gelu_f32_neon, PG_CPU_NEON },
#endif
    { "c",    pg_gelu_f32_c,    0 },
};

const PgGeluVariant *pg_gelu_variants(size_t *count)
{
    *count = sizeof(g_variants) / sizeof(g_variants[0]);
    return g_variants;
}

void pg_gelu_dsp_init(PgGeluDSP *dsp, unsigned cpu_flags)
{
    size_t n;
    const PgGeluVariant *v = pg_gelu_variants(&n);
    for (size_t i = 0; i < n; i++)
        if ((cpu_flags & v[i].req_flags) == v[i].req_flags) {
            dsp->gelu_f32 = v[i].fn;
            return;
        }
    dsp->gelu_f32 = pg_gelu_f32_c;
}
