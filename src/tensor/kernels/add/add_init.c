/*
 * peregrine - f32 elementwise add: runtime dispatch + variant table.
 */
#include "tensor/kernels/add/add.h"
#include "util/cpu.h"

static const PgAddVariant g_variants[] = {
#if PG_ARCH_X86_64
    { "avx2", pg_add_f32_avx2, PG_CPU_AVX2 | PG_CPU_FMA },
#elif PG_ARCH_AARCH64
    { "neon", pg_add_f32_neon, PG_CPU_NEON },
#endif
    { "c",    pg_add_f32_c,    0 },
};

const PgAddVariant *pg_add_variants(size_t *count)
{
    *count = sizeof(g_variants) / sizeof(g_variants[0]);
    return g_variants;
}

void pg_add_dsp_init(PgAddDSP *dsp, unsigned cpu_flags)
{
    size_t n;
    const PgAddVariant *v = pg_add_variants(&n);
    for (size_t i = 0; i < n; i++)
        if ((cpu_flags & v[i].req_flags) == v[i].req_flags) {
            dsp->add_f32 = v[i].fn;
            return;
        }
    dsp->add_f32 = pg_add_f32_c;
}
