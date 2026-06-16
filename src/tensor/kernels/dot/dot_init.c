/*
 * peregrine - f32 dot product: runtime dispatch + variant table.
 *
 * The variant table is the single source of truth. Dispatch walks it and
 * takes the first entry the host CPU satisfies (fastest-first ordering);
 * checkasm walks the same table to exercise every runnable implementation.
 * Arch guards keep other ISAs out of the binary entirely.
 */
#include "tensor/kernels/dot/dot.h"
#include "util/cpu.h"

static const PgDotVariant g_variants[] = {
#if PG_ARCH_X86_64
    { "avx2", pg_dot_f32_avx2, PG_CPU_AVX2 | PG_CPU_FMA },
#elif PG_ARCH_AARCH64
    { "neon", pg_dot_f32_neon, PG_CPU_NEON },
#endif
    { "c",    pg_dot_f32_c,    0 },  /* always last, always runnable */
};

const PgDotVariant *pg_dot_variants(size_t *count)
{
    *count = sizeof(g_variants) / sizeof(g_variants[0]);
    return g_variants;
}

void pg_dot_dsp_init(PgDotDSP *dsp, unsigned cpu_flags)
{
    size_t n;
    const PgDotVariant *v = pg_dot_variants(&n);

    for (size_t i = 0; i < n; i++) {
        if ((cpu_flags & v[i].req_flags) == v[i].req_flags) {
            dsp->dot_f32 = v[i].fn;
            return;
        }
    }
    dsp->dot_f32 = pg_dot_f32_c;  /* unreachable: the "c" entry always matches */
}
