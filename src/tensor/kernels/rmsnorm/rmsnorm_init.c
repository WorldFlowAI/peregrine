/*
 * peregrine - f32 RMSNorm: runtime dispatch + variant table.
 */
#include "tensor/kernels/rmsnorm/rmsnorm.h"
#include "util/cpu.h"

static const PgRmsnormVariant g_variants[] = {
#if PG_ARCH_X86_64
    { "avx2", pg_rmsnorm_f32_avx2, PG_CPU_AVX2 | PG_CPU_FMA },
#elif PG_ARCH_AARCH64
    { "neon", pg_rmsnorm_f32_neon, PG_CPU_NEON },
#endif
    { "c",    pg_rmsnorm_f32_c,    0 },
};

const PgRmsnormVariant *pg_rmsnorm_variants(size_t *count)
{
    *count = sizeof(g_variants) / sizeof(g_variants[0]);
    return g_variants;
}

void pg_rmsnorm_dsp_init(PgRmsnormDSP *dsp, unsigned cpu_flags)
{
    size_t n;
    const PgRmsnormVariant *v = pg_rmsnorm_variants(&n);
    for (size_t i = 0; i < n; i++) {
        if ((cpu_flags & v[i].req_flags) == v[i].req_flags) {
            dsp->rmsnorm_f32 = v[i].fn;
            return;
        }
    }
    dsp->rmsnorm_f32 = pg_rmsnorm_f32_c;
}
