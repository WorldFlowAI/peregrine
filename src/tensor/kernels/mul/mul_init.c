/*
 * peregrine - f32 elementwise multiply: runtime dispatch + variant table.
 */
#include "tensor/kernels/mul/mul.h"
#include "util/cpu.h"

static const PgMulVariant g_variants[] = {
#if PG_ARCH_X86_64
    { "avx2", pg_mul_f32_avx2, PG_CPU_AVX2 | PG_CPU_FMA },
#elif PG_ARCH_AARCH64
    { "neon", pg_mul_f32_neon, PG_CPU_NEON },
#endif
    { "c",    pg_mul_f32_c,    0 },
};

const PgMulVariant *pg_mul_variants(size_t *count)
{
    *count = sizeof(g_variants) / sizeof(g_variants[0]);
    return g_variants;
}

void pg_mul_dsp_init(PgMulDSP *dsp, unsigned cpu_flags)
{
    size_t n;
    const PgMulVariant *v = pg_mul_variants(&n);
    for (size_t i = 0; i < n; i++)
        if ((cpu_flags & v[i].req_flags) == v[i].req_flags) {
            dsp->mul_f32 = v[i].fn;
            return;
        }
    dsp->mul_f32 = pg_mul_f32_c;
}
