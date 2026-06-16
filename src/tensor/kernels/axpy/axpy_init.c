/*
 * peregrine - f32 axpy: runtime dispatch + variant table.
 */
#include "tensor/kernels/axpy/axpy.h"
#include "util/cpu.h"

static const PgAxpyVariant g_variants[] = {
#if PG_ARCH_X86_64
    { "avx2", pg_axpy_f32_avx2, PG_CPU_AVX2 | PG_CPU_FMA },
#elif PG_ARCH_AARCH64
    { "neon", pg_axpy_f32_neon, PG_CPU_NEON },
#endif
    { "c",    pg_axpy_f32_c,    0 },
};

const PgAxpyVariant *pg_axpy_variants(size_t *count)
{
    *count = sizeof(g_variants) / sizeof(g_variants[0]);
    return g_variants;
}

void pg_axpy_dsp_init(PgAxpyDSP *dsp, unsigned cpu_flags)
{
    size_t n;
    const PgAxpyVariant *v = pg_axpy_variants(&n);
    for (size_t i = 0; i < n; i++) {
        if ((cpu_flags & v[i].req_flags) == v[i].req_flags) {
            dsp->axpy_f32 = v[i].fn;
            return;
        }
    }
    dsp->axpy_f32 = pg_axpy_f32_c;
}
