/*
 * peregrine - f32 softmax: runtime dispatch + variant table.
 */
#include "tensor/kernels/softmax/softmax.h"
#include "util/cpu.h"

static const PgSoftmaxVariant g_variants[] = {
#if PG_ARCH_X86_64
    { "avx2", pg_softmax_f32_avx2, PG_CPU_AVX2 | PG_CPU_FMA },
#elif PG_ARCH_AARCH64
    { "neon", pg_softmax_f32_neon, PG_CPU_NEON },
#endif
    { "c",    pg_softmax_f32_c,    0 },
};

const PgSoftmaxVariant *pg_softmax_variants(size_t *count)
{
    *count = sizeof(g_variants) / sizeof(g_variants[0]);
    return g_variants;
}

void pg_softmax_dsp_init(PgSoftmaxDSP *dsp, unsigned cpu_flags)
{
    size_t n;
    const PgSoftmaxVariant *v = pg_softmax_variants(&n);
    for (size_t i = 0; i < n; i++)
        if ((cpu_flags & v[i].req_flags) == v[i].req_flags) {
            dsp->softmax_f32 = v[i].fn;
            return;
        }
    dsp->softmax_f32 = pg_softmax_f32_c;
}
