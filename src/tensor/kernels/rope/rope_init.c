/*
 * peregrine - f32 RoPE: runtime dispatch + variant table (C only for now).
 */
#include "tensor/kernels/rope/rope.h"
#include "util/cpu.h"

static const PgRopeVariant g_variants[] = {
#if PG_ARCH_AARCH64
    { "neon", pg_rope_f32_neon, PG_CPU_NEON },
#elif PG_ARCH_X86_64
    { "avx2", pg_rope_f32_avx2, PG_CPU_AVX2 },
#endif
    { "c",    pg_rope_f32_c,    0 },
};

const PgRopeVariant *pg_rope_variants(size_t *count)
{
    *count = sizeof(g_variants) / sizeof(g_variants[0]);
    return g_variants;
}

void pg_rope_dsp_init(PgRopeDSP *dsp, unsigned cpu_flags)
{
    dsp->cache = pg_rope_cache_f32_c;
    dsp->apply = pg_rope_f32_c;
    for (size_t i = 0; i < sizeof(g_variants) / sizeof(g_variants[0]); i++) {
        if ((cpu_flags & g_variants[i].req_flags) == g_variants[i].req_flags) {
            dsp->apply = g_variants[i].fn;
            return;
        }
    }
}
