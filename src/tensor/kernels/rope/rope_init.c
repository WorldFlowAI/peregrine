/*
 * peregrine - f32 RoPE: runtime dispatch + variant table (C only for now).
 */
#include "tensor/kernels/rope/rope.h"
#include "util/cpu.h"

static const PgRopeVariant g_variants[] = {
#if PG_ARCH_AARCH64
    { "neon", pg_rope_f32_neon, PG_CPU_NEON },
#endif
    /* x86-64 AVX2 rope (with interleaved-mode de-interleave) is a follow-up. */
    { "c",    pg_rope_f32_c,    0 },
};

const PgRopeVariant *pg_rope_variants(size_t *count)
{
    *count = sizeof(g_variants) / sizeof(g_variants[0]);
    return g_variants;
}

void pg_rope_dsp_init(PgRopeDSP *dsp, unsigned cpu_flags)
{
    (void)cpu_flags;
    dsp->cache = pg_rope_cache_f32_c;
    dsp->apply = pg_rope_f32_c;
}
