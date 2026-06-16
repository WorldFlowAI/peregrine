/*
 * peregrine - f32 SiLU: runtime dispatch + variant table (C only for now).
 */
#include "tensor/kernels/silu/silu.h"
#include "util/cpu.h"

static const PgSiluVariant g_variants[] = {
    { "c", pg_silu_f32_c, 0 },
};

const PgSiluVariant *pg_silu_variants(size_t *count)
{
    *count = sizeof(g_variants) / sizeof(g_variants[0]);
    return g_variants;
}

void pg_silu_dsp_init(PgSiluDSP *dsp, unsigned cpu_flags)
{
    (void)cpu_flags;
    dsp->silu_f32 = pg_silu_f32_c;
}
