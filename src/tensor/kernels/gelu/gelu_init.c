/*
 * peregrine - f32 GELU: runtime dispatch + variant table (C only for now).
 */
#include "tensor/kernels/gelu/gelu.h"
#include "util/cpu.h"

static const PgGeluVariant g_variants[] = {
    { "c", pg_gelu_f32_c, 0 },
};

const PgGeluVariant *pg_gelu_variants(size_t *count)
{
    *count = sizeof(g_variants) / sizeof(g_variants[0]);
    return g_variants;
}

void pg_gelu_dsp_init(PgGeluDSP *dsp, unsigned cpu_flags)
{
    (void)cpu_flags;
    dsp->gelu_f32 = pg_gelu_f32_c;
}
