/*
 * peregrine - f32 softmax: runtime dispatch + variant table (C only for now).
 */
#include "tensor/kernels/softmax/softmax.h"
#include "util/cpu.h"

static const PgSoftmaxVariant g_variants[] = {
    { "c", pg_softmax_f32_c, 0 },
};

const PgSoftmaxVariant *pg_softmax_variants(size_t *count)
{
    *count = sizeof(g_variants) / sizeof(g_variants[0]);
    return g_variants;
}

void pg_softmax_dsp_init(PgSoftmaxDSP *dsp, unsigned cpu_flags)
{
    (void)cpu_flags;
    dsp->softmax_f32 = pg_softmax_f32_c;
}
