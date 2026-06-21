/*
 * peregrine - fp16/f32 conversion dispatch.
 */
#include "tensor/kernels/fp16/fp16.h"
#include "util/cpu.h"

static const PgFp16ConvertVariant g_variants[] = {
#if PG_ARCH_AARCH64
    { "neon", pg_fp16_to_f32_array_neon, pg_f32_to_fp16_array_neon,
      PG_CPU_NEON | PG_CPU_FP16 },
#endif
    { "c", pg_fp16_to_f32_array_c, pg_f32_to_fp16_array_c, 0 },
};

const PgFp16ConvertVariant *pg_fp16_convert_variants(size_t *count)
{
    *count = sizeof(g_variants) / sizeof(g_variants[0]);
    return g_variants;
}

void pg_fp16_dsp_init(PgFp16DSP *dsp, unsigned cpu_flags)
{
#if PG_ARCH_AARCH64
    if ((cpu_flags & (PG_CPU_NEON | PG_CPU_FP16)) == (PG_CPU_NEON | PG_CPU_FP16)) {
        dsp->to_f32 = pg_fp16_to_f32_array_neon;
        dsp->from_f32 = pg_f32_to_fp16_array_neon;
        return;
    }
#endif
    (void)cpu_flags;
    dsp->to_f32 = pg_fp16_to_f32_array_c;
    dsp->from_f32 = pg_f32_to_fp16_array_c;
}
