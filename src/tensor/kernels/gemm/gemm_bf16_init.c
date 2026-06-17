/*
 * peregrine - bf16-input GEMM: runtime dispatch + variant table.
 */
#include "tensor/kernels/gemm/gemm_bf16.h"
#include "util/cpu.h"

static const PgBf16GemmVariant g_variants[] = {
#if PG_ARCH_AARCH64
    { "bfmmla", pg_bf16gemm_bfmmla, PG_CPU_NEON | PG_CPU_BF16 },  /* native bf16 */
    { "neon",   pg_bf16gemm_neon,   PG_CPU_NEON },                /* storage path */
#elif PG_ARCH_X86_64
    { "avx512bf16", pg_bf16gemm_avx512bf16, PG_CPU_AVX512 | PG_CPU_BF16 },
    { "avx2",       pg_bf16gemm_avx2,       PG_CPU_AVX2 | PG_CPU_FMA },
#endif
    { "c",      pg_bf16gemm_c,      0 },
};

/*
 * Note on BFMMLA: it does a 2x2x4 bf16 MAC per instruction, so it wins big on
 * cores with full-rate BFMMLA (Neoverse V1/V2). On Apple Silicon, matrix
 * throughput lives in AMX and NEON BFMMLA is NOT high-rate -- measured slower
 * than the f32-FMLA storage path (M2 Pro: bfmmla 417 vs storage 658 GFLOP/s @
 * 1024^3). So dispatch does NOT auto-prefer bfmmla on FEAT_BF16 alone; selecting
 * it should be benchmark-driven (a startup micro-bench, TODO). It stays in the
 * table so checkasm exercises and benches it everywhere.
 */

const PgBf16GemmVariant *pg_bf16gemm_variants(size_t *count)
{
    *count = sizeof(g_variants) / sizeof(g_variants[0]);
    return g_variants;
}

void pg_bf16gemm_dsp_init(PgBf16GemmDSP *dsp, unsigned cpu_flags)
{
#if PG_ARCH_AARCH64
    /* Default to the storage path: it is faster than NEON BFMMLA on Apple cores
     * (see note above). A benchmark-driven selector can promote bfmmla on cores
     * where it actually wins. */
    if (cpu_flags & PG_CPU_NEON) {
        dsp->gemm = pg_bf16gemm_neon;
        return;
    }
#elif PG_ARCH_X86_64
    if ((cpu_flags & (PG_CPU_AVX512 | PG_CPU_BF16)) == (PG_CPU_AVX512 | PG_CPU_BF16)) {
        dsp->gemm = pg_bf16gemm_avx512bf16;
        return;
    }
    if ((cpu_flags & (PG_CPU_AVX2 | PG_CPU_FMA)) == (PG_CPU_AVX2 | PG_CPU_FMA)) {
        dsp->gemm = pg_bf16gemm_avx2;
        return;
    }
#endif
    (void)cpu_flags;
    dsp->gemm = pg_bf16gemm_c;
}
