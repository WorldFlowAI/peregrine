/*
 * peregrine - runtime CPU feature detection
 *
 * One bitfield, queried once at startup, that every kernel dispatch table
 * consults to choose the fastest hand-written routine the host can run.
 * This is the mechanism that lets a single binary ship SSE2..AVX-512 and
 * NEON..SVE2 code paths and pick the best one at runtime (dav1d/FFmpeg model).
 */
#ifndef PEREGRINE_UTIL_CPU_H
#define PEREGRINE_UTIL_CPU_H

#include "util/arch.h"

enum PgCpuFlags {
    /* x86-64 */
    PG_CPU_SSE2   = 1u << 0,
    PG_CPU_AVX2   = 1u << 1,
    PG_CPU_FMA    = 1u << 2,
    PG_CPU_AVX512 = 1u << 3,
    /* aarch64 */
    PG_CPU_NEON   = 1u << 8,
    PG_CPU_SVE2   = 1u << 9,
    PG_CPU_BF16   = 1u << 10,  /* FEAT_BF16: BFMMLA / BFDOT */
    PG_CPU_FP16   = 1u << 11,  /* FEAT_FP16: half-precision vector FMLA */
};

/* Detect host CPU features. Cheap; safe to call more than once. */
unsigned pg_get_cpu_flags(void);

/* Human-readable flag list into buf (e.g. "neon"). Returns buf. */
const char *pg_cpu_flags_str(unsigned flags, char *buf, unsigned buflen);

#endif /* PEREGRINE_UTIL_CPU_H */
