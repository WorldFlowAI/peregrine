/*
 * peregrine - runtime CPU feature detection (implementation)
 */
#include "util/cpu.h"

#include <string.h>
#include <stdio.h>

#if PG_ARCH_X86_64
#  include <cpuid.h>
#endif

#if PG_ARCH_AARCH64 && defined(__linux__)
#  include <sys/auxv.h>
#  include <asm/hwcap.h>
#endif

unsigned pg_get_cpu_flags(void)
{
    unsigned flags = 0;

#if PG_ARCH_X86_64
    unsigned a, b, c, d;

    /* NOTE: a production build must also verify OS support for the AVX state
     * via xgetbv before advertising AVX2/AVX-512, or we risk #UD on kernels
     * that disabled extended state. Left as a TODO for the scaffold. */
    if (__get_cpuid(1, &a, &b, &c, &d)) {
        if (d & (1u << 26)) flags |= PG_CPU_SSE2;
        if (c & (1u << 12)) flags |= PG_CPU_FMA;
    }
    if (__get_cpuid_count(7, 0, &a, &b, &c, &d)) {
        if (b & (1u << 5))  flags |= PG_CPU_AVX2;
        if (b & (1u << 16)) flags |= PG_CPU_AVX512;
    }
#elif PG_ARCH_AARCH64
    /* NEON (ASIMD) is mandatory on every AArch64 core. */
    flags |= PG_CPU_NEON;
#  if defined(__linux__)
    {
        unsigned long hw2 = getauxval(AT_HWCAP2);
#    ifdef HWCAP2_SVE2
        if (hw2 & HWCAP2_SVE2) flags |= PG_CPU_SVE2;
#    endif
    }
#  endif
    /* macOS/Apple: SVE2 not exposed; NEON/AMX only. */
#endif

    return flags;
}

const char *pg_cpu_flags_str(unsigned flags, char *buf, unsigned buflen)
{
    static const struct { unsigned bit; const char *name; } tbl[] = {
        { PG_CPU_SSE2, "sse2" }, { PG_CPU_AVX2, "avx2" },
        { PG_CPU_FMA, "fma" },   { PG_CPU_AVX512, "avx512" },
        { PG_CPU_NEON, "neon" }, { PG_CPU_SVE2, "sve2" },
    };
    unsigned i, off = 0;
    if (!buflen) return buf;
    buf[0] = '\0';
    for (i = 0; i < sizeof(tbl) / sizeof(tbl[0]); i++) {
        if (!(flags & tbl[i].bit)) continue;
        off += (unsigned)snprintf(buf + off, off < buflen ? buflen - off : 0,
                                  "%s%s", off ? " " : "", tbl[i].name);
    }
    return buf;
}
