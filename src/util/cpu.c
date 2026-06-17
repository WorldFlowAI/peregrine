/*
 * peregrine - runtime CPU feature detection (implementation)
 */
/* sysctl Darwin extensions are hidden under strict -std=c11 / _POSIX_C_SOURCE;
 * _DARWIN_C_SOURCE re-exposes them (and the BSD types <sys/sysctl.h> needs). */
#define _DARWIN_C_SOURCE 1

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

#if PG_ARCH_AARCH64 && defined(__APPLE__)
#  include <sys/sysctl.h>
static int pg_sysctl_flag(const char *name)
{
    int v = 0;
    size_t sz = sizeof v;
    return sysctlbyname(name, &v, &sz, NULL, 0) == 0 && v != 0;
}
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
        unsigned long hw = getauxval(AT_HWCAP);
        unsigned long hw2 = getauxval(AT_HWCAP2);
#    ifdef HWCAP_ASIMDHP
        if (hw & HWCAP_ASIMDHP) flags |= PG_CPU_FP16;
#    else
        (void)hw;
#    endif
#    ifdef HWCAP2_SVE2
        if (hw2 & HWCAP2_SVE2) flags |= PG_CPU_SVE2;
#    endif
#    ifdef HWCAP2_BF16
        if (hw2 & HWCAP2_BF16) flags |= PG_CPU_BF16;
#    endif
    }
#  elif defined(__APPLE__)
    if (pg_sysctl_flag("hw.optional.arm.FEAT_BF16")) flags |= PG_CPU_BF16;
    if (pg_sysctl_flag("hw.optional.arm.FEAT_FP16")) flags |= PG_CPU_FP16;
#  endif
#endif

    return flags;
}

const char *pg_cpu_flags_str(unsigned flags, char *buf, unsigned buflen)
{
    static const struct { unsigned bit; const char *name; } tbl[] = {
        { PG_CPU_SSE2, "sse2" }, { PG_CPU_AVX2, "avx2" },
        { PG_CPU_FMA, "fma" },   { PG_CPU_AVX512, "avx512" },
        { PG_CPU_NEON, "neon" }, { PG_CPU_SVE2, "sve2" },
        { PG_CPU_BF16, "bf16" }, { PG_CPU_FP16, "fp16" },
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
