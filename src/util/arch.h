/*
 * peregrine - architecture detection macros
 *
 * Compile-time host architecture flags. Used to guard which hand-written
 * assembly kernels and which dispatch branches are compiled in. Never use
 * these to pick an implementation at *runtime* - that is the job of the CPU
 * feature flags in cpu.h.
 */
#ifndef PEREGRINE_UTIL_ARCH_H
#define PEREGRINE_UTIL_ARCH_H

#if defined(__x86_64__) || defined(_M_X64)
#  define PG_ARCH_X86_64 1
#else
#  define PG_ARCH_X86_64 0
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#  define PG_ARCH_AARCH64 1
#else
#  define PG_ARCH_AARCH64 0
#endif

#endif /* PEREGRINE_UTIL_ARCH_H */
