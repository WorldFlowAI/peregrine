/*
 * peregrine - checkasm: per-variant correctness fuzzing + benchmarking.
 *
 * Modelled on the dav1d/FFmpeg checkasm harness. For every kernel domain we
 * walk its variant table and, for each implementation the host CPU can run:
 *   - CORRECTNESS: fuzz many randomised inputs against the C reference.
 *   - BENCH (only under --bench): time it and print a throughput table with
 *     speedups relative to the C reference.
 *
 * A kernel test module implements one `checkasm_check_<name>(void)` and is
 * registered in the table in checkasm.c. Inside it, use the helpers below.
 *
 * NOT yet ported from upstream: the per-arch register-clobber check (verifying
 * callee-saved GPR/vector regs survive the call). That needs arch-specific asm
 * trampolines and is tracked separately in ROADMAP.md.
 */
#ifndef PEREGRINE_CHECKASM_H
#define PEREGRINE_CHECKASM_H

#include <stddef.h>
#include <stdint.h>

/* ---- test modules (each defined in its own checkasm_<name>.c) ------------ */
void checkasm_check_dot(void);
void checkasm_check_axpy(void);
void checkasm_check_rmsnorm(void);

/* ---- randomness (seedable, reproducible) -------------------------------- */
/* Uniform 64-bit value from the run's PRNG (xoshiro256**, seeded once). */
uint64_t checkasm_rng(void);
/* Uniform float in [-mag, mag). */
float    checkasm_randf(float mag);
/* Uniform size_t in [lo, hi]. */
size_t   checkasm_rand_range(size_t lo, size_t hi);

/* ---- correctness reporting ---------------------------------------------- */
/* Print a per-variant pass/fail line and tally it. `passed` is 0/1. */
void checkasm_report(const char *kernel, const char *variant, int passed);
/* Record a fuzz mismatch (prints context once, marks the run failed). */
void checkasm_fail(const char *fmt, ...);

/* ---- register-clobber checking ------------------------------------------ */
/*
 * Invoke a kernel through a per-arch trampoline that loads every callee-saved
 * register with a canary, calls the kernel, then verifies the canaries
 * survived (i.e. the kernel honoured the ABI). Integer/pointer args ride in the
 * GPRs and are shifted past the fn pointer by the trampoline; float/double args
 * ride in v0-v7 / xmm0-7 and pass straight through untouched. That covers every
 * kernel shape we have, so each one just casts the trampoline to its own
 * signature via the macros below.
 *
 * On a host with no trampoline (CK_HAVE_TRAMPOLINE undefined) the macros call
 * the kernel directly, so tests still build and run everywhere.
 */
#if defined(__aarch64__) || defined(__x86_64__)
#  define CK_HAVE_TRAMPOLINE 1
void checkasm_checked_call(void);   /* opaque; only ever called via the casts */
#endif

/* Typed call wrappers: cast the trampoline to each kernel's signature. */
#ifdef CK_HAVE_TRAMPOLINE
#  define CK_CALL_DOT(fn, a, b, n) \
     ((float (*)(void *, const float *, const float *, size_t))checkasm_checked_call) \
         ((void *)(fn), (a), (b), (n))
#  define CK_CALL_AXPY(fn, al, x, y, n) \
     ((void (*)(void *, float, const float *, float *, size_t))checkasm_checked_call) \
         ((void *)(fn), (al), (x), (y), (n))
#  define CK_CALL_RMSNORM(fn, o, x, w, n, eps) \
     ((void (*)(void *, float *, const float *, const float *, size_t, float))checkasm_checked_call) \
         ((void *)(fn), (o), (x), (w), (n), (eps))
#else
#  define CK_CALL_DOT(fn, a, b, n)             (fn)((a), (b), (n))
#  define CK_CALL_AXPY(fn, al, x, y, n)        (fn)((al), (x), (y), (n))
#  define CK_CALL_RMSNORM(fn, o, x, w, n, eps) (fn)((o), (x), (w), (n), (eps))
#endif

/*
 * If the most recent checked call clobbered a callee-saved register, return 1
 * and set *reg to its name; otherwise return 0. Clears the latch either way.
 */
int checkasm_clobbered(const char **reg);

/* ---- benchmarking -------------------------------------------------------- */
/* True when the user passed --bench. */
int    checkasm_bench_enabled(void);
/* Monotonic high-resolution clock, in seconds. */
double checkasm_now(void);

/*
 * Bench table helpers. A module calls _begin once, then _row per variant; the
 * row named "c" is treated as the baseline for the printed speedup column.
 */
void checkasm_bench_begin(const char *kernel, const char *unit);
void checkasm_bench_row(const char *variant, double per_call_sec, double metric);
void checkasm_bench_end(void);

#endif /* PEREGRINE_CHECKASM_H */
