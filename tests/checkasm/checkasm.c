/*
 * peregrine - checkasm framework (driver, RNG, reporting, bench table).
 *
 * Usage:
 *   checkasm [--bench] [--test=<name>] [--seed=<hex|dec>]
 *     --bench         also run the throughput benchmarks (off by default)
 *     --test=dot      run only the named kernel test (default: all)
 *     --seed=0x123    fix the PRNG seed (default: time-based, printed so any
 *                     failure is reproducible with the same --seed)
 */
#include "checkasm.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---- registry ----------------------------------------------------------- */
static const struct {
    const char *name;
    void (*run)(void);
} g_tests[] = {
    { "dot",     checkasm_check_dot },
    { "axpy",    checkasm_check_axpy },
    { "rmsnorm", checkasm_check_rmsnorm },
    { "sum",     checkasm_check_sum },
    { "max",     checkasm_check_max },
    { "mul",     checkasm_check_mul },
    { "add",     checkasm_check_add },
    { "silu",    checkasm_check_silu },
    { "gelu",    checkasm_check_gelu },
    { "softmax", checkasm_check_softmax },
    { "rope",    checkasm_check_rope },
    { "exp",     checkasm_check_exp },
};

/* ---- global run state --------------------------------------------------- */
static uint64_t g_state[4];
static int      g_bench;
static int      g_pass;
static int      g_fail;

/* ---- PRNG: splitmix64 seed -> xoshiro256** ------------------------------ */
static uint64_t splitmix64(uint64_t *x)
{
    uint64_t z = (*x += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

static void seed_rng(uint64_t seed)
{
    uint64_t s = seed;
    for (int i = 0; i < 4; i++)
        g_state[i] = splitmix64(&s);
}

static inline uint64_t rotl(uint64_t x, int k)
{
    return (x << k) | (x >> (64 - k));
}

uint64_t checkasm_rng(void)
{
    const uint64_t result = rotl(g_state[1] * 5, 7) * 9;
    const uint64_t t = g_state[1] << 17;
    g_state[2] ^= g_state[0];
    g_state[3] ^= g_state[1];
    g_state[1] ^= g_state[2];
    g_state[0] ^= g_state[3];
    g_state[2] ^= t;
    g_state[3] = rotl(g_state[3], 45);
    return result;
}

float checkasm_randf(float mag)
{
    /* 24 random mantissa bits -> [0,1) -> [-mag, mag) */
    uint32_t bits = (uint32_t)(checkasm_rng() >> 40);
    float u = (float)bits / (float)(1u << 24);
    return (u * 2.0f - 1.0f) * mag;
}

size_t checkasm_rand_range(size_t lo, size_t hi)
{
    if (hi <= lo)
        return lo;
    return lo + (size_t)(checkasm_rng() % (uint64_t)(hi - lo + 1));
}

/* ---- timing ------------------------------------------------------------- */
double checkasm_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ---- correctness reporting ---------------------------------------------- */
void checkasm_report(const char *kernel, const char *variant, int passed)
{
    printf("  [%s] %s.%s\n", passed ? "ok" : "FAIL", kernel, variant);
    if (passed) g_pass++;
    else        g_fail++;
}

void checkasm_fail(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    fputs("      mismatch: ", stdout);
    vprintf(fmt, ap);
    putchar('\n');
    va_end(ap);
}

int checkasm_bench_enabled(void) { return g_bench; }

/* ---- register-clobber checking ------------------------------------------ */
/* Canary values loaded into callee-saved regs by the asm trampoline. The asm
 * indexes this by 8-byte slot, so keep it a flat u64 array. */
uint64_t checkasm_register_init[18];

static uint64_t g_clobber_mask;

/* Called from the trampoline with a bitmask of clobbered register indices. */
void checkasm_clobber(uint64_t mask) { g_clobber_mask = mask; }

#if defined(__aarch64__)
static const char *const g_regnames[] = {
    "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28",
    "v8",  "v9",  "v10", "v11", "v12", "v13", "v14", "v15",
};
#elif defined(__x86_64__)
static const char *const g_regnames[] = {
    "rbx", "rbp", "r12", "r13", "r14", "r15",
};
#else
static const char *const g_regnames[] = { "?" };
#endif

static void seed_register_init(void)
{
    for (size_t i = 0; i < sizeof checkasm_register_init / sizeof checkasm_register_init[0]; i++)
        checkasm_register_init[i] = checkasm_rng();
}

int checkasm_clobbered(const char **reg)
{
    uint64_t m = g_clobber_mask;
    g_clobber_mask = 0;
    if (!m) {
        if (reg) *reg = NULL;
        return 0;
    }
    unsigned idx = (unsigned)__builtin_ctzll(m);
    if (reg)
        *reg = idx < sizeof g_regnames / sizeof g_regnames[0] ? g_regnames[idx] : "?";
    return 1;
}


/* ---- bench table -------------------------------------------------------- */
static struct {
    char   kernel[48];
    char   unit[16];
    struct { char name[16]; double per_call; double metric; } rows[16];
    int    nrows;
} g_bench_tbl;

void checkasm_bench_begin(const char *kernel, const char *unit)
{
    snprintf(g_bench_tbl.kernel, sizeof g_bench_tbl.kernel, "%s", kernel);
    snprintf(g_bench_tbl.unit, sizeof g_bench_tbl.unit, "%s", unit);
    g_bench_tbl.nrows = 0;
}

void checkasm_bench_row(const char *variant, double per_call_sec, double metric)
{
    int i = g_bench_tbl.nrows;
    if (i >= (int)(sizeof g_bench_tbl.rows / sizeof g_bench_tbl.rows[0]))
        return;
    snprintf(g_bench_tbl.rows[i].name, sizeof g_bench_tbl.rows[i].name, "%s", variant);
    g_bench_tbl.rows[i].per_call = per_call_sec;
    g_bench_tbl.rows[i].metric   = metric;
    g_bench_tbl.nrows++;
}

void checkasm_bench_end(void)
{
    /* baseline = the "c" row's per-call time, for the speedup column. */
    double base = 0.0;
    for (int i = 0; i < g_bench_tbl.nrows; i++)
        if (!strcmp(g_bench_tbl.rows[i].name, "c"))
            base = g_bench_tbl.rows[i].per_call;

    printf("\n  %s\n", g_bench_tbl.kernel);
    printf("    %-8s %14s %12s %9s\n", "variant", "ns/call", g_bench_tbl.unit, "speedup");
    for (int i = 0; i < g_bench_tbl.nrows; i++) {
        double per = g_bench_tbl.rows[i].per_call;
        double spd = (base > 0.0 && per > 0.0) ? base / per : 0.0;
        printf("    %-8s %14.1f %12.2f %8.2fx\n",
               g_bench_tbl.rows[i].name, per * 1e9, g_bench_tbl.rows[i].metric, spd);
    }
}

/* ---- driver ------------------------------------------------------------- */
int main(int argc, char **argv)
{
    const char *only = NULL;
    uint64_t seed = (uint64_t)time(NULL) ^ 0xc0ffee5eedull;
    int seed_set = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--bench"))
            g_bench = 1;
        else if (!strncmp(argv[i], "--test=", 7))
            only = argv[i] + 7;
        else if (!strncmp(argv[i], "--seed=", 7)) {
            seed = (uint64_t)strtoull(argv[i] + 7, NULL, 0);
            seed_set = 1;
        } else {
            fprintf(stderr, "checkasm: unknown arg '%s'\n", argv[i]);
            return 2;
        }
    }

    seed_rng(seed);
    seed_register_init();
    printf("peregrine checkasm  (seed: 0x%016llx%s, bench: %s)\n\n",
           (unsigned long long)seed, seed_set ? "" : ", auto",
           g_bench ? "on" : "off");

    const int ntests = (int)(sizeof g_tests / sizeof g_tests[0]);
    int ran = 0;
    for (int i = 0; i < ntests; i++) {
        if (only && strcmp(only, g_tests[i].name))
            continue;
        g_tests[i].run();
        ran++;
    }

    if (!ran) {
        fprintf(stderr, "checkasm: no test matched '%s'\n", only ? only : "");
        return 2;
    }

    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    if (g_fail)
        printf("reproduce with: --seed=0x%016llx\n", (unsigned long long)seed);
    return g_fail ? 1 : 0;
}
