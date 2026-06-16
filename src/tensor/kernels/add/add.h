/*
 * peregrine - f32 elementwise add:  out[i] = a[i] + b[i]
 */
#ifndef PEREGRINE_TENSOR_KERNELS_ADD_H
#define PEREGRINE_TENSOR_KERNELS_ADD_H

#include <stddef.h>
#include "util/arch.h"

typedef void (*pg_add_f32_fn)(const float *a, const float *b, float *out, size_t n);

typedef struct PgAddDSP { pg_add_f32_fn add_f32; } PgAddDSP;

typedef struct PgAddVariant {
    const char    *name;
    pg_add_f32_fn  fn;
    unsigned       req_flags;
} PgAddVariant;

void pg_add_dsp_init(PgAddDSP *dsp, unsigned cpu_flags);
const PgAddVariant *pg_add_variants(size_t *count);

void pg_add_f32_c(const float *a, const float *b, float *out, size_t n);
#if PG_ARCH_X86_64
void pg_add_f32_avx2(const float *a, const float *b, float *out, size_t n);
#endif
#if PG_ARCH_AARCH64
void pg_add_f32_neon(const float *a, const float *b, float *out, size_t n);
#endif

#endif /* PEREGRINE_TENSOR_KERNELS_ADD_H */
