/*
 * peregrine - fp16/f32 conversion kernels.
 */
#ifndef PEREGRINE_TENSOR_KERNELS_FP16_H
#define PEREGRINE_TENSOR_KERNELS_FP16_H

#include <stddef.h>
#include "tensor/kernels/gemm/gemm_fp16.h"
#include "util/arch.h"

typedef void (*pg_fp16_to_f32_array_fn)(float *dst, const pg_fp16 *src, size_t n);
typedef void (*pg_f32_to_fp16_array_fn)(pg_fp16 *dst, const float *src, size_t n);

typedef struct PgFp16DSP {
    pg_fp16_to_f32_array_fn to_f32;
    pg_f32_to_fp16_array_fn from_f32;
} PgFp16DSP;

typedef struct PgFp16ConvertVariant {
    const char *name;
    pg_fp16_to_f32_array_fn to_f32;
    pg_f32_to_fp16_array_fn from_f32;
    unsigned req_flags;
} PgFp16ConvertVariant;

void pg_fp16_dsp_init(PgFp16DSP *dsp, unsigned cpu_flags);
const PgFp16ConvertVariant *pg_fp16_convert_variants(size_t *count);

void pg_fp16_to_f32_array_c(float *dst, const pg_fp16 *src, size_t n);
void pg_f32_to_fp16_array_c(pg_fp16 *dst, const float *src, size_t n);

#if PG_ARCH_AARCH64
void pg_fp16_to_f32_array_neon(float *dst, const pg_fp16 *src, size_t n);
void pg_f32_to_fp16_array_neon(pg_fp16 *dst, const float *src, size_t n);
#endif

#endif /* PEREGRINE_TENSOR_KERNELS_FP16_H */
