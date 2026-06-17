/*
 * peregrine - fp16 GEMM microkernel, AArch64 + FEAT_FP16.
 *
 *   void pg_fp16gemm_ukernel_8x8_f16(size_t kc, const pg_fp16 *a,
 *                                    const pg_fp16 *b, float *c, size_t ldc)
 *     x0=kc  x1=a (packed)  x2=b (packed)  x3=c  x4=ldc (elements)
 *
 * Computes an 8x8 tile with fp16 FMLA accumulation. Packed panels:
 *   a[k*8 + r] = row r at depth k
 *   b[k*8 + c] = col c at depth k
 * Accumulator v[16+r].8h holds row r, cols 0..7. Stores widen to f32.
 */
#include "ext/arm/asm.S"

.arch armv8.2-a+fp16

function fp16gemm_ukernel_8x8_f16, export=1
    movi    v16.8h, #0
    movi    v17.8h, #0
    movi    v18.8h, #0
    movi    v19.8h, #0
    movi    v20.8h, #0
    movi    v21.8h, #0
    movi    v22.8h, #0
    movi    v23.8h, #0

    cbz     x0, 2f
1:
    ld1     {v0.8h}, [x2], #16
    ld1     {v1.8h}, [x1], #16
    fmla    v16.8h, v0.8h, v1.h[0]
    fmla    v17.8h, v0.8h, v1.h[1]
    fmla    v18.8h, v0.8h, v1.h[2]
    fmla    v19.8h, v0.8h, v1.h[3]
    fmla    v20.8h, v0.8h, v1.h[4]
    fmla    v21.8h, v0.8h, v1.h[5]
    fmla    v22.8h, v0.8h, v1.h[6]
    fmla    v23.8h, v0.8h, v1.h[7]
    subs    x0, x0, #1
    b.ne    1b
2:
    lsl     x4, x4, #2

    fcvtl   v0.4s, v16.4h
    fcvtl2  v1.4s, v16.8h
    st1     {v0.4s, v1.4s}, [x3], x4
    fcvtl   v0.4s, v17.4h
    fcvtl2  v1.4s, v17.8h
    st1     {v0.4s, v1.4s}, [x3], x4
    fcvtl   v0.4s, v18.4h
    fcvtl2  v1.4s, v18.8h
    st1     {v0.4s, v1.4s}, [x3], x4
    fcvtl   v0.4s, v19.4h
    fcvtl2  v1.4s, v19.8h
    st1     {v0.4s, v1.4s}, [x3], x4
    fcvtl   v0.4s, v20.4h
    fcvtl2  v1.4s, v20.8h
    st1     {v0.4s, v1.4s}, [x3], x4
    fcvtl   v0.4s, v21.4h
    fcvtl2  v1.4s, v21.8h
    st1     {v0.4s, v1.4s}, [x3], x4
    fcvtl   v0.4s, v22.4h
    fcvtl2  v1.4s, v22.8h
    st1     {v0.4s, v1.4s}, [x3], x4
    fcvtl   v0.4s, v23.4h
    fcvtl2  v1.4s, v23.8h
    st1     {v0.4s, v1.4s}, [x3], x4
    ret
endfunc
