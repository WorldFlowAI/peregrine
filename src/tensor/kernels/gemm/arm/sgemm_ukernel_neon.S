/*
 * peregrine - f32 GEMM register-blocked microkernel, AArch64 NEON.
 *
 *   void pg_sgemm_ukernel_8x8_neon(size_t kc, const float *a, const float *b,
 *                                  float *c, size_t ldc)
 *     x0=kc  x1=a (packed)  x2=b (packed)  x3=c  x4=ldc (elements)
 *
 * Computes an 8x8 block of C:  C[8x8] = sum_{k<kc} a[k][0..7] (x) b[k][0..7],
 * the rank-1 outer product accumulated over the packed depth kc. Panels are
 * pre-packed by the driver so both a and b are read as contiguous 8-float rows:
 *   a[k*8 + r]  = the r-th row's element at depth k
 *   b[k*8 + col]= the col-th column's element at depth k
 * C is overwritten (the driver issues one call per 8x8 block spanning all K).
 *
 * 16 accumulators (v16..v31, all caller-saved) hold the 8x8 tile: row r lives
 * in v[16+2r] (cols 0-3) and v[17+2r] (cols 4-7). Per depth step: 2 loads + 16
 * FMAs (lane-broadcast of each a row element across the two b vectors).
 */
#include "ext/arm/asm.S"

function sgemm_ukernel_8x8_neon, export=1
    movi    v16.4s, #0
    movi    v17.4s, #0
    movi    v18.4s, #0
    movi    v19.4s, #0
    movi    v20.4s, #0
    movi    v21.4s, #0
    movi    v22.4s, #0
    movi    v23.4s, #0
    movi    v24.4s, #0
    movi    v25.4s, #0
    movi    v26.4s, #0
    movi    v27.4s, #0
    movi    v28.4s, #0
    movi    v29.4s, #0
    movi    v30.4s, #0
    movi    v31.4s, #0

    cbz     x0, 2f
1:
    ld1     {v2.4s, v3.4s}, [x2], #32     // b[k]: vb0=cols0-3, vb1=cols4-7
    ld1     {v0.4s, v1.4s}, [x1], #32     // a[k]: va0=rows0-3, va1=rows4-7
    fmla    v16.4s, v2.4s, v0.s[0]
    fmla    v17.4s, v3.4s, v0.s[0]
    fmla    v18.4s, v2.4s, v0.s[1]
    fmla    v19.4s, v3.4s, v0.s[1]
    fmla    v20.4s, v2.4s, v0.s[2]
    fmla    v21.4s, v3.4s, v0.s[2]
    fmla    v22.4s, v2.4s, v0.s[3]
    fmla    v23.4s, v3.4s, v0.s[3]
    fmla    v24.4s, v2.4s, v1.s[0]
    fmla    v25.4s, v3.4s, v1.s[0]
    fmla    v26.4s, v2.4s, v1.s[1]
    fmla    v27.4s, v3.4s, v1.s[1]
    fmla    v28.4s, v2.4s, v1.s[2]
    fmla    v29.4s, v3.4s, v1.s[2]
    fmla    v30.4s, v2.4s, v1.s[3]
    fmla    v31.4s, v3.4s, v1.s[3]
    subs    x0, x0, #1
    b.ne    1b
2:
    lsl     x4, x4, #2                    // ldc: elements -> bytes
    st1     {v16.4s, v17.4s}, [x3], x4
    st1     {v18.4s, v19.4s}, [x3], x4
    st1     {v20.4s, v21.4s}, [x3], x4
    st1     {v22.4s, v23.4s}, [x3], x4
    st1     {v24.4s, v25.4s}, [x3], x4
    st1     {v26.4s, v27.4s}, [x3], x4
    st1     {v28.4s, v29.4s}, [x3], x4
    st1     {v30.4s, v31.4s}, [x3], x4
    ret
endfunc
