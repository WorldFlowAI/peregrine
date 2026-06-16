/*
 * peregrine - f32 RMSNorm, AArch64 NEON
 *
 *   void pg_rmsnorm_f32_neon(float *out, const float *x, const float *w,
 *                            size_t n, float eps)
 *     x0=out, x1=x, x2=w, x3=n, s0=eps
 *
 * Pass 1: sum of squares of x (4 accumulators + reduce + scalar tail).
 * Scale:  scale = 1 / sqrt(ss/n + eps), computed in scalar FP.
 * Pass 2: out[i] = x[i] * scale * w[i].
 */
#include "ext/arm/asm.S"

function rmsnorm_f32_neon, export=1
    cbz     x3, 9f                 // n == 0: nothing to do

    /* ---- pass 1: ss = sum(x*x) ---- */
    movi    v1.16b, #0
    movi    v2.16b, #0
    movi    v3.16b, #0
    movi    v4.16b, #0
    mov     x5, x1                // x5 walks x (keep x1 for pass 2)
    lsr     x4, x3, #4            // n / 16
    cbz     x4, 2f
1:
    ld1     {v16.4s, v17.4s, v18.4s, v19.4s}, [x5], #64
    fmla    v1.4s, v16.4s, v16.4s
    fmla    v2.4s, v17.4s, v17.4s
    fmla    v3.4s, v18.4s, v18.4s
    fmla    v4.4s, v19.4s, v19.4s
    subs    x4, x4, #1
    b.ne    1b
2:
    fadd    v1.4s, v1.4s, v2.4s
    fadd    v3.4s, v3.4s, v4.4s
    fadd    v1.4s, v1.4s, v3.4s
    faddp   v1.4s, v1.4s, v1.4s
    faddp   v1.4s, v1.4s, v1.4s   // s1 = partial ss
    and     x4, x3, #15
    cbz     x4, 4f
3:
    ldr     s16, [x5], #4
    fmadd   s1, s16, s16, s1
    subs    x4, x4, #1
    b.ne    3b
4:
    /* ---- scale = 1 / sqrt(ss/n + eps) ---- */
    ucvtf   s2, x3                // (float)n
    fdiv    s1, s1, s2            // mean = ss / n
    fadd    s1, s1, s0            // + eps
    fsqrt   s1, s1
    fmov    s2, #1.0
    fdiv    s1, s2, s1            // scale (scalar, in s1)
    dup     v5.4s, v1.s[0]        // broadcast scale

    /* ---- pass 2: out = x * scale * w ---- */
    lsr     x4, x3, #4            // n / 16
    cbz     x4, 6f
5:
    ld1     {v16.4s, v17.4s, v18.4s, v19.4s}, [x1], #64   // x
    ld1     {v20.4s, v21.4s, v22.4s, v23.4s}, [x2], #64   // w
    fmul    v16.4s, v16.4s, v5.4s
    fmul    v17.4s, v17.4s, v5.4s
    fmul    v18.4s, v18.4s, v5.4s
    fmul    v19.4s, v19.4s, v5.4s
    fmul    v16.4s, v16.4s, v20.4s
    fmul    v17.4s, v17.4s, v21.4s
    fmul    v18.4s, v18.4s, v22.4s
    fmul    v19.4s, v19.4s, v23.4s
    st1     {v16.4s, v17.4s, v18.4s, v19.4s}, [x0], #64   // out
    subs    x4, x4, #1
    b.ne    5b
6:
    and     x4, x3, #15
    cbz     x4, 9f
7:
    ldr     s16, [x1], #4
    ldr     s20, [x2], #4
    fmul    s16, s16, s1          // * scale
    fmul    s16, s16, s20         // * w
    str     s16, [x0], #4
    subs    x4, x4, #1
    b.ne    7b
9:
    ret
endfunc
