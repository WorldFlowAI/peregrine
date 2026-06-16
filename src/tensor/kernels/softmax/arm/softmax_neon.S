/*
 * peregrine - f32 softmax over n elements, AArch64 NEON (fused, numerically stable).
 *   x0 = in, x1 = out, x2 = n
 *
 * Pass 1: max(in).   Pass 2: out = e^(in-max), accumulate sum.   Pass 3: out *= 1/sum.
 * Reuses the shared exp macro (EXP_PS/EXP_SS clobber v1..v4 / v16..v28).
 * v31 holds the broadcast max (pass 2) then the broadcast 1/sum (pass 3);
 * v5 is the max/sum accumulator; pointers x5/x6/x7 keep each pass independent.
 */
#include "tensor/kernels/simd/exp_neon.h"

function softmax_f32_neon, export=1
    cbz     x2, 9f                   // n == 0
    EXP_LOAD_CONSTS

    /* ---- pass 1: max (4 accumulators to break the serial chain) ---- */
    mov     x5, x0
    ldr     s5, [x5]
    dup     v5.4s, v5.s[0]           // seed accumulators with in[0]
    mov     v6.16b, v5.16b
    mov     v7.16b, v5.16b
    mov     v29.16b, v5.16b
    lsr     x3, x2, #4               // n / 16
    cbz     x3, 2f
1:
    ld1     {v0.4s, v1.4s, v2.4s, v3.4s}, [x5], #64
    fmax    v5.4s, v5.4s, v0.4s
    fmax    v6.4s, v6.4s, v1.4s
    fmax    v7.4s, v7.4s, v2.4s
    fmax    v29.4s, v29.4s, v3.4s
    subs    x3, x3, #1
    b.ne    1b
    fmax    v5.4s, v5.4s, v6.4s
    fmax    v7.4s, v7.4s, v29.4s
    fmax    v5.4s, v5.4s, v7.4s
2:
    fmaxv   s5, v5.4s                // horizontal max
    and     x4, x2, #15              // n % 16
    cbz     x4, 3f
10:
    ldr     s0, [x5], #4
    fmax    s5, s5, s0
    subs    x4, x4, #1
    b.ne    10b
3:
    dup     v31.4s, v5.s[0]          // broadcast max

    /* ---- pass 2: out = e^(in-max), sum ---- */
    mov     x6, x0
    mov     x7, x1
    movi    v5.16b, #0               // sum accumulator
    lsr     x3, x2, #2
    cbz     x3, 4f
11:
    ld1     {v0.4s}, [x6], #16
    fsub    v0.4s, v0.4s, v31.4s     // in - max
    EXP_PS  v0
    st1     {v0.4s}, [x7], #16
    fadd    v5.4s, v5.4s, v0.4s
    subs    x3, x3, #1
    b.ne    11b
4:
    faddp   v5.4s, v5.4s, v5.4s
    faddp   v5.4s, v5.4s, v5.4s      // s5 = partial sum
    and     x4, x2, #3
    cbz     x4, 5f
12:
    ldr     s0, [x6], #4
    fsub    s0, s0, s31
    EXP_SS  s0
    str     s0, [x7], #4
    fadd    s5, s5, s0
    subs    x4, x4, #1
    b.ne    12b
5:
    fmov    s6, #1.0
    fdiv    s5, s6, s5               // 1 / sum
    dup     v31.4s, v5.s[0]          // broadcast scale

    /* ---- pass 3: out *= 1/sum ---- */
    lsr     x3, x2, #2
    cbz     x3, 6f
13:
    ld1     {v0.4s}, [x1]
    fmul    v0.4s, v0.4s, v31.4s
    st1     {v0.4s}, [x1], #16
    subs    x3, x3, #1
    b.ne    13b
6:
    and     x2, x2, #3
    cbz     x2, 9f
14:
    ldr     s0, [x1]
    fmul    s0, s0, s31
    str     s0, [x1], #4
    subs    x2, x2, #1
    b.ne    14b
9:
    ret
endfunc
