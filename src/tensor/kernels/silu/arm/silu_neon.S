/*
 * peregrine - f32 SiLU, AArch64 NEON:  out[i] = x / (1 + e^-x)
 *   x0 = in, x1 = out, x2 = n
 *
 * Reuses the shared EXP_PS / EXP_SS macros. x is kept in v5/s5 (which the exp
 * macros do not touch); v25 holds 1.0 after EXP_LOAD_CONSTS.
 */
#include "tensor/kernels/simd/exp_neon.h"

function silu_f32_neon, export=1
    EXP_LOAD_CONSTS
    lsr     x3, x2, #2            // n / 4
    cbz     x3, 2f
1:
    ld1     {v5.4s}, [x0], #16
    fneg    v0.4s, v5.4s         // -x
    EXP_PS  v0                   // e^-x
    fadd    v0.4s, v0.4s, v25.4s // 1 + e^-x
    fdiv    v0.4s, v5.4s, v0.4s  // x / (1 + e^-x)
    st1     {v0.4s}, [x1], #16
    subs    x3, x3, #1
    b.ne    1b
2:
    and     x2, x2, #3           // scalar tail
    cbz     x2, 4f
3:
    ldr     s5, [x0], #4
    fneg    s0, s5
    EXP_SS  s0
    fadd    s0, s0, s25
    fdiv    s0, s5, s0
    str     s0, [x1], #4
    subs    x2, x2, #1
    b.ne    3b
4:
    ret
endfunc
