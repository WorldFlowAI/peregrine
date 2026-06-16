/*
 * peregrine - f32 GELU (tanh approx), AArch64 NEON.
 *   x0 = in, x1 = out, x2 = n
 *
 * gelu(x) = 0.5x(1 + tanh(K(x + 0.044715 x^3)))  =  x / (1 + e^-w),
 *   w = 2K(x + 0.044715 x^3),  K = sqrt(2/pi).
 * So it is silu's shape on a transformed argument and reuses the exp macro.
 * x in v5, exp temps v1..v4 / v16..v28; gelu consts in v29 (0.044715), v30 (-2K).
 */
#include "tensor/kernels/simd/exp_neon.h"

const gelu_neon_consts, align=4
    .float 0.044715
    .float -1.5957691216057308   /* -2*sqrt(2/pi) */
endconst

function gelu_f32_neon, export=1
    EXP_LOAD_CONSTS
    pg_movrel x4, gelu_neon_consts
    ld1r    {v29.4s}, [x4], #4   // 0.044715
    ld1r    {v30.4s}, [x4], #4   // -2K
    lsr     x3, x2, #2
    cbz     x3, 2f
1:
    ld1     {v5.4s}, [x0], #16
    fmul    v6.4s, v5.4s, v5.4s
    fmul    v6.4s, v6.4s, v5.4s      // x^3
    mov     v7.16b, v5.16b           // x
    fmla    v7.4s, v6.4s, v29.4s     // x + 0.044715 x^3
    fmul    v0.4s, v7.4s, v30.4s     // -w = -2K(...)
    EXP_PS  v0                       // e^-w
    fadd    v0.4s, v0.4s, v25.4s     // 1 + e^-w
    fdiv    v0.4s, v5.4s, v0.4s      // x / (1 + e^-w)
    st1     {v0.4s}, [x1], #16
    subs    x3, x3, #1
    b.ne    1b
2:
    and     x2, x2, #3
    cbz     x2, 4f
3:
    ldr     s5, [x0], #4
    fmul    s6, s5, s5
    fmul    s6, s6, s5
    fmadd   s7, s6, s29, s5          // x + 0.044715 x^3
    fmul    s0, s7, s30              // -w
    EXP_SS  s0
    fadd    s0, s0, s25
    fdiv    s0, s5, s0
    str     s0, [x1], #4
    subs    x2, x2, #1
    b.ne    3b
4:
    ret
endfunc
