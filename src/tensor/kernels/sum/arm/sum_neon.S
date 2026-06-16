/*
 * peregrine - f32 sum, AArch64 NEON:  s0 = sum_i x[i]
 *   x0 = x, x1 = n
 */
#include "ext/arm/asm.S"

function sum_f32_neon, export=1
    movi    v0.16b, #0
    movi    v1.16b, #0
    movi    v2.16b, #0
    movi    v3.16b, #0
    lsr     x2, x1, #4            // n / 16
    cbz     x2, 2f
1:
    ld1     {v16.4s, v17.4s, v18.4s, v19.4s}, [x0], #64
    fadd    v0.4s, v0.4s, v16.4s
    fadd    v1.4s, v1.4s, v17.4s
    fadd    v2.4s, v2.4s, v18.4s
    fadd    v3.4s, v3.4s, v19.4s
    subs    x2, x2, #1
    b.ne    1b
2:
    fadd    v0.4s, v0.4s, v1.4s
    fadd    v2.4s, v2.4s, v3.4s
    fadd    v0.4s, v0.4s, v2.4s
    faddp   v0.4s, v0.4s, v0.4s
    faddp   v0.4s, v0.4s, v0.4s   // s0 = partial sum
    and     x1, x1, #15           // tail: n % 16
    cbz     x1, 4f
3:
    ldr     s16, [x0], #4
    fadd    s0, s0, s16
    subs    x1, x1, #1
    b.ne    3b
4:
    ret
endfunc
