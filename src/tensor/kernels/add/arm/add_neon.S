/*
 * peregrine - f32 elementwise add, AArch64 NEON:  out[i] = a[i] + b[i]
 *   x0 = a, x1 = b, x2 = out, x3 = n
 */
#include "ext/arm/asm.S"

function add_f32_neon, export=1
    lsr     x4, x3, #4            // n / 16
    cbz     x4, 2f
1:
    ld1     {v16.4s, v17.4s, v18.4s, v19.4s}, [x0], #64
    ld1     {v20.4s, v21.4s, v22.4s, v23.4s}, [x1], #64
    fadd    v16.4s, v16.4s, v20.4s
    fadd    v17.4s, v17.4s, v21.4s
    fadd    v18.4s, v18.4s, v22.4s
    fadd    v19.4s, v19.4s, v23.4s
    st1     {v16.4s, v17.4s, v18.4s, v19.4s}, [x2], #64
    subs    x4, x4, #1
    b.ne    1b
2:
    and     x3, x3, #15           // tail: n % 16
    cbz     x3, 4f
3:
    ldr     s16, [x0], #4
    ldr     s20, [x1], #4
    fadd    s16, s16, s20
    str     s16, [x2], #4
    subs    x3, x3, #1
    b.ne    3b
4:
    ret
endfunc
