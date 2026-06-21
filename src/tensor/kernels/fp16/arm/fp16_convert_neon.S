/*
 * peregrine - fp16/f32 array conversion, AArch64 + FEAT_FP16.
 */
#include "ext/arm/asm.S"

.arch armv8.2-a+fp16

function fp16_to_f32_array_neon, export=1
    lsr     x3, x2, #3
    cbz     x3, 2f
1:
    ld1     {v0.8h}, [x1], #16
    fcvtl   v1.4s, v0.4h
    fcvtl2  v2.4s, v0.8h
    st1     {v1.4s, v2.4s}, [x0], #32
    subs    x3, x3, #1
    b.ne    1b
2:
    and     x2, x2, #7
    cbz     x2, 4f
3:
    ldr     h0, [x1], #2
    fcvt    s0, h0
    str     s0, [x0], #4
    subs    x2, x2, #1
    b.ne    3b
4:
    ret
endfunc

function f32_to_fp16_array_neon, export=1
    lsr     x3, x2, #3
    cbz     x3, 2f
1:
    ld1     {v0.4s, v1.4s}, [x1], #32
    fcvtn   v2.4h, v0.4s
    fcvtn2  v2.8h, v1.4s
    st1     {v2.8h}, [x0], #16
    subs    x3, x3, #1
    b.ne    1b
2:
    and     x2, x2, #7
    cbz     x2, 4f
3:
    ldr     s0, [x1], #4
    fcvt    h0, s0
    str     h0, [x0], #2
    subs    x2, x2, #1
    b.ne    3b
4:
    ret
endfunc
