/*
 * peregrine - f32 max, AArch64 NEON:  s0 = max_i x[i]   (n >= 1)
 *   x0 = x, x1 = n
 *
 * Accumulators are seeded with x[0], so the running max is always valid; the
 * main loop re-reads from x[0] (idempotent for max).
 */
#include "ext/arm/asm.S"

function max_f32_neon, export=1
    ldr     s0, [x0]              // seed scalar/vector max with x[0]
    dup     v0.4s, v0.s[0]
    mov     v1.16b, v0.16b
    mov     v2.16b, v0.16b
    mov     v3.16b, v0.16b
    lsr     x2, x1, #4            // n / 16
    cbz     x2, 2f
1:
    ld1     {v16.4s, v17.4s, v18.4s, v19.4s}, [x0], #64
    fmax    v0.4s, v0.4s, v16.4s
    fmax    v1.4s, v1.4s, v17.4s
    fmax    v2.4s, v2.4s, v18.4s
    fmax    v3.4s, v3.4s, v19.4s
    subs    x2, x2, #1
    b.ne    1b
2:
    fmax    v0.4s, v0.4s, v1.4s
    fmax    v2.4s, v2.4s, v3.4s
    fmax    v0.4s, v0.4s, v2.4s
    fmaxv   s0, v0.4s             // horizontal max of the 4 lanes
    and     x1, x1, #15           // tail: n % 16
    cbz     x1, 4f
3:
    ldr     s16, [x0], #4
    fmax    s0, s0, s16
    subs    x1, x1, #1
    b.ne    3b
4:
    ret
endfunc
