/*
 * peregrine - f32 axpy, AArch64 NEON:  y[i] += alpha * x[i]
 *
 *   void pg_axpy_f32_neon(float alpha, const float *x, float *y, size_t n)
 *     s0 = alpha, x0 = x, x1 = y, x2 = n
 *
 * 16 floats/iteration: load x and y, fmla y += alpha*x, store y back.
 */
#include "ext/arm/asm.S"

function axpy_f32_neon, export=1
    dup     v0.4s, v0.s[0]         // broadcast alpha into all lanes (keeps s0=alpha)
    lsr     x3, x2, #4             // x3 = n / 16
    cbz     x3, 2f
1:
    ld1     {v16.4s, v17.4s, v18.4s, v19.4s}, [x0], #64   // x
    ld1     {v20.4s, v21.4s, v22.4s, v23.4s}, [x1]        // y (peek, store later)
    fmla    v20.4s, v16.4s, v0.4s
    fmla    v21.4s, v17.4s, v0.4s
    fmla    v22.4s, v18.4s, v0.4s
    fmla    v23.4s, v19.4s, v0.4s
    st1     {v20.4s, v21.4s, v22.4s, v23.4s}, [x1], #64   // y back
    subs    x3, x3, #1
    b.ne    1b
2:
    and     x2, x2, #15           // scalar tail: n % 16
    cbz     x2, 4f
3:
    ldr     s16, [x0], #4
    ldr     s20, [x1]
    fmadd   s20, s16, s0, s20     // y = x*alpha + y
    str     s20, [x1], #4
    subs    x2, x2, #1
    b.ne    3b
4:
    ret
endfunc
