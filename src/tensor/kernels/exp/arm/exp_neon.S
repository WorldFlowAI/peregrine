/*
 * peregrine - f32 exp, AArch64 NEON (cephes range reduction + minimax poly).
 *
 *   void pg_exp_f32_neon(const float *in, float *out, size_t n)
 *     x0 = in, x1 = out, x2 = n
 *
 * e^x = 2^m * e^r,  m = round(x*log2e),  r = x - m*ln2 (Cody-Waite split),
 * e^r via a degree-6 polynomial, 2^m built straight into the exponent field.
 * Constants are loaded once into v16..v28; the loop body matches the C poly in
 * exp.h that was validated to ~1e-7 relative error.
 */
#include "ext/arm/asm.S"

.macro movrel rd, sym
#ifdef __APPLE__
    adrp    \rd, \sym@PAGE
    add     \rd, \rd, \sym@PAGEOFF
#else
    adrp    \rd, \sym
    add     \rd, \rd, :lo12:\sym
#endif
.endm

const exp_neon_consts, align=4
    .float 1.44269504088896341    /* 0  log2e         */
    .float 0.693359375            /* 1  ln2 hi (C1)   */
    .float -2.12194440e-4         /* 2  ln2 lo (C2)   */
    .float 1.9875691500e-4        /* 3  p0            */
    .float 1.3981999507e-3        /* 4  p1            */
    .float 8.3334519073e-3        /* 5  p2            */
    .float 4.1665795894e-2        /* 6  p3            */
    .float 1.6666665459e-1        /* 7  p4            */
    .float 5.0000001201e-1        /* 8  p5            */
    .float 1.0                    /* 9  one           */
    .float 88.3762626647950       /* 10 hi clamp      */
    .float -88.3762626647950      /* 11 lo clamp      */
endconst

function exp_f32_neon, export=1
    movrel  x4, exp_neon_consts
    ld1r    {v16.4s}, [x4], #4         // log2e
    ld1r    {v17.4s}, [x4], #4         // C1
    ld1r    {v18.4s}, [x4], #4         // C2
    ld1r    {v19.4s}, [x4], #4         // p0
    ld1r    {v20.4s}, [x4], #4         // p1
    ld1r    {v21.4s}, [x4], #4         // p2
    ld1r    {v22.4s}, [x4], #4         // p3
    ld1r    {v23.4s}, [x4], #4         // p4
    ld1r    {v24.4s}, [x4], #4         // p5
    ld1r    {v25.4s}, [x4], #4         // one
    ld1r    {v26.4s}, [x4], #4         // hi
    ld1r    {v27.4s}, [x4], #4         // lo
    movi    v28.4s, #127              // exponent bias

    lsr     x3, x2, #2                // n / 4
    cbz     x3, 2f
1:
    ld1     {v0.4s}, [x0], #16
    fmin    v0.4s, v0.4s, v26.4s     // clamp hi
    fmax    v0.4s, v0.4s, v27.4s     // clamp lo
    fmul    v1.4s, v0.4s, v16.4s     // x*log2e
    frintn  v1.4s, v1.4s             // m = round()
    fmls    v0.4s, v1.4s, v17.4s     // r = x - m*C1
    fmls    v0.4s, v1.4s, v18.4s     // r -= m*C2
    fmul    v2.4s, v0.4s, v0.4s      // z = r*r
    mov     v3.16b, v19.16b          // y = p0
    mov     v4.16b, v20.16b
    fmla    v4.4s, v3.4s, v0.4s      // p1 + p0*r
    mov     v3.16b, v21.16b
    fmla    v3.4s, v4.4s, v0.4s
    mov     v4.16b, v22.16b
    fmla    v4.4s, v3.4s, v0.4s
    mov     v3.16b, v23.16b
    fmla    v3.4s, v4.4s, v0.4s
    mov     v4.16b, v24.16b
    fmla    v4.4s, v3.4s, v0.4s      // y5
    fmla    v0.4s, v4.4s, v2.4s      // r + y5*z
    fadd    v0.4s, v0.4s, v25.4s     // + 1  -> e^r
    fcvtns  v1.4s, v1.4s             // m -> int
    add     v1.4s, v1.4s, v28.4s     // + 127
    shl     v1.4s, v1.4s, #23        // exponent field -> 2^m as float bits
    fmul    v0.4s, v0.4s, v1.4s      // e^r * 2^m
    st1     {v0.4s}, [x1], #16
    subs    x3, x3, #1
    b.ne    1b
2:
    and     x2, x2, #3               // scalar tail
    cbz     x2, 4f
3:
    ldr     s0, [x0], #4
    fmin    s0, s0, s26
    fmax    s0, s0, s27
    fmul    s1, s0, s16
    frintn  s1, s1
    fmsub   s0, s1, s17, s0
    fmsub   s0, s1, s18, s0
    fmul    s2, s0, s0
    fmov    s3, s19
    fmadd   s4, s3, s0, s20
    fmadd   s3, s4, s0, s21
    fmadd   s4, s3, s0, s22
    fmadd   s3, s4, s0, s23
    fmadd   s4, s3, s0, s24
    fmadd   s0, s4, s2, s0
    fadd    s0, s0, s25
    fcvtns  w5, s1
    add     w5, w5, #127
    lsl     w5, w5, #23
    fmov    s1, w5
    fmul    s0, s0, s1
    str     s0, [x1], #4
    subs    x2, x2, #1
    b.ne    3b
4:
    ret
endfunc
