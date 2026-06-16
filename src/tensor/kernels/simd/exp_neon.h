/*
 * peregrine - shared NEON exp building block (cephes range reduction + minimax).
 *
 * Reused by the exp kernel and by exp-based kernels (silu, gelu, softmax, ...).
 * Contract:
 *   - call EXP_LOAD_CONSTS once before the loop (clobbers x4); it fills the
 *     constant registers v16..v28.
 *   - EXP_PS r computes exp(\r) in place on vector \r, clobbering v1,v2,v3,v4
 *     and reading v16..v28. \r must not be one of those.
 *   - EXP_SS s computes exp on scalar \s in place, clobbering s1..s4 and w5.
 * v25 stays loaded with 1.0 after EXP_LOAD_CONSTS, which callers reuse.
 *
 * Each translation unit that includes this gets its own (local) const table;
 * the data is tiny, and keeping it local avoids cross-object symbol clashes.
 */
#ifndef PEREGRINE_SIMD_EXP_NEON_H
#define PEREGRINE_SIMD_EXP_NEON_H

#include "ext/arm/asm.S"

.macro pg_movrel rd, sym
#ifdef __APPLE__
    adrp    \rd, \sym@PAGE
    add     \rd, \rd, \sym@PAGEOFF
#else
    adrp    \rd, \sym
    add     \rd, \rd, :lo12:\sym
#endif
.endm

const pg_exp_consts, align=4
    .float 1.44269504088896341    /* log2e */
    .float 0.693359375            /* ln2 hi (C1) */
    .float -2.12194440e-4         /* ln2 lo (C2) */
    .float 1.9875691500e-4        /* p0 */
    .float 1.3981999507e-3        /* p1 */
    .float 8.3334519073e-3        /* p2 */
    .float 4.1665795894e-2        /* p3 */
    .float 1.6666665459e-1        /* p4 */
    .float 5.0000001201e-1        /* p5 */
    .float 1.0                    /* one */
    .float 88.3762626647950       /* hi clamp */
    .float -87.3365478515625      /* lo clamp: keeps m >= -126 so 2^m stays a
                                     valid normal (a lower clamp lets m hit -128,
                                     and the exponent-field trick then yields -inf) */
endconst

.macro EXP_LOAD_CONSTS
    pg_movrel x4, pg_exp_consts
    ld1r    {v16.4s}, [x4], #4    // log2e
    ld1r    {v17.4s}, [x4], #4    // C1
    ld1r    {v18.4s}, [x4], #4    // C2
    ld1r    {v19.4s}, [x4], #4    // p0
    ld1r    {v20.4s}, [x4], #4    // p1
    ld1r    {v21.4s}, [x4], #4    // p2
    ld1r    {v22.4s}, [x4], #4    // p3
    ld1r    {v23.4s}, [x4], #4    // p4
    ld1r    {v24.4s}, [x4], #4    // p5
    ld1r    {v25.4s}, [x4], #4    // one
    ld1r    {v26.4s}, [x4], #4    // hi
    ld1r    {v27.4s}, [x4], #4    // lo
    movi    v28.4s, #127         // exponent bias
.endm

.macro EXP_PS r
    fmin    \r\().4s, \r\().4s, v26.4s
    fmax    \r\().4s, \r\().4s, v27.4s
    fmul    v1.4s, \r\().4s, v16.4s
    frintn  v1.4s, v1.4s
    fmls    \r\().4s, v1.4s, v17.4s
    fmls    \r\().4s, v1.4s, v18.4s
    fmul    v2.4s, \r\().4s, \r\().4s
    mov     v3.16b, v19.16b
    mov     v4.16b, v20.16b
    fmla    v4.4s, v3.4s, \r\().4s
    mov     v3.16b, v21.16b
    fmla    v3.4s, v4.4s, \r\().4s
    mov     v4.16b, v22.16b
    fmla    v4.4s, v3.4s, \r\().4s
    mov     v3.16b, v23.16b
    fmla    v3.4s, v4.4s, \r\().4s
    mov     v4.16b, v24.16b
    fmla    v4.4s, v3.4s, \r\().4s
    fmla    \r\().4s, v4.4s, v2.4s
    fadd    \r\().4s, \r\().4s, v25.4s
    fcvtns  v1.4s, v1.4s
    add     v1.4s, v1.4s, v28.4s
    shl     v1.4s, v1.4s, #23
    fmul    \r\().4s, \r\().4s, v1.4s
.endm

.macro EXP_SS s
    fmin    \s, \s, s26
    fmax    \s, \s, s27
    fmul    s1, \s, s16
    frintn  s1, s1
    fmsub   \s, s1, s17, \s
    fmsub   \s, s1, s18, \s
    fmul    s2, \s, \s
    fmov    s3, s19
    fmadd   s4, s3, \s, s20
    fmadd   s3, s4, \s, s21
    fmadd   s4, s3, \s, s22
    fmadd   s3, s4, \s, s23
    fmadd   s4, s3, \s, s24
    fmadd   \s, s4, s2, \s
    fadd    \s, \s, s25
    fcvtns  w5, s1
    add     w5, w5, #127
    lsl     w5, w5, #23
    fmov    s1, w5
    fmul    \s, \s, s1
.endm

#endif /* PEREGRINE_SIMD_EXP_NEON_H */
