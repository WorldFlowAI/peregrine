/*
 * peregrine - f32 dot product, AArch64 NEON (hand-written, no intrinsics)
 *
 *   float pg_dot_f32_neon(const float *a, const float *b, size_t n)
 *     x0 = a, x1 = b, x2 = n   ->   s0
 *
 * Four independent accumulators hide the FMA latency; the main loop consumes
 * 16 floats/iteration, then a horizontal reduce and a scalar tail handle the
 * remainder. This is the shape every NEON kernel in peregrine follows.
 *
 * The `function ... export=1` / `endfunc` frame (symbol naming, Mach-O vs ELF
 * underscore, alignment, BTI markers) comes from dav1d's ext/arm/asm.S.
 */
#include "ext/arm/asm.S"

function dot_f32_neon, export=1
    movi    v0.16b, #0              // acc0
    movi    v1.16b, #0             // acc1
    movi    v2.16b, #0            // acc2
    movi    v3.16b, #0           // acc3

    lsr     x3, x2, #4            // x3 = n / 16  (vector iterations)
    cbz     x3, 2f
1:
    ld1     {v16.4s, v17.4s, v18.4s, v19.4s}, [x0], #64
    ld1     {v20.4s, v21.4s, v22.4s, v23.4s}, [x1], #64
    fmla    v0.4s, v16.4s, v20.4s
    fmla    v1.4s, v17.4s, v21.4s
    fmla    v2.4s, v18.4s, v22.4s
    fmla    v3.4s, v19.4s, v23.4s
    subs    x3, x3, #1
    b.ne    1b
2:
    fadd    v0.4s, v0.4s, v1.4s
    fadd    v2.4s, v2.4s, v3.4s
    fadd    v0.4s, v0.4s, v2.4s
    faddp   v0.4s, v0.4s, v0.4s    // pairwise reduce 4 -> 2
    faddp   v0.4s, v0.4s, v0.4s    // -> 1 ; sum now in s0

    and     x2, x2, #15            // scalar tail: n % 16
    cbz     x2, 4f
3:
    ldr     s16, [x0], #4
    ldr     s20, [x1], #4
    fmadd   s0, s16, s20, s0       // s0 += a*b
    subs    x2, x2, #1
    b.ne    3b
4:
    ret
endfunc
