/*
 * peregrine - f32 exp, AArch64 NEON.
 *
 *   void pg_exp_f32_neon(const float *in, float *out, size_t n)
 *     x0 = in, x1 = out, x2 = n
 *
 * The math (cephes range reduction + degree-6 minimax, validated to ~1e-7 rel
 * error) lives in the shared EXP_PS / EXP_SS macros so exp-based kernels (silu,
 * gelu, softmax) reuse the exact same sequence.
 */
#include "tensor/kernels/simd/exp_neon.h"

function exp_f32_neon, export=1
    EXP_LOAD_CONSTS
    lsr     x3, x2, #2            // n / 4
    cbz     x3, 2f
1:
    ld1     {v0.4s}, [x0], #16
    EXP_PS  v0
    st1     {v0.4s}, [x1], #16
    subs    x3, x3, #1
    b.ne    1b
2:
    and     x2, x2, #3            // scalar tail
    cbz     x2, 4f
3:
    ldr     s0, [x0], #4
    EXP_SS  s0
    str     s0, [x1], #4
    subs    x2, x2, #1
    b.ne    3b
4:
    ret
endfunc
