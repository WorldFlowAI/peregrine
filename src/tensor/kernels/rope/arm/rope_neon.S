/*
 * peregrine - f32 RoPE rotation, AArch64 NEON (pure mul/add, no transcendentals).
 *
 *   void pg_rope_f32_neon(float *out, const float *in, const float *cos,
 *                         const float *sin, size_t n_tokens, size_t n_heads,
 *                         size_t head_dim, PgRopeMode mode)
 *     x0=out x1=in x2=cos x3=sin x4=n_tokens x5=n_heads x6=head_dim x7=mode
 *
 * Per pair (a,b):  out0 = a*cos - b*sin ;  out1 = a*sin + b*cos.
 * NeoX:        a = x[i], b = x[i+half]  (two contiguous halves).
 * Interleaved: a = x[2i], b = x[2i+1]   (ld2/st2 de/re-interleave).
 * cos/sin are [n_tokens, half] (one per pair, per token). out may alias in.
 *
 * Uses only caller-saved regs (x9-x16, v0-v5), so it is safe under the checkasm
 * register-clobber contract even though rope's 8-arg signature bypasses the
 * clobber trampoline.
 */
#include "ext/arm/asm.S"

function rope_f32_neon, export=1
    lsr     x9, x6, #1               // half = head_dim / 2
    cbz     x4, 9f                   // n_tokens == 0
    cbz     x9, 9f                   // head_dim == 0
    cmp     x7, #1                   // mode: 1 = NeoX
    b.eq    1f

    /* ===== interleaved: a=even, b=odd ===== */
    mov     x10, x4
11:
    cbz     x10, 9f
    mov     x11, x5
12:
    mov     x12, x2                  // cos (this token)
    mov     x13, x3                  // sin
    lsr     x16, x9, #2              // half / 4
    cbz     x16, 14f
13:
    ld2     {v0.4s, v1.4s}, [x1], #32   // v0=a, v1=b
    ld1     {v2.4s}, [x12], #16        // cos
    ld1     {v3.4s}, [x13], #16        // sin
    fmul    v4.4s, v0.4s, v2.4s
    fmls    v4.4s, v1.4s, v3.4s        // a*cos - b*sin
    fmul    v5.4s, v0.4s, v3.4s
    fmla    v5.4s, v1.4s, v2.4s        // a*sin + b*cos
    st2     {v4.4s, v5.4s}, [x0], #32
    subs    x16, x16, #1
    b.ne    13b
14:
    and     x16, x9, #3
    cbz     x16, 16f
15:
    ldr     s0, [x1], #4
    ldr     s1, [x1], #4
    ldr     s2, [x12], #4
    ldr     s3, [x13], #4
    fmul    s4, s0, s2
    fmsub   s4, s1, s3, s4
    fmul    s5, s0, s3
    fmadd   s5, s1, s2, s5
    str     s4, [x0], #4
    str     s5, [x0], #4
    subs    x16, x16, #1
    b.ne    15b
16:
    subs    x11, x11, #1
    b.ne    12b
    add     x2, x2, x9, lsl #2         // cos/sin advance one token
    add     x3, x3, x9, lsl #2
    sub     x10, x10, #1
    b       11b

    /* ===== NeoX: a=first half, b=second half ===== */
1:
    mov     x10, x4
20:
    cbz     x10, 9f
    mov     x11, x5
21:
    mov     x12, x2
    mov     x13, x3
    add     x14, x1, x9, lsl #2        // b in-ptr  = in + half
    add     x15, x0, x9, lsl #2        // b out-ptr = out + half
    lsr     x16, x9, #2
    cbz     x16, 23f
22:
    ld1     {v0.4s}, [x1], #16         // a
    ld1     {v1.4s}, [x14], #16        // b
    ld1     {v2.4s}, [x12], #16        // cos
    ld1     {v3.4s}, [x13], #16        // sin
    fmul    v4.4s, v0.4s, v2.4s
    fmls    v4.4s, v1.4s, v3.4s
    fmul    v5.4s, v0.4s, v3.4s
    fmla    v5.4s, v1.4s, v2.4s
    st1     {v4.4s}, [x0], #16
    st1     {v5.4s}, [x15], #16
    subs    x16, x16, #1
    b.ne    22b
23:
    and     x16, x9, #3
    cbz     x16, 24f
25:
    ldr     s0, [x1], #4
    ldr     s1, [x14], #4
    ldr     s2, [x12], #4
    ldr     s3, [x13], #4
    fmul    s4, s0, s2
    fmsub   s4, s1, s3, s4
    fmul    s5, s0, s3
    fmadd   s5, s1, s2, s5
    str     s4, [x0], #4
    str     s5, [x15], #4
    subs    x16, x16, #1
    b.ne    25b
24:
    add     x1, x1, x9, lsl #2         // skip the b half -> next head
    add     x0, x0, x9, lsl #2
    subs    x11, x11, #1
    b.ne    21b
    add     x2, x2, x9, lsl #2
    add     x3, x3, x9, lsl #2
    sub     x10, x10, #1
    b       20b
9:
    ret
endfunc
