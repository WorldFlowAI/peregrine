/*
 * peregrine - Q8_0 GEMV 2-row microkernel, AArch64 NEON + FEAT_FP16.
 *
 *   void pg_q8_0_gemv_2x_neon(const void *row0, const void *row1,
 *                             const float *x, size_t K, float *y)
 *     x0=row0  x1=row1  x2=x  x3=K  x4=y
 *
 * Computes two packed Q8_0 rows against the same activation vector. The x
 * block is loaded once and reused for both rows.
 */
#include "ext/arm/asm.S"

.arch armv8.2-a+fp16

function q8_0_gemv_2x_neon, export=1
    movi    v0.16b, #0
    movi    v1.16b, #0
    movi    v2.16b, #0
    movi    v3.16b, #0
    movi    v4.16b, #0
    movi    v5.16b, #0
    movi    v6.16b, #0
    movi    v7.16b, #0

    lsr     x3, x3, #5
    cbz     x3, 2f
1:
    ld1     {v19.4s, v20.4s, v21.4s, v22.4s}, [x2], #64
    ld1     {v23.4s, v24.4s, v25.4s, v26.4s}, [x2], #64

    ldr     h16, [x0], #2
    fcvt    s16, h16
    ld1     {v17.16b, v18.16b}, [x0], #32
    sshll   v27.8h, v17.8b, #0
    sshll2  v28.8h, v17.16b, #0
    sshll   v29.8h, v18.8b, #0
    sshll2  v30.8h, v18.16b, #0

    sshll   v17.4s, v27.4h, #0
    sshll2  v27.4s, v27.8h, #0
    scvtf   v17.4s, v17.4s
    scvtf   v27.4s, v27.4s
    fmul    v17.4s, v17.4s, v16.s[0]
    fmul    v27.4s, v27.4s, v16.s[0]
    fmla    v0.4s, v17.4s, v19.4s
    fmla    v1.4s, v27.4s, v20.4s

    sshll   v17.4s, v28.4h, #0
    sshll2  v28.4s, v28.8h, #0
    scvtf   v17.4s, v17.4s
    scvtf   v28.4s, v28.4s
    fmul    v17.4s, v17.4s, v16.s[0]
    fmul    v28.4s, v28.4s, v16.s[0]
    fmla    v2.4s, v17.4s, v21.4s
    fmla    v3.4s, v28.4s, v22.4s

    sshll   v17.4s, v29.4h, #0
    sshll2  v29.4s, v29.8h, #0
    scvtf   v17.4s, v17.4s
    scvtf   v29.4s, v29.4s
    fmul    v17.4s, v17.4s, v16.s[0]
    fmul    v29.4s, v29.4s, v16.s[0]
    fmla    v0.4s, v17.4s, v23.4s
    fmla    v1.4s, v29.4s, v24.4s

    sshll   v17.4s, v30.4h, #0
    sshll2  v30.4s, v30.8h, #0
    scvtf   v17.4s, v17.4s
    scvtf   v30.4s, v30.4s
    fmul    v17.4s, v17.4s, v16.s[0]
    fmul    v30.4s, v30.4s, v16.s[0]
    fmla    v2.4s, v17.4s, v25.4s
    fmla    v3.4s, v30.4s, v26.4s

    ldr     h31, [x1], #2
    fcvt    s31, h31
    ld1     {v17.16b, v18.16b}, [x1], #32
    sshll   v27.8h, v17.8b, #0
    sshll2  v28.8h, v17.16b, #0
    sshll   v29.8h, v18.8b, #0
    sshll2  v30.8h, v18.16b, #0

    sshll   v17.4s, v27.4h, #0
    sshll2  v27.4s, v27.8h, #0
    scvtf   v17.4s, v17.4s
    scvtf   v27.4s, v27.4s
    fmul    v17.4s, v17.4s, v31.s[0]
    fmul    v27.4s, v27.4s, v31.s[0]
    fmla    v4.4s, v17.4s, v19.4s
    fmla    v5.4s, v27.4s, v20.4s

    sshll   v17.4s, v28.4h, #0
    sshll2  v28.4s, v28.8h, #0
    scvtf   v17.4s, v17.4s
    scvtf   v28.4s, v28.4s
    fmul    v17.4s, v17.4s, v31.s[0]
    fmul    v28.4s, v28.4s, v31.s[0]
    fmla    v6.4s, v17.4s, v21.4s
    fmla    v7.4s, v28.4s, v22.4s

    sshll   v17.4s, v29.4h, #0
    sshll2  v29.4s, v29.8h, #0
    scvtf   v17.4s, v17.4s
    scvtf   v29.4s, v29.4s
    fmul    v17.4s, v17.4s, v31.s[0]
    fmul    v29.4s, v29.4s, v31.s[0]
    fmla    v4.4s, v17.4s, v23.4s
    fmla    v5.4s, v29.4s, v24.4s

    sshll   v17.4s, v30.4h, #0
    sshll2  v30.4s, v30.8h, #0
    scvtf   v17.4s, v17.4s
    scvtf   v30.4s, v30.4s
    fmul    v17.4s, v17.4s, v31.s[0]
    fmul    v30.4s, v30.4s, v31.s[0]
    fmla    v6.4s, v17.4s, v25.4s
    fmla    v7.4s, v30.4s, v26.4s

    subs    x3, x3, #1
    b.ne    1b
2:
    fadd    v0.4s, v0.4s, v1.4s
    fadd    v2.4s, v2.4s, v3.4s
    fadd    v0.4s, v0.4s, v2.4s
    faddp   v0.4s, v0.4s, v0.4s
    faddp   v0.4s, v0.4s, v0.4s

    fadd    v4.4s, v4.4s, v5.4s
    fadd    v6.4s, v6.4s, v7.4s
    fadd    v4.4s, v4.4s, v6.4s
    faddp   v4.4s, v4.4s, v4.4s
    faddp   v4.4s, v4.4s, v4.4s

    st1     {v0.s}[0], [x4], #4
    st1     {v4.s}[0], [x4]
    ret
endfunc
