/*
 * peregrine - bf16 GEMM microkernel, AArch64 + FEAT_BF16 (BFMMLA).
 *
 *   void pg_bf16gemm_ukernel_8x8_bfmmla(size_t kg, const pg_bf16 *a,
 *                                       const pg_bf16 *b, float *c, size_t ldc)
 *     kg=x0 (K groups of 4)  a=x1 (packed)  b=x2 (packed)  c=x3  ldc=x4 (elem)
 *
 * Native bf16 matrix-multiply: each BFMMLA does a 2x2x4 bf16->f32 MAC (16 MACs,
 * 4x the FLOPs of an f32 FMLA), so an 8x8 tile is 16 BFMMLAs per depth-4 step.
 *
 * Packed layout (built by the driver, K zero-padded to a multiple of 4):
 *   a[kg]: 4 row-pairs, each 8 bf16 = [r0k0..r0k3, r1k0..r1k3]  (BFMMLA Vn 2x4)
 *   b[kg]: 4 col-pairs, each 8 bf16 = [k0c0,k0c1, k1c0,k1c1, ...] (BFMMLA Vm 4x2)
 * Accumulator v[16 + rp*4 + cp].4s holds the 2x2 block for rows (2rp,2rp+1) x
 * cols (2cp,2cp+1) as [r0c0, r0c1, r1c0, r1c1]. zip1/zip2 .2d reassemble whole
 * rows for the store.
 */
#include "ext/arm/asm.S"

.arch armv8.2-a+bf16

function bf16gemm_ukernel_8x8_bfmmla, export=1
    movi    v16.4s, #0
    movi    v17.4s, #0
    movi    v18.4s, #0
    movi    v19.4s, #0
    movi    v20.4s, #0
    movi    v21.4s, #0
    movi    v22.4s, #0
    movi    v23.4s, #0
    movi    v24.4s, #0
    movi    v25.4s, #0
    movi    v26.4s, #0
    movi    v27.4s, #0
    movi    v28.4s, #0
    movi    v29.4s, #0
    movi    v30.4s, #0
    movi    v31.4s, #0

    cbz     x0, 2f
1:
    ld1     {v0.8h, v1.8h, v2.8h, v3.8h}, [x1], #64   // a: row-pairs 0..3
    ld1     {v4.8h, v5.8h, v6.8h, v7.8h}, [x2], #64   // b: col-pairs 0..3
    bfmmla  v16.4s, v0.8h, v4.8h
    bfmmla  v17.4s, v0.8h, v5.8h
    bfmmla  v18.4s, v0.8h, v6.8h
    bfmmla  v19.4s, v0.8h, v7.8h
    bfmmla  v20.4s, v1.8h, v4.8h
    bfmmla  v21.4s, v1.8h, v5.8h
    bfmmla  v22.4s, v1.8h, v6.8h
    bfmmla  v23.4s, v1.8h, v7.8h
    bfmmla  v24.4s, v2.8h, v4.8h
    bfmmla  v25.4s, v2.8h, v5.8h
    bfmmla  v26.4s, v2.8h, v6.8h
    bfmmla  v27.4s, v2.8h, v7.8h
    bfmmla  v28.4s, v3.8h, v4.8h
    bfmmla  v29.4s, v3.8h, v5.8h
    bfmmla  v30.4s, v3.8h, v6.8h
    bfmmla  v31.4s, v3.8h, v7.8h
    subs    x0, x0, #1
    b.ne    1b
2:
    lsl     x4, x4, #2               // ldc: elements -> bytes
    mov     x5, x3                   // walking C row pointer

    // row-pair 0 -> rows 0,1  (v16,v17,v18,v19)
    zip1    v0.2d, v16.2d, v17.2d    // row0 cols 0-3
    zip1    v1.2d, v18.2d, v19.2d    // row0 cols 4-7
    zip2    v2.2d, v16.2d, v17.2d    // row1 cols 0-3
    zip2    v3.2d, v18.2d, v19.2d    // row1 cols 4-7
    st1     {v0.4s, v1.4s}, [x5], x4
    st1     {v2.4s, v3.4s}, [x5], x4
    // row-pair 1 -> rows 2,3  (v20,v21,v22,v23)
    zip1    v0.2d, v20.2d, v21.2d
    zip1    v1.2d, v22.2d, v23.2d
    zip2    v2.2d, v20.2d, v21.2d
    zip2    v3.2d, v22.2d, v23.2d
    st1     {v0.4s, v1.4s}, [x5], x4
    st1     {v2.4s, v3.4s}, [x5], x4
    // row-pair 2 -> rows 4,5  (v24,v25,v26,v27)
    zip1    v0.2d, v24.2d, v25.2d
    zip1    v1.2d, v26.2d, v27.2d
    zip2    v2.2d, v24.2d, v25.2d
    zip2    v3.2d, v26.2d, v27.2d
    st1     {v0.4s, v1.4s}, [x5], x4
    st1     {v2.4s, v3.4s}, [x5], x4
    // row-pair 3 -> rows 6,7  (v28,v29,v30,v31)
    zip1    v0.2d, v28.2d, v29.2d
    zip1    v1.2d, v30.2d, v31.2d
    zip2    v2.2d, v28.2d, v29.2d
    zip2    v3.2d, v30.2d, v31.2d
    st1     {v0.4s, v1.4s}, [x5], x4
    st1     {v2.4s, v3.4s}, [x5], x4
    ret
endfunc
