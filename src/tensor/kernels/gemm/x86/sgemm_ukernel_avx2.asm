; peregrine - f32 GEMM register-blocked microkernel, x86-64 AVX2+FMA.
;
;   void pg_sgemm_ukernel_6x16_avx2(size_t kc, const float *a, const float *b,
;                                   float *c, size_t ldc)
;     kc=rdi  a=rsi (packed)  b=rdx (packed)  c=rcx  ldc=r8 (elements)
;
; Computes a 6x16 block of C as the rank-1 outer product accumulated over the
; packed depth kc:  C[6x16] = sum_{k<kc} a[k][0..5] (x) b[k][0..15].
; Packed panels (built by the driver) are read contiguously:
;   a[k*6 + r]  = row r's element at depth k   (6 floats / step)
;   b[k*16 + col] = col's element at depth k    (16 floats / step)
; C is overwritten (one call spans all of K).
;
; 12 accumulators (ymm4..ymm15) hold the tile: row r -> ymm[4+2r] (cols 0-7),
; ymm[5+2r] (cols 8-15). Per depth step: 2 b-loads + 6 a-broadcasts + 12 FMAs.
; All ymm are caller-saved on SysV; RET emits vzeroupper (INIT_YMM).

%include "config.asm"
%include "ext/x86/x86inc.asm"

SECTION .text

INIT_YMM avx2
cglobal sgemm_ukernel_6x16, 5, 5, 16, kc, a, b, c, ldc
    vxorps      m4,  m4,  m4
    vxorps      m5,  m5,  m5
    vxorps      m6,  m6,  m6
    vxorps      m7,  m7,  m7
    vxorps      m8,  m8,  m8
    vxorps      m9,  m9,  m9
    vxorps      m10, m10, m10
    vxorps      m11, m11, m11
    vxorps      m12, m12, m12
    vxorps      m13, m13, m13
    vxorps      m14, m14, m14
    vxorps      m15, m15, m15

    test        kcq, kcq
    jz          .store
.loop:
    vmovups     m0, [bq]              ; b cols 0-7
    vmovups     m1, [bq + 32]         ; b cols 8-15
    vbroadcastss m2, [aq]             ; row 0
    vfmadd231ps m4,  m0, m2
    vfmadd231ps m5,  m1, m2
    vbroadcastss m3, [aq + 4]         ; row 1
    vfmadd231ps m6,  m0, m3
    vfmadd231ps m7,  m1, m3
    vbroadcastss m2, [aq + 8]         ; row 2
    vfmadd231ps m8,  m0, m2
    vfmadd231ps m9,  m1, m2
    vbroadcastss m3, [aq + 12]        ; row 3
    vfmadd231ps m10, m0, m3
    vfmadd231ps m11, m1, m3
    vbroadcastss m2, [aq + 16]        ; row 4
    vfmadd231ps m12, m0, m2
    vfmadd231ps m13, m1, m2
    vbroadcastss m3, [aq + 20]        ; row 5
    vfmadd231ps m14, m0, m3
    vfmadd231ps m15, m1, m3
    add         aq, 24                ; 6 floats
    add         bq, 64                ; 16 floats
    dec         kcq
    jnz         .loop
.store:
    shl         ldcq, 2               ; ldc: elements -> bytes
    vmovups     [cq],      m4
    vmovups     [cq + 32], m5
    add         cq, ldcq
    vmovups     [cq],      m6
    vmovups     [cq + 32], m7
    add         cq, ldcq
    vmovups     [cq],      m8
    vmovups     [cq + 32], m9
    add         cq, ldcq
    vmovups     [cq],      m10
    vmovups     [cq + 32], m11
    add         cq, ldcq
    vmovups     [cq],      m12
    vmovups     [cq + 32], m13
    add         cq, ldcq
    vmovups     [cq],      m14
    vmovups     [cq + 32], m15
    RET
