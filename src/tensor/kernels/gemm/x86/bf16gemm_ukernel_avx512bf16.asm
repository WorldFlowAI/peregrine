; peregrine - bf16 GEMM microkernel, x86-64 AVX512-BF16.
;
;   void pg_bf16gemm_ukernel_6x16_avx512bf16(size_t kg, const pg_bf16 *a,
;                                            const pg_bf16 *b, float *c,
;                                            size_t ldc)
;     kg=rdi  a=rsi (packed)  b=rdx (packed)  c=rcx  ldc=r8 (elements)
;
; Computes a 6x16 block of C using VDPBF16PS. One K-group holds two bf16 depth
; values: A stores one dword pair per row, B stores one dword pair per output
; column. Accumulator zmm[4+r] holds row r, cols 0..15 as f32.

%include "config.asm"
%include "ext/x86/x86inc.asm"

SECTION .text

%assign cpuflags_avx512bf16 (1<<28) | cpuflags_avx512
INIT_ZMM avx512bf16
cglobal bf16gemm_ukernel_6x16, 5, 5, 10, kg, a, b, c, ldc
    vxorps      m4, m4, m4
    vxorps      m5, m5, m5
    vxorps      m6, m6, m6
    vxorps      m7, m7, m7
    vxorps      m8, m8, m8
    vxorps      m9, m9, m9

    test        kgq, kgq
    jz          .store
.loop:
    vmovups     m0, [bq]          ; 16 dword lanes: (b[k0,col], b[k1,col])
    vbroadcastss m1, [aq]         ; row 0: (a[k0], a[k1]) in every dword lane
    vdpbf16ps   m4, m1, m0
    vbroadcastss m1, [aq + 4]     ; row 1
    vdpbf16ps   m5, m1, m0
    vbroadcastss m1, [aq + 8]     ; row 2
    vdpbf16ps   m6, m1, m0
    vbroadcastss m1, [aq + 12]    ; row 3
    vdpbf16ps   m7, m1, m0
    vbroadcastss m1, [aq + 16]    ; row 4
    vdpbf16ps   m8, m1, m0
    vbroadcastss m1, [aq + 20]    ; row 5
    vdpbf16ps   m9, m1, m0
    add         aq, 24            ; 6 rows * 2 bf16
    add         bq, 64            ; 16 cols * 2 bf16
    dec         kgq
    jnz         .loop
.store:
    shl         ldcq, 2           ; ldc: elements -> bytes
    vmovups     [cq], m4
    add         cq, ldcq
    vmovups     [cq], m5
    add         cq, ldcq
    vmovups     [cq], m6
    add         cq, ldcq
    vmovups     [cq], m7
    add         cq, ldcq
    vmovups     [cq], m8
    add         cq, ldcq
    vmovups     [cq], m9
    RET
