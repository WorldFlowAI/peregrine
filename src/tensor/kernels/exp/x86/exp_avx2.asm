; peregrine - f32 exp, x86-64 AVX2+FMA.
;
;   void pg_exp_f32_avx2(const float *in, float *out, size_t n)
;     rdi = in, rsi = out, rdx = n
;
; The math (cephes range reduction + degree-6 minimax) lives in the shared
; EXP_PS_AVX2 / EXP_SS_AVX2 macros so exp-based kernels reuse it.

%include "config.asm"
%include "ext/x86/x86inc.asm"

INIT_YMM avx2
%include "tensor/kernels/simd/exp_avx2.inc"

SECTION .text

cglobal exp_f32, 3, 4, 4, in, out, n
    mov         r3, nq
    shr         r3, 3                   ; n / 8
    jz          .tail
.loop:
    vmovups     m0, [inq]
    EXP_PS_AVX2 m0
    vmovups     [outq], m0
    add         inq, 32
    add         outq, 32
    dec         r3
    jnz         .loop
.tail:
    and         nq, 7
    jz          .done
.t1:
    vmovss      xm0, [inq]
    EXP_SS_AVX2 xm0
    vmovss      [outq], xm0
    add         inq, 4
    add         outq, 4
    dec         nq
    jnz         .t1
.done:
    RET
