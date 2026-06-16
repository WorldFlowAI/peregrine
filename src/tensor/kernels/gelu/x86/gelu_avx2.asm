; peregrine - f32 GELU (tanh approx), x86-64 AVX2+FMA.
;   rdi = in, rsi = out, rdx = n
;
; gelu(x) = x / (1 + e^-w),  w = 2K(x + 0.044715 x^3),  K = sqrt(2/pi).
; Reuses the shared exp macro (clobbers m1..m3); x in m4, tmp in m5.

%include "config.asm"
%include "ext/x86/x86inc.asm"

INIT_YMM avx2
%include "tensor/kernels/simd/exp_avx2.inc"

SECTION_RODATA 32
g_044715: times 8 dd 0.044715
g_neg2k:  times 8 dd -1.5957691216057308   ; -2*sqrt(2/pi)

SECTION .text

cglobal gelu_f32, 3, 4, 6, in, out, n
    mov         r3, nq
    shr         r3, 3
    jz          .tail
.loop:
    vmovups     m4, [inq]              ; x
    vmulps      m5, m4, m4
    vmulps      m5, m5, m4             ; x^3
    vmulps      m5, m5, [g_044715]     ; 0.044715 x^3
    vaddps      m5, m5, m4             ; x + 0.044715 x^3
    vmulps      m0, m5, [g_neg2k]      ; -w
    EXP_PS_AVX2 m0                      ; e^-w
    vaddps      m0, m0, [c_one]        ; 1 + e^-w
    vdivps      m0, m4, m0             ; gelu
    vmovups     [outq], m0
    add         inq, 32
    add         outq, 32
    dec         r3
    jnz         .loop
.tail:
    and         nq, 7
    jz          .done
.t1:
    vmovss      xm4, [inq]
    vmulss      xm5, xm4, xm4
    vmulss      xm5, xm5, xm4
    vmulss      xm5, xm5, [g_044715]
    vaddss      xm5, xm5, xm4
    vmulss      xm0, xm5, [g_neg2k]
    EXP_SS_AVX2 xm0
    vaddss      xm0, xm0, [c_one]
    vdivss      xm0, xm4, xm0
    vmovss      [outq], xm0
    add         inq, 4
    add         outq, 4
    dec         nq
    jnz         .t1
.done:
    RET
