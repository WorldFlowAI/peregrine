; peregrine - f32 SiLU, x86-64 AVX2+FMA:  out[i] = x / (1 + e^-x)
;   rdi = in, rsi = out, rdx = n
;
; Reuses the shared EXP_PS_AVX2 / EXP_SS_AVX2 macros (they clobber m1..m3);
; x is kept in m4, c_one comes from the included constant table.

%include "config.asm"
%include "ext/x86/x86inc.asm"

INIT_YMM avx2
%include "tensor/kernels/simd/exp_avx2.inc"

SECTION .text

cglobal silu_f32, 3, 4, 6, in, out, n
    mov         r3, nq
    shr         r3, 3                   ; n / 8
    jz          .tail
.loop:
    vmovups     m4, [inq]               ; x
    vxorps      m0, m0, m0
    vsubps      m0, m0, m4              ; -x
    EXP_PS_AVX2 m0                       ; e^-x
    vaddps      m0, m0, [c_one]         ; 1 + e^-x
    vdivps      m0, m4, m0              ; x / (1 + e^-x)
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
    vxorps      xm0, xm0, xm0
    vsubss      xm0, xm0, xm4
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
