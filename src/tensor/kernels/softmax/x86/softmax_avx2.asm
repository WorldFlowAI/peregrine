; peregrine - f32 softmax over n elements, x86-64 AVX2+FMA (fused, stable).
;   rdi = in, rsi = out, rdx = n
;
; Pass 1: max.  Pass 2: out = e^(in-max), sum.  Pass 3: out *= 1/sum.
; Reuses the shared exp macro (clobbers m1..m3). m4 = max then scale broadcast;
; m5 = sum accumulator; r5/r4 walk in/out so each pass is independent.

%include "config.asm"
%include "ext/x86/x86inc.asm"

INIT_YMM avx2
%include "tensor/kernels/simd/exp_avx2.inc"

SECTION .text

cglobal softmax_f32, 3, 6, 6, in, out, n
    test        nq, nq
    jz          .done

    ; ---- pass 1: max ----
    mov         r5, inq
    vbroadcastss m4, [r5]
    mov         r3, nq
    shr         r3, 3
    jz          .p1red
.p1:
    vmaxps      m4, m4, [r5]
    add         r5, 32
    dec         r3
    jnz         .p1
.p1red:
    vextractf128 xm0, m4, 1
    vmaxps      xm4, xm4, xm0
    vshufps     xm0, xm4, xm4, 0xb1
    vmaxps      xm4, xm4, xm0
    vshufps     xm0, xm4, xm4, 0x4e
    vmaxps      xm4, xm4, xm0          ; xm4[0] = max of the vector part
    mov         r3, nq
    and         r3, 7
    jz          .p1bc
.p1t:
    vmaxss      xm4, xm4, [r5]
    add         r5, 4
    dec         r3
    jnz         .p1t
.p1bc:
    vbroadcastss m4, xm4               ; broadcast max

    ; ---- pass 2: out = e^(in-max), sum ----
    mov         r5, inq
    mov         r4, outq
    vxorps      m5, m5, m5
    mov         r3, nq
    shr         r3, 3
    jz          .p2red
.p2:
    vmovups     m0, [r5]
    vsubps      m0, m0, m4
    EXP_PS_AVX2 m0
    vmovups     [r4], m0
    vaddps      m5, m5, m0
    add         r5, 32
    add         r4, 32
    dec         r3
    jnz         .p2
.p2red:
    vextractf128 xm0, m5, 1
    vaddps      xm5, xm5, xm0
    vhaddps     xm5, xm5, xm5
    vhaddps     xm5, xm5, xm5          ; xm5[0] = vector sum
    mov         r3, nq
    and         r3, 7
    jz          .scale
.p2t:
    vmovss      xm0, [r5]
    vsubss      xm0, xm0, xm4
    EXP_SS_AVX2 xm0
    vmovss      [r4], xm0
    vaddss      xm5, xm5, xm0
    add         r5, 4
    add         r4, 4
    dec         r3
    jnz         .p2t
.scale:
    mov         eax, 0x3f800000        ; 1.0f
    vmovd       xm0, eax
    vdivss      xm0, xm0, xm5          ; 1 / sum
    vbroadcastss m4, xm0               ; broadcast scale

    ; ---- pass 3: out *= 1/sum ----
    mov         r3, nq
    shr         r3, 3
    jz          .p3tail
.p3:
    vmovups     m0, [outq]
    vmulps      m0, m0, m4
    vmovups     [outq], m0
    add         outq, 32
    dec         r3
    jnz         .p3
.p3tail:
    mov         r3, nq
    and         r3, 7
    jz          .done
.p3t:
    vmovss      xm0, [outq]
    vmulss      xm0, xm0, xm4
    vmovss      [outq], xm0
    add         outq, 4
    dec         r3
    jnz         .p3t
.done:
    RET
