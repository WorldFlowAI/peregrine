; peregrine - f32 elementwise add, x86-64 AVX2:  out[i] = a[i] + b[i]
;   rdi = a, rsi = b, rdx = out, rcx = n
%include "config.asm"
%include "ext/x86/x86inc.asm"

SECTION .text

INIT_YMM avx2
cglobal add_f32, 4, 5, 4, a, b, out, n
    mov         r4, nq
    shr         r4, 5                   ; n / 32
    jz          .tail
.loop:
    vmovups     m0, [aq]
    vaddps      m0, m0, [bq]
    vmovups     [outq], m0
    vmovups     m1, [aq + 32]
    vaddps      m1, m1, [bq + 32]
    vmovups     [outq + 32], m1
    vmovups     m2, [aq + 64]
    vaddps      m2, m2, [bq + 64]
    vmovups     [outq + 64], m2
    vmovups     m3, [aq + 96]
    vaddps      m3, m3, [bq + 96]
    vmovups     [outq + 96], m3
    add         aq, 128
    add         bq, 128
    add         outq, 128
    dec         r4
    jnz         .loop
.tail:
    and         nq, 31                  ; tail: n % 32
    jz          .done
.t1:
    vmovss      xm0, [aq]
    vaddss      xm0, xm0, [bq]
    vmovss      [outq], xm0
    add         aq, 4
    add         bq, 4
    add         outq, 4
    dec         nq
    jnz         .t1
.done:
    RET
