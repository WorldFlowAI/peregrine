; peregrine - f32 sum, x86-64 AVX2:  xmm0 = sum_i x[i]
;   rdi = x, rsi = n
%include "config.asm"
%include "ext/x86/x86inc.asm"

SECTION .text

INIT_YMM avx2
cglobal sum_f32, 2, 3, 4, x, n
    vxorps      m0, m0, m0
    vxorps      m1, m1, m1
    vxorps      m2, m2, m2
    vxorps      m3, m3, m3
    mov         r2, nq
    shr         r2, 5                   ; n / 32
    jz          .reduce
.loop:
    vaddps      m0, m0, [xq]
    vaddps      m1, m1, [xq + 32]
    vaddps      m2, m2, [xq + 64]
    vaddps      m3, m3, [xq + 96]
    add         xq, 128
    dec         r2
    jnz         .loop
.reduce:
    vaddps      m0, m0, m1
    vaddps      m2, m2, m3
    vaddps      m0, m0, m2
    vextractf128 xm1, m0, 1
    vaddps      xm0, xm0, xm1
    vhaddps     xm0, xm0, xm0
    vhaddps     xm0, xm0, xm0
    and         nq, 31                  ; tail: n % 32
    jz          .done
.tail:
    vaddss      xm0, xm0, [xq]
    add         xq, 4
    dec         nq
    jnz         .tail
.done:
    RET
