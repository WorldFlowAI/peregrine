; peregrine - f32 max, x86-64 AVX2:  xmm0 = max_i x[i]   (n >= 1)
;   rdi = x, rsi = n
%include "config.asm"
%include "ext/x86/x86inc.asm"

SECTION .text

INIT_YMM avx2
cglobal max_f32, 2, 3, 4, x, n
    vbroadcastss m0, [xq]               ; seed all lanes with x[0]
    vmovaps     m1, m0
    vmovaps     m2, m0
    vmovaps     m3, m0
    mov         r2, nq
    shr         r2, 5                   ; n / 32
    jz          .reduce
.loop:
    vmaxps      m0, m0, [xq]
    vmaxps      m1, m1, [xq + 32]
    vmaxps      m2, m2, [xq + 64]
    vmaxps      m3, m3, [xq + 96]
    add         xq, 128
    dec         r2
    jnz         .loop
.reduce:
    vmaxps      m0, m0, m1
    vmaxps      m2, m2, m3
    vmaxps      m0, m0, m2
    vextractf128 xm1, m0, 1
    vmaxps      xm0, xm0, xm1           ; 4 lanes
    vshufps     xm1, xm0, xm0, 0xb1     ; [1,0,3,2]
    vmaxps      xm0, xm0, xm1
    vshufps     xm1, xm0, xm0, 0x4e     ; [2,3,0,1]
    vmaxps      xm0, xm0, xm1           ; xm0[0] = max
    and         nq, 31                  ; tail: n % 32
    jz          .done
.tail:
    vmaxss      xm0, xm0, [xq]
    add         xq, 4
    dec         nq
    jnz         .tail
.done:
    RET
