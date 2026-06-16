; peregrine - f32 axpy, x86-64 AVX2+FMA:  y[i] += alpha * x[i]
;
;   void pg_axpy_f32_avx2(float alpha, const float *x, float *y, size_t n)
;     xmm0 = alpha, rdi = x, rsi = y, rdx = n   (alpha is the 1st FP arg, in xmm0)
;
; 16 floats/iteration (two independent groups for ILP), load-modify-store.

%include "config.asm"
%include "ext/x86/x86inc.asm"

SECTION .text

INIT_YMM avx2
cglobal axpy_f32, 3, 4, 8, x, y, n
    vbroadcastss m1, xm0           ; m1 = alpha in all lanes (xmm0 holds alpha)
    mov         r3, nq
    shr         r3, 4              ; r3 = n / 16
    jz          .tail
.loop:
    vmovups     m2, [xq]
    vmovups     m3, [yq]
    vfmadd231ps m3, m2, m1         ; y = x*alpha + y
    vmovups     m4, [xq + 32]
    vmovups     m5, [yq + 32]
    vfmadd231ps m5, m4, m1
    vmovups     [yq], m3
    vmovups     [yq + 32], m5
    add         xq, 64
    add         yq, 64
    dec         r3
    jnz         .loop
.tail:
    and         nq, 15             ; scalar tail: n % 16
    jz          .done
.t1:
    vmovss      xm2, [xq]
    vmovss      xm3, [yq]
    vfmadd231ss xm3, xm2, xm0      ; y += x*alpha
    vmovss      [yq], xm3
    add         xq, 4
    add         yq, 4
    dec         nq
    jnz         .t1
.done:
    RET
