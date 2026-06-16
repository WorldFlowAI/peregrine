; peregrine - f32 dot product, x86-64 AVX2+FMA (hand-written NASM, no intrinsics)
;
;   float pg_dot_f32_avx2(const float *a, const float *b, size_t n)   ->   xmm0
;
; Four 256-bit accumulators (32 floats/iter) hide FMA latency, then a
; horizontal reduce to a scalar and an FMA tail.
;
; Frame via x86inc.asm: cglobal handles the ABI + symbol mangling (pg_*),
; INIT_YMM marks this an AVX function so RET auto-emits vzeroupper, and the
; named args (a/b/n) map to the right registers on SysV/Win64 alike. We keep
; the explicit v-prefixed vector ops for clarity.

%include "config.asm"
%include "ext/x86/x86inc.asm"

SECTION .text

; INIT_YMM avx2 makes cglobal auto-append the ISA suffix, so `dot_f32` here
; exports the symbol `pg_dot_f32_avx2` that dot_init.c calls.
INIT_YMM avx2
cglobal dot_f32, 3, 4, 8, a, b, n
    vxorps      m0, m0, m0
    vxorps      m1, m1, m1
    vxorps      m2, m2, m2
    vxorps      m3, m3, m3

    mov         r3, nq
    shr         r3, 5                   ; r3 = n / 32
    jz          .reduce
.loop:
    vmovups     m4, [aq]
    vfmadd231ps m0, m4, [bq]
    vmovups     m5, [aq + 32]
    vfmadd231ps m1, m5, [bq + 32]
    vmovups     m6, [aq + 64]
    vfmadd231ps m2, m6, [bq + 64]
    vmovups     m7, [aq + 96]
    vfmadd231ps m3, m7, [bq + 96]
    add         aq, 128
    add         bq, 128
    dec         r3
    jnz         .loop
.reduce:
    vaddps      m0, m0, m1
    vaddps      m2, m2, m3
    vaddps      m0, m0, m2
    vextractf128 xm1, m0, 1
    vaddps      xm0, xm0, xm1           ; 4 partial sums
    vhaddps     xm0, xm0, xm0
    vhaddps     xm0, xm0, xm0           ; scalar sum in xm0[0]

    and         nq, 31                  ; scalar tail: n % 32
    jz          .done
.tail:
    vmovss      xm4, [aq]
    vfmadd231ss xm0, xm4, [bq]          ; xmm0 += a*b
    add         aq, 4
    add         bq, 4
    dec         nq
    jnz         .tail
.done:
    RET
