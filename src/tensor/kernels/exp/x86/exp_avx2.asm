; peregrine - f32 exp, x86-64 AVX2+FMA (cephes range reduction + minimax poly).
;
;   void pg_exp_f32_avx2(const float *in, float *out, size_t n)
;     rdi = in, rsi = out, rdx = n
;
; Same algorithm and constants as the NEON version (validated to ~1e-7 rel err).
; Constants are stored broadcast (8 copies) so they can be used directly as
; 256-bit memory operands, keeping register pressure to 4 ymm regs.

%include "config.asm"
%include "ext/x86/x86inc.asm"

SECTION_RODATA 32
c_log2e: times 8 dd 1.44269504088896341
c_c1:    times 8 dd 0.693359375
c_c2:    times 8 dd -2.12194440e-4
c_p0:    times 8 dd 1.9875691500e-4
c_p1:    times 8 dd 1.3981999507e-3
c_p2:    times 8 dd 8.3334519073e-3
c_p3:    times 8 dd 4.1665795894e-2
c_p4:    times 8 dd 1.6666665459e-1
c_p5:    times 8 dd 5.0000001201e-1
c_one:   times 8 dd 1.0
c_hi:    times 8 dd 88.3762626647950
c_lo:    times 8 dd -88.3762626647950
c_127i:  times 8 dd 127

SECTION .text

INIT_YMM avx2
cglobal exp_f32, 3, 4, 4, in, out, n
    mov         r3, nq
    shr         r3, 3                   ; n / 8
    jz          .tail
.loop:
    vmovups     m0, [inq]
    vminps      m0, m0, [c_hi]
    vmaxps      m0, m0, [c_lo]
    vmulps      m1, m0, [c_log2e]
    vroundps    m1, m1, 0x00            ; m = round-to-nearest
    vfnmadd231ps m0, m1, [c_c1]         ; r = x - m*C1
    vfnmadd231ps m0, m1, [c_c2]         ; r -= m*C2
    vmulps      m2, m0, m0              ; z = r*r
    vmovups     m3, [c_p0]
    vfmadd213ps m3, m0, [c_p1]          ; y = y*r + p1
    vfmadd213ps m3, m0, [c_p2]
    vfmadd213ps m3, m0, [c_p3]
    vfmadd213ps m3, m0, [c_p4]
    vfmadd213ps m3, m0, [c_p5]          ; y5
    vfmadd213ps m3, m2, m0              ; y = y5*z + r
    vaddps      m3, m3, [c_one]         ; + 1  -> e^r
    vcvtps2dq   m1, m1                  ; m -> int
    vpaddd      m1, m1, [c_127i]        ; + 127
    vpslld      m1, m1, 23              ; exponent field -> 2^m
    vmulps      m0, m3, m1              ; e^r * 2^m
    vmovups     [outq], m0
    add         inq, 32
    add         outq, 32
    dec         r3
    jnz         .loop
.tail:
    and         nq, 7                   ; scalar tail
    jz          .done
.t1:
    vmovss      xm0, [inq]
    vminss      xm0, xm0, [c_hi]
    vmaxss      xm0, xm0, [c_lo]
    vmulss      xm1, xm0, [c_log2e]
    vroundss    xm1, xm1, xm1, 0x00
    vfnmadd231ss xm0, xm1, [c_c1]
    vfnmadd231ss xm0, xm1, [c_c2]
    vmulss      xm2, xm0, xm0
    vmovss      xm3, [c_p0]
    vfmadd213ss xm3, xm0, [c_p1]
    vfmadd213ss xm3, xm0, [c_p2]
    vfmadd213ss xm3, xm0, [c_p3]
    vfmadd213ss xm3, xm0, [c_p4]
    vfmadd213ss xm3, xm0, [c_p5]
    vfmadd213ss xm3, xm2, xm0
    vaddss      xm3, xm3, [c_one]
    vcvtss2si   eax, xm1
    add         eax, 127
    shl         eax, 23
    vmovd       xm1, eax
    vmulss      xm0, xm3, xm1
    vmovss      [outq], xm0
    add         inq, 4
    add         outq, 4
    dec         nq
    jnz         .t1
.done:
    RET
