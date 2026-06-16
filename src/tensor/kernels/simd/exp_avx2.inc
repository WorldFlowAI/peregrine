; peregrine - shared AVX2 exp building block (cephes range reduction + minimax).
;
; Reused by the exp kernel and exp-based kernels. Assumes config.asm and
; ext/x86/x86inc.asm are already %included, and INIT_YMM avx2 is in effect.
; Constants are stored broadcast (8 copies) so they work as 256-bit memory
; operands; the labels are object-local, so each including TU gets its own copy
; (the data is tiny).
;
; Contract:
;   EXP_PS_AVX2 r  - exp(r) in place on ymm r, clobbering m1,m2,m3.
;   EXP_SS_AVX2 r  - exp(r) in place on xmm r (scalar), clobbering xm1,xm2,xm3,eax.

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
c_lo:    times 8 dd -87.3365478515625   ; keeps m >= -126 (a lower clamp lets m
                                        ; reach -128, making the 2^m bit-trick -inf)
c_127i:  times 8 dd 127

%macro EXP_PS_AVX2 1
    vminps      %1, %1, [c_hi]
    vmaxps      %1, %1, [c_lo]
    vmulps      m1, %1, [c_log2e]
    vroundps    m1, m1, 0x00
    vfnmadd231ps %1, m1, [c_c1]
    vfnmadd231ps %1, m1, [c_c2]
    vmulps      m2, %1, %1
    vmovups     m3, [c_p0]
    vfmadd213ps m3, %1, [c_p1]
    vfmadd213ps m3, %1, [c_p2]
    vfmadd213ps m3, %1, [c_p3]
    vfmadd213ps m3, %1, [c_p4]
    vfmadd213ps m3, %1, [c_p5]
    vfmadd213ps m3, m2, %1
    vaddps      m3, m3, [c_one]
    vcvtps2dq   m1, m1
    vpaddd      m1, m1, [c_127i]
    vpslld      m1, m1, 23
    vmulps      %1, m3, m1
%endmacro

%macro EXP_SS_AVX2 1
    vminss      %1, %1, [c_hi]
    vmaxss      %1, %1, [c_lo]
    vmulss      xm1, %1, [c_log2e]
    vroundss    xm1, xm1, xm1, 0x00
    vfnmadd231ss %1, xm1, [c_c1]
    vfnmadd231ss %1, xm1, [c_c2]
    vmulss      xm2, %1, %1
    vmovss      xm3, [c_p0]
    vfmadd213ss xm3, %1, [c_p1]
    vfmadd213ss xm3, %1, [c_p2]
    vfmadd213ss xm3, %1, [c_p3]
    vfmadd213ss xm3, %1, [c_p4]
    vfmadd213ss xm3, %1, [c_p5]
    vfmadd213ss xm3, xm2, %1
    vaddss      xm3, xm3, [c_one]
    vcvtss2si   eax, xm1
    add         eax, 127
    shl         eax, 23
    vmovd       xm1, eax
    vmulss      %1, xm3, xm1
%endmacro
