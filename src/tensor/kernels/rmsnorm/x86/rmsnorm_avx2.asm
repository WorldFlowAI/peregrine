; peregrine - f32 RMSNorm, x86-64 AVX2+FMA
;
;   void pg_rmsnorm_f32_avx2(float *out, const float *x, const float *w,
;                            size_t n, float eps)
;     rdi=out, rsi=x, rdx=w, rcx=n, xmm0=eps
;
; Pass 1: ss = sum(x*x).  Scale = 1/sqrt(ss/n + eps) (scalar).
; Pass 2: out[i] = x[i] * scale * w[i].

%include "config.asm"
%include "ext/x86/x86inc.asm"

SECTION .text

INIT_YMM avx2
cglobal rmsnorm_f32, 4, 6, 12, out, x, w, n
    test        nq, nq
    jz          .ret                    ; n == 0

    ; ---- pass 1: ss = sum(x*x) ----
    vxorps      m1, m1, m1
    vxorps      m2, m2, m2
    vxorps      m3, m3, m3
    vxorps      m4, m4, m4
    mov         r4, xq                  ; r4 walks x (keep xq for pass 2)
    mov         r5, nq
    shr         r5, 5                   ; n / 32
    jz          .p1tail
.p1:
    vmovups     m8,  [r4]
    vfmadd231ps m1, m8, m8
    vmovups     m9,  [r4 + 32]
    vfmadd231ps m2, m9, m9
    vmovups     m10, [r4 + 64]
    vfmadd231ps m3, m10, m10
    vmovups     m11, [r4 + 96]
    vfmadd231ps m4, m11, m11
    add         r4, 128
    dec         r5
    jnz         .p1
.p1tail:
    vaddps      m1, m1, m2
    vaddps      m3, m3, m4
    vaddps      m1, m1, m3
    vextractf128 xm2, m1, 1
    vaddps      xm1, xm1, xm2
    vhaddps     xm1, xm1, xm1
    vhaddps     xm1, xm1, xm1           ; xm1[0] = partial ss
    mov         r5, nq
    and         r5, 31
    jz          .scale
.p1t:
    vmovss      xm8, [r4]
    vfmadd231ss xm1, xm8, xm8
    add         r4, 4
    dec         r5
    jnz         .p1t
.scale:
    ; scale = 1 / sqrt(ss/n + eps)
    vcvtsi2ss   xm2, xm2, nq            ; (float)n  (64-bit source)
    vdivss      xm1, xm1, xm2           ; mean = ss/n
    vaddss      xm1, xm1, xm0           ; + eps
    vsqrtss     xm1, xm1, xm1
    mov         eax, 0x3f800000         ; 1.0f
    vmovd       xm2, eax
    vdivss      xm1, xm2, xm1           ; scale (scalar, in xm1)
    vbroadcastss m5, xm1

    ; ---- pass 2: out = x * scale * w ----
    mov         r5, nq
    shr         r5, 4                   ; n / 16
    jz          .p2tail
.p2:
    vmovups     m8,  [xq]
    vmovups     m9,  [wq]
    vmulps      m8, m8, m5
    vmulps      m8, m8, m9
    vmovups     [outq], m8
    vmovups     m10, [xq + 32]
    vmovups     m11, [wq + 32]
    vmulps      m10, m10, m5
    vmulps      m10, m10, m11
    vmovups     [outq + 32], m10
    add         xq, 64
    add         wq, 64
    add         outq, 64
    dec         r5
    jnz         .p2
.p2tail:
    mov         r5, nq
    and         r5, 15
    jz          .ret
.p2t:
    vmovss      xm8, [xq]
    vmovss      xm9, [wq]
    vmulss      xm8, xm8, xm1           ; * scale
    vmulss      xm8, xm8, xm9           ; * w
    vmovss      [outq], xm8
    add         xq, 4
    add         wq, 4
    add         outq, 4
    dec         r5
    jnz         .p2t
.ret:
    RET
