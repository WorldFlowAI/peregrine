; peregrine - f32 RoPE rotation, x86-64 AVX2.
;   void pg_rope_f32_avx2(float *out, const float *in, const float *cos,
;                         const float *sin, size_t n_tokens, size_t n_heads,
;                         size_t head_dim, PgRopeMode mode)
%include "config.asm"
%include "ext/x86/x86inc.asm"

SECTION .text

INIT_YMM avx2
cglobal rope_f32, 8, 15, 8, out, in, cos, sin, nt, nh, hd, mode
    mov         r8, hdq
    shr         r8, 1                  ; half = head_dim / 2
    jz          .done
    test        ntq, ntq
    jz          .done

    lea         r9, [r8 * 4]           ; half bytes
    cmp         modeq, 1
    je          .neox

; ===== interleaved scalar fallback: a=x[2i], b=x[2i+1] =====
.itok:
    mov         r11, nhq
.ihead:
    mov         r12, cosq
    mov         r13, sinq
    mov         r14, r8
.ipair:
    vmovss      xm0, [inq]
    vmovss      xm1, [inq + 4]
    vmovss      xm2, [r12]
    vmovss      xm3, [r13]
    vmulss      xm4, xm0, xm2
    vmulss      xm5, xm1, xm3
    vsubss      xm4, xm4, xm5
    vmulss      xm6, xm0, xm3
    vmulss      xm7, xm1, xm2
    vaddss      xm6, xm6, xm7
    vmovss      [outq], xm4
    vmovss      [outq + 4], xm6
    add         inq, 8
    add         outq, 8
    add         r12, 4
    add         r13, 4
    dec         r14
    jnz         .ipair
    dec         r11
    jnz         .ihead
    add         cosq, r9
    add         sinq, r9
    dec         ntq
    jnz         .itok
    RET

; ===== NeoX vector path: a=x[i], b=x[i+half] =====
.neox:
.ntok:
    mov         r11, nhq
.nhead:
    mov         r12, cosq
    mov         r13, sinq
    mov         r14, r8
    shr         r14, 3
    jz          .ntail
.nvec:
    vmovups     m0, [inq]
    vmovups     m1, [inq + r9]
    vmovups     m2, [r12]
    vmovups     m3, [r13]
    vmulps      m4, m0, m2
    vmulps      m5, m1, m3
    vsubps      m4, m4, m5             ; a*cos - b*sin
    vmulps      m6, m0, m3
    vmulps      m7, m1, m2
    vaddps      m6, m6, m7             ; a*sin + b*cos
    vmovups     [outq], m4
    vmovups     [outq + r9], m6
    add         inq, 32
    add         outq, 32
    add         r12, 32
    add         r13, 32
    dec         r14
    jnz         .nvec
.ntail:
    mov         r14, r8
    and         r14, 7
    jz          .nnext_head
.nt1:
    vmovss      xm0, [inq]
    vmovss      xm1, [inq + r9]
    vmovss      xm2, [r12]
    vmovss      xm3, [r13]
    vmulss      xm4, xm0, xm2
    vmulss      xm5, xm1, xm3
    vsubss      xm4, xm4, xm5
    vmulss      xm6, xm0, xm3
    vmulss      xm7, xm1, xm2
    vaddss      xm6, xm6, xm7
    vmovss      [outq], xm4
    vmovss      [outq + r9], xm6
    add         inq, 4
    add         outq, 4
    add         r12, 4
    add         r13, 4
    dec         r14
    jnz         .nt1
.nnext_head:
    add         inq, r9
    add         outq, r9
    dec         r11
    jnz         .nhead
    add         cosq, r9
    add         sinq, r9
    dec         ntq
    jnz         .ntok
.done:
    RET
