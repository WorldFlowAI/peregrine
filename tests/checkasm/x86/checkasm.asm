; peregrine - checkasm x86-64 System V register-clobber trampoline.
;
;   float checkasm_checked_call(void *fn, const void *a, const void *b, size_t n)
;     SysV: rdi=fn, rsi=a, rdx=b, rcx=n, r8, r9
;
; Poison the SysV callee-saved GPRs (rbx, rbp, r12-r15) with canaries from
; checkasm_register_init[], forward (a,b,n,...) to fn, then verify the canaries
; survived. Clobbered-register indices are packed into a bitmask for
; checkasm_clobber(). fn's return (rax / xmm0) is preserved across the check.
; (SysV has no callee-saved XMM registers, so only 6 GPRs are checked. Win64,
; which also preserves rsi/rdi/xmm6-15, is a separate ABI - tracked in ROADMAP.)
;
; Assembled-and-symbol-verified on the arm64 dev box via nasm cross-assembly;
; runtime-verified on x86-64 hardware/CI is pending.

%ifidn __OUTPUT_FORMAT__, macho64
    %define cdecl(x) _ %+ x
%else
    %define cdecl(x) x
%endif

extern cdecl(checkasm_register_init)
extern cdecl(checkasm_clobber)

SECTION .text

global cdecl(checkasm_checked_call)
cdecl(checkasm_checked_call):
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 8                  ; realign to 16 before the call

    ; poison callee-saved regs with canaries
    lea     r10, [rel cdecl(checkasm_register_init)]
    mov     rbx, [r10 + 0]
    mov     rbp, [r10 + 8]
    mov     r12, [r10 + 16]
    mov     r13, [r10 + 24]
    mov     r14, [r10 + 32]
    mov     r15, [r10 + 40]

    ; fn -> rax, shift forwarded args down one register
    mov     rax, rdi
    mov     rdi, rsi
    mov     rsi, rdx
    mov     rdx, rcx
    mov     rcx, r8
    mov     r8, r9
    call    rax

    ; verify canaries; build clobber bitmask in r11
    lea     r10, [rel cdecl(checkasm_register_init)]
    xor     r11d, r11d
    cmp     rbx, [r10 + 0]
    setne   r9b
    movzx   r9, r9b
    or      r11, r9
    cmp     rbp, [r10 + 8]
    setne   r9b
    movzx   r9, r9b
    shl     r9, 1
    or      r11, r9
    cmp     r12, [r10 + 16]
    setne   r9b
    movzx   r9, r9b
    shl     r9, 2
    or      r11, r9
    cmp     r13, [r10 + 24]
    setne   r9b
    movzx   r9, r9b
    shl     r9, 3
    or      r11, r9
    cmp     r14, [r10 + 32]
    setne   r9b
    movzx   r9, r9b
    shl     r9, 4
    or      r11, r9
    cmp     r15, [r10 + 40]
    setne   r9b
    movzx   r9, r9b
    shl     r9, 5
    or      r11, r9

    test    r11, r11
    jz      .ok
    ; clobber detected: report, preserving fn's rax + xmm0 return
    sub     rsp, 16
    mov     [rsp], rax
    movsd   [rsp + 8], xmm0
    mov     rdi, r11
    call    cdecl(checkasm_clobber)
    movsd   xmm0, [rsp + 8]
    mov     rax, [rsp]
    add     rsp, 16
.ok:
    add     rsp, 8
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret
