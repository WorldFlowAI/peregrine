/*
 * peregrine - checkasm AArch64 register-clobber trampoline.
 *
 *   float checkasm_checked_call(void *fn, const void *a, const void *b, size_t n)
 *
 * Forwards (a, b, n, ...) to fn, but first loads every AAPCS64 callee-saved
 * register (x19-x28 and the low 64 bits of v8-v15) with a random canary from
 * checkasm_register_init[]. After fn returns we verify each canary survived;
 * any that didn't means fn violated the ABI. The index of every clobbered
 * register is packed into a bitmask and handed to checkasm_clobber().
 *
 * fn's own return value (x0 / v0) is preserved across the check.
 *
 * NB: each directive is on its own line - LLVM's AArch64 assembler does not
 * treat ';' as a statement separator.
 */
#ifdef __APPLE__
#  define CSYM(x) _##x
#else
#  define CSYM(x) x
#endif

.macro movrel rd, val
#ifdef __APPLE__
    adrp    \rd, \val@PAGE
    add     \rd, \rd, \val@PAGEOFF
#else
    adrp    \rd, \val
    add     \rd, \rd, :lo12:\val
#endif
.endm

    .text
    .p2align 2
    .global CSYM(checkasm_checked_call)
CSYM(checkasm_checked_call):
    /* save the real callee-saved regs we are about to trample, plus lr */
    stp     x19, x20, [sp, #-160]!
    stp     x21, x22, [sp, #16]
    stp     x23, x24, [sp, #32]
    stp     x25, x26, [sp, #48]
    stp     x27, x28, [sp, #64]
    stp     d8,  d9,  [sp, #80]
    stp     d10, d11, [sp, #96]
    stp     d12, d13, [sp, #112]
    stp     d14, d15, [sp, #128]
    stp     x29, x30, [sp, #144]

    /* poison every callee-saved reg with its canary */
    movrel  x9, CSYM(checkasm_register_init)
    ldp     x19, x20, [x9, #0]
    ldp     x21, x22, [x9, #16]
    ldp     x23, x24, [x9, #32]
    ldp     x25, x26, [x9, #48]
    ldp     x27, x28, [x9, #64]
    ldp     d8,  d9,  [x9, #80]
    ldp     d10, d11, [x9, #96]
    ldp     d12, d13, [x9, #112]
    ldp     d14, d15, [x9, #128]

    /* func -> x16, shift forwarded args down one register */
    mov     x16, x0
    mov     x0, x1
    mov     x1, x2
    mov     x2, x3
    mov     x3, x4
    mov     x4, x5
    mov     x5, x6
    mov     x6, x7
    blr     x16

    /* verify canaries; build clobber bitmask in x17 */
    movrel  x9, CSYM(checkasm_register_init)
    mov     x17, #0
    ldp     x10, x11, [x9, #0]
    cmp     x19, x10
    cset    x12, ne
    orr     x17, x17, x12, lsl #0
    cmp     x20, x11
    cset    x12, ne
    orr     x17, x17, x12, lsl #1
    ldp     x10, x11, [x9, #16]
    cmp     x21, x10
    cset    x12, ne
    orr     x17, x17, x12, lsl #2
    cmp     x22, x11
    cset    x12, ne
    orr     x17, x17, x12, lsl #3
    ldp     x10, x11, [x9, #32]
    cmp     x23, x10
    cset    x12, ne
    orr     x17, x17, x12, lsl #4
    cmp     x24, x11
    cset    x12, ne
    orr     x17, x17, x12, lsl #5
    ldp     x10, x11, [x9, #48]
    cmp     x25, x10
    cset    x12, ne
    orr     x17, x17, x12, lsl #6
    cmp     x26, x11
    cset    x12, ne
    orr     x17, x17, x12, lsl #7
    ldp     x10, x11, [x9, #64]
    cmp     x27, x10
    cset    x12, ne
    orr     x17, x17, x12, lsl #8
    cmp     x28, x11
    cset    x12, ne
    orr     x17, x17, x12, lsl #9
    ldp     x10, x11, [x9, #80]
    fmov    x13, d8
    cmp     x13, x10
    cset    x12, ne
    orr     x17, x17, x12, lsl #10
    fmov    x13, d9
    cmp     x13, x11
    cset    x12, ne
    orr     x17, x17, x12, lsl #11
    ldp     x10, x11, [x9, #96]
    fmov    x13, d10
    cmp     x13, x10
    cset    x12, ne
    orr     x17, x17, x12, lsl #12
    fmov    x13, d11
    cmp     x13, x11
    cset    x12, ne
    orr     x17, x17, x12, lsl #13
    ldp     x10, x11, [x9, #112]
    fmov    x13, d12
    cmp     x13, x10
    cset    x12, ne
    orr     x17, x17, x12, lsl #14
    fmov    x13, d13
    cmp     x13, x11
    cset    x12, ne
    orr     x17, x17, x12, lsl #15
    ldp     x10, x11, [x9, #128]
    fmov    x13, d14
    cmp     x13, x10
    cset    x12, ne
    orr     x17, x17, x12, lsl #16
    fmov    x13, d15
    cmp     x13, x11
    cset    x12, ne
    orr     x17, x17, x12, lsl #17

    cbz     x17, .Lcc_ok
    /* clobber detected: report, preserving fn's int (x0) + float (d0) return */
    sub     sp, sp, #16
    str     x0, [sp]
    str     d0, [sp, #8]
    mov     x0, x17
    bl      CSYM(checkasm_clobber)
    ldr     d0, [sp, #8]
    ldr     x0, [sp]
    add     sp, sp, #16
.Lcc_ok:
    /* restore the real callee-saved regs and return */
    ldp     x29, x30, [sp, #144]
    ldp     d14, d15, [sp, #128]
    ldp     d12, d13, [sp, #112]
    ldp     d10, d11, [sp, #96]
    ldp     d8,  d9,  [sp, #80]
    ldp     x27, x28, [sp, #64]
    ldp     x25, x26, [sp, #48]
    ldp     x23, x24, [sp, #32]
    ldp     x21, x22, [sp, #16]
    ldp     x19, x20, [sp], #160
    ret
