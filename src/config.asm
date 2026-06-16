;*****************************************************************************
;* config.asm - Peregrine build-config shim for x86inc.asm
;*****************************************************************************
;* x86inc.asm expects a handful of build-time symbols to be defined before it
;* is included. In dav1d/FFmpeg these come from a meson/configure-generated
;* config.asm; Peregrine authors them here (config values are not part of
;* x86inc's ISC license). Include this BEFORE ext/x86/x86inc.asm.
;*****************************************************************************

; Symbol mangling: cglobal foo -> <private_prefix>_foo, i.e. pg_foo.
%define private_prefix pg

; We only assemble 64-bit kernels.
%define ARCH_X86_64 1

; Mach-O prepends an underscore to C symbols; define PREFIX so x86inc's
; mangle() adds it. ELF does not, so leave PREFIX undefined there.
%ifidn __OUTPUT_FORMAT__, macho64
    %define PREFIX
%elifidn __OUTPUT_FORMAT__, macho32
    %define PREFIX
%endif
