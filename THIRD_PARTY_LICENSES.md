# Third-party licenses

Peregrine is licensed under BSD-2-Clause (see [LICENSE](LICENSE)). It vendors a
small number of third-party files, kept verbatim under their own licenses. Those
licenses are reproduced in the file headers and summarised here.

| Path | Component | Upstream | License |
|------|-----------|----------|---------|
| `src/ext/x86/x86inc.asm` | x86 assembly macro layer | x264 project (also used by FFmpeg, dav1d) | ISC |
| `src/ext/arm/asm.S` | AArch64 assembly macro layer | VideoLAN / dav1d authors | BSD-2-Clause |

Both licenses are permissive and compatible with Peregrine's BSD-2-Clause. The
full license text for each component is in the header of the corresponding file;
do not strip those headers. See `src/ext/README.md` for provenance and the
commands used to re-vendor these files.
