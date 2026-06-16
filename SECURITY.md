# Security Policy

Peregrine is a performance-critical library that does a lot of hand-written
assembly and raw pointer work, so memory-safety and correctness reports are
especially valued.

## Reporting a vulnerability

Please **do not** open a public issue for security problems. Instead, use
GitHub's private vulnerability reporting (the **"Report a vulnerability"** button
under the repository's **Security** tab), or email **security@worldflowai.com**.

Include, where possible:
- affected version / commit,
- architecture and ISA path (e.g. `aarch64` NEON, `x86-64` AVX2),
- a minimal reproducer (input sizes, alignment, values) — the checkasm
  `--seed=` value if a fuzz case triggered it,
- the impact you observed (crash, OOB read/write, incorrect result).

We aim to acknowledge reports within a few business days.

## Supported versions

Peregrine is pre-1.0; only the `main` branch is currently supported. Once we cut
releases this section will track supported version ranges.
