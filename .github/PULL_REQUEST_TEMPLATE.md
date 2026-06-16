<!-- Thanks for contributing to Peregrine! Keep PRs focused and small. -->

## What & why

<!-- What does this change and why? Link any issue. -->

## Kernel checklist (delete if not a kernel change)

- [ ] C reference present and **not** optimised (it is the correctness oracle)
- [ ] No intrinsics in `src/tensor/kernels/` — hand-written asm only
- [ ] asm goes through the macro layer (`cglobal` / `function`)
- [ ] checkasm entry added/updated; passes fuzzing **and** the register-clobber check
- [ ] Benchmarked — numbers (machine + before/after) in the description below

## Verification

```
meson test -C build            # or: make -f Makefile.bootstrap test
```

<!-- Paste relevant checkasm / bench output. -->

## Sign-off

- [ ] Commits are `Signed-off-by:` (DCO) — `git commit -s`
