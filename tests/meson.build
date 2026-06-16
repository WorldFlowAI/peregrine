# per-arch register-clobber trampoline (target_x86/target_arm from ../meson.build)
checkasm_asm = []
if target_arm
  checkasm_asm += files('checkasm/arm/checkasm.S')
elif target_x86
  checkasm_asm += files('checkasm/x86/checkasm.asm')
endif

checkasm = executable('checkasm',
  'checkasm/checkasm.c',
  'checkasm/checkasm_dot.c',
  'checkasm/checkasm_axpy.c',
  'checkasm/checkasm_rmsnorm.c',
  'checkasm/checkasm_sum.c',
  'checkasm/checkasm_max.c',
  'checkasm/checkasm_mul.c',
  'checkasm/checkasm_add.c',
  'checkasm/checkasm_silu.c',
  'checkasm/checkasm_gelu.c',
  'checkasm/checkasm_softmax.c',
  'checkasm/checkasm_rope.c',
  'checkasm/checkasm_exp.c',
  'checkasm/checkasm_gemm.c',
  'checkasm/checkasm_gemv.c',
  checkasm_asm,
  include_directories : include_directories('checkasm'),
  dependencies : peregrine_dep)

# Correctness fuzzing is the gate wired into `meson test` / `ninja test`.
test('checkasm', checkasm)

# Benchmarks are opt-in (slow); run manually with: build/tests/checkasm --bench
