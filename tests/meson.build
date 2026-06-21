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
  'checkasm/checkasm_qgemv.c',
  'checkasm/checkasm_fp16.c',
  'checkasm/checkasm_gemm_bf16.c',
  'checkasm/checkasm_gemm_fp16.c',
  checkasm_asm,
  include_directories : include_directories('checkasm'),
  dependencies : peregrine_dep)

# Correctness fuzzing is the gate wired into `meson test` / `ninja test`.
test('checkasm', checkasm)

model_file_test = executable('test_model_file',
  'test_model_file.c',
  dependencies : peregrine_dep)

test('model_file', model_file_test)

tokenizer_test = executable('test_tokenizer',
  'test_tokenizer.c',
  dependencies : peregrine_dep)

test('tokenizer', tokenizer_test)

llama_test = executable('test_llama',
  'test_llama.c',
  dependencies : peregrine_dep)

test('llama', llama_test)

python3 = find_program('python3', required : false)
if python3.found()
  test('llama_cpp_parity', python3,
    args : [files('test_llama_cpp_parity.py'), peregrine_exe],
    timeout : 120)
endif

# Benchmarks are opt-in (slow); run manually with: build/tests/checkasm --bench
