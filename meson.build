project('peregrine', 'c',
  version : '0.1.0',
  license : 'BSD-2-Clause',
  default_options : [
    'c_std=c11',
    'optimization=3',
    'warning_level=2',
    'b_ndebug=if-release',
  ])

# Pick the asm path from the C compiler's OWN target macros, not Meson's
# host_machine detection. The latter reads the arch of the Python running
# Meson, which is wrong when Meson runs under an emulated interpreter (e.g. a
# Rosetta/x86_64 Python on an arm64 Mac) and would mis-select the kernels.
# The compiler macros always match the objects clang actually emits.
cc = meson.get_compiler('c')
target_x86 = cc.get_define('__x86_64__') != ''
target_arm = cc.get_define('__aarch64__') != ''

# With -std=c11, glibc hides POSIX symbols (clock_gettime, posix_memalign, ...)
# unless a feature-test macro is set. macOS exposes them regardless, so this is
# needed for Linux/glibc builds.
add_project_arguments('-D_POSIX_C_SOURCE=200809L', language : 'c')

# libm is separate on Linux/glibc (rmsnorm uses sqrtf); a no-op on macOS.
m_dep = cc.find_library('m', required : false)

# pthreads for the thread pool (util/thread); folded into libc on macOS.
threads_dep = dependency('threads')

# NASM is only needed on x86-64; on ARM the .S kernels go through the C
# compiler's integrated assembler.
if target_x86
  add_languages('nasm', native : false)
endif

incdir = include_directories('include', 'src')

c_sources = files(
  'src/util/cpu.c',
  'src/util/thread.c',
  'src/util/mem.c',
  'src/util/filemap.c',
  'src/model/model_file.c',
  'src/token/tokenizer.c',
  'src/tensor/kernels/dot/dot_ref.c',
  'src/tensor/kernels/dot/dot_init.c',
  'src/tensor/kernels/axpy/axpy_ref.c',
  'src/tensor/kernels/axpy/axpy_init.c',
  'src/tensor/kernels/rmsnorm/rmsnorm_ref.c',
  'src/tensor/kernels/rmsnorm/rmsnorm_init.c',
  'src/tensor/kernels/sum/sum_ref.c',
  'src/tensor/kernels/sum/sum_init.c',
  'src/tensor/kernels/max/max_ref.c',
  'src/tensor/kernels/max/max_init.c',
  'src/tensor/kernels/mul/mul_ref.c',
  'src/tensor/kernels/mul/mul_init.c',
  'src/tensor/kernels/add/add_ref.c',
  'src/tensor/kernels/add/add_init.c',
  'src/tensor/kernels/silu/silu_ref.c',
  'src/tensor/kernels/silu/silu_init.c',
  'src/tensor/kernels/gelu/gelu_ref.c',
  'src/tensor/kernels/gelu/gelu_init.c',
  'src/tensor/kernels/softmax/softmax_ref.c',
  'src/tensor/kernels/softmax/softmax_init.c',
  'src/tensor/kernels/rope/rope_ref.c',
  'src/tensor/kernels/rope/rope_init.c',
  'src/tensor/kernels/gemm/gemm_ref.c',
  'src/tensor/kernels/gemm/gemm_init.c',
  'src/tensor/kernels/gemm/gemm_blocked.c',
  'src/tensor/kernels/gemv/gemv_ref.c',
  'src/tensor/kernels/gemv/gemv_init.c',
  'src/tensor/kernels/gemv/gemv_driver.c',
  'src/tensor/kernels/gemm/gemm_bf16.c',
  'src/tensor/kernels/gemm/gemm_bf16_init.c',
  'src/tensor/kernels/gemm/gemm_fp16.c',
  'src/tensor/kernels/gemm/gemm_fp16_init.c',
  'src/tensor/kernels/exp/exp_ref.c',
  'src/tensor/kernels/exp/exp_init.c',
)

asm_sources = []
if target_x86
  asm_sources += files(
    'src/tensor/kernels/dot/x86/dot_avx2.asm',
    'src/tensor/kernels/axpy/x86/axpy_avx2.asm',
    'src/tensor/kernels/rmsnorm/x86/rmsnorm_avx2.asm',
    'src/tensor/kernels/sum/x86/sum_avx2.asm',
    'src/tensor/kernels/max/x86/max_avx2.asm',
    'src/tensor/kernels/mul/x86/mul_avx2.asm',
    'src/tensor/kernels/add/x86/add_avx2.asm',
    'src/tensor/kernels/exp/x86/exp_avx2.asm',
    'src/tensor/kernels/silu/x86/silu_avx2.asm',
    'src/tensor/kernels/gelu/x86/gelu_avx2.asm',
    'src/tensor/kernels/softmax/x86/softmax_avx2.asm',
    'src/tensor/kernels/gemm/x86/sgemm_ukernel_avx2.asm',
    'src/tensor/kernels/gemm/gemm_avx2.c',
    'src/tensor/kernels/gemv/gemv_avx2.c',
    'src/tensor/kernels/gemm/gemm_bf16_avx2.c',
    'src/tensor/kernels/gemm/gemm_bf16_avx512bf16.c',
    'src/tensor/kernels/gemm/x86/bf16gemm_ukernel_avx512bf16.asm',
    'src/tensor/kernels/gemm/gemm_fp16_avx2.c',
  )
elif target_arm
  asm_sources += files(
    'src/tensor/kernels/dot/arm/dot_neon.S',
    'src/tensor/kernels/axpy/arm/axpy_neon.S',
    'src/tensor/kernels/rmsnorm/arm/rmsnorm_neon.S',
    'src/tensor/kernels/sum/arm/sum_neon.S',
    'src/tensor/kernels/max/arm/max_neon.S',
    'src/tensor/kernels/mul/arm/mul_neon.S',
    'src/tensor/kernels/add/arm/add_neon.S',
    'src/tensor/kernels/exp/arm/exp_neon.S',
    'src/tensor/kernels/silu/arm/silu_neon.S',
    'src/tensor/kernels/gelu/arm/gelu_neon.S',
    'src/tensor/kernels/softmax/arm/softmax_neon.S',
    'src/tensor/kernels/rope/arm/rope_neon.S',
    'src/tensor/kernels/gemm/arm/sgemm_ukernel_neon.S',
    'src/tensor/kernels/gemm/gemm_neon.c',
    'src/tensor/kernels/gemv/gemv_neon.c',
    'src/tensor/kernels/gemm/gemm_bf16_neon.c',
    'src/tensor/kernels/gemm/gemm_bf16_mmla.c',
    'src/tensor/kernels/gemm/arm/bf16gemm_ukernel_neon.S',
    'src/tensor/kernels/gemm/gemm_fp16_neon.c',
    'src/tensor/kernels/gemm/gemm_fp16_f16.c',
    'src/tensor/kernels/gemm/arm/fp16gemm_ukernel_neon.S',
  )
endif

libperegrine = static_library('peregrine',
  c_sources, asm_sources,
  include_directories : incdir,
  dependencies : [m_dep, threads_dep],
  install : true)

peregrine_dep = declare_dependency(
  link_with : libperegrine,
  include_directories : incdir,
  dependencies : [m_dep, threads_dep])

executable('peregrine',
  'src/cli/peregrine.c',
  dependencies : peregrine_dep,
  install : true)

subdir('tests')
