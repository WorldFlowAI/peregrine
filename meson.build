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

# NASM is only needed on x86-64; on ARM the .S kernels go through the C
# compiler's integrated assembler.
if target_x86
  add_languages('nasm', native : false)
endif

incdir = include_directories('include', 'src')

c_sources = files(
  'src/util/cpu.c',
  'src/util/mem.c',
  'src/tensor/kernels/dot/dot_ref.c',
  'src/tensor/kernels/dot/dot_init.c',
  'src/tensor/kernels/axpy/axpy_ref.c',
  'src/tensor/kernels/axpy/axpy_init.c',
  'src/tensor/kernels/rmsnorm/rmsnorm_ref.c',
  'src/tensor/kernels/rmsnorm/rmsnorm_init.c',
)

asm_sources = []
if target_x86
  asm_sources += files(
    'src/tensor/kernels/dot/x86/dot_avx2.asm',
    'src/tensor/kernels/axpy/x86/axpy_avx2.asm',
    'src/tensor/kernels/rmsnorm/x86/rmsnorm_avx2.asm',
  )
elif target_arm
  asm_sources += files(
    'src/tensor/kernels/dot/arm/dot_neon.S',
    'src/tensor/kernels/axpy/arm/axpy_neon.S',
    'src/tensor/kernels/rmsnorm/arm/rmsnorm_neon.S',
  )
endif

libperegrine = static_library('peregrine',
  c_sources, asm_sources,
  include_directories : incdir,
  install : true)

peregrine_dep = declare_dependency(
  link_with : libperegrine,
  include_directories : incdir)

executable('peregrine',
  'src/cli/peregrine.c',
  dependencies : peregrine_dep,
  install : true)

subdir('tests')
