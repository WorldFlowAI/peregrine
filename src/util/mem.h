/*
 * peregrine - aligned memory helpers
 *
 * Hot kernels assume their inputs are aligned to the widest vector they use
 * (64 bytes covers AVX-512 and gives ARM cache-line alignment). All tensor
 * buffers route through here so SIMD loads can stay aligned.
 */
#ifndef PEREGRINE_UTIL_MEM_H
#define PEREGRINE_UTIL_MEM_H

#include <stddef.h>

#define PG_ALIGN 64

/* Allocate `size` bytes aligned to `alignment` (power of two). NULL on OOM. */
void *pg_aligned_alloc(size_t alignment, size_t size);

/* Free a pointer returned by pg_aligned_alloc. */
void pg_aligned_free(void *ptr);

#endif /* PEREGRINE_UTIL_MEM_H */
