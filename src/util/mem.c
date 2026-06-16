/*
 * peregrine - aligned memory helpers (implementation)
 */
#include "util/mem.h"

#include <stdlib.h>

void *pg_aligned_alloc(size_t alignment, size_t size)
{
    void *ptr = NULL;
    if (size == 0)
        size = 1;
    if (posix_memalign(&ptr, alignment, size) != 0)
        return NULL;
    return ptr;
}

void pg_aligned_free(void *ptr)
{
    free(ptr);
}
