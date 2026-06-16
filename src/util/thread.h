/*
 * peregrine - minimal persistent thread pool + blocking parallel-for.
 *
 * One pool of worker threads is created once and reused; pg_parallel_for splits
 * an index range into grain-sized chunks that workers (and the calling thread)
 * claim from a shared atomic cursor, then blocks until every chunk is done. This
 * is the parallel substrate for the tiled GEMM driver and any other kernel that
 * partitions over disjoint output regions.
 *
 * The callback runs on an arbitrary worker thread and must only touch state it
 * owns for its [begin, end) slice (the GEMM driver gives each chunk a disjoint
 * stripe of C, so no locking is needed in the callback).
 */
#ifndef PEREGRINE_UTIL_THREAD_H
#define PEREGRINE_UTIL_THREAD_H

#include <stddef.h>

typedef struct PgThreadPool PgThreadPool;

/* Create a pool with n_threads workers; n_threads <= 0 picks the online CPU
 * count. Returns NULL on failure. */
PgThreadPool *pg_threadpool_create(int n_threads);
void          pg_threadpool_destroy(PgThreadPool *pool);
int           pg_threadpool_size(const PgThreadPool *pool);

/*
 * Process [0, n) in chunks of ~grain by calling fn(ctx, begin, end) on the pool
 * (the calling thread participates). Blocks until all chunks complete. With a
 * NULL pool, or n small enough to not be worth splitting, fn runs inline on the
 * caller. grain == 0 is treated as 1.
 */
void pg_parallel_for(PgThreadPool *pool, size_t n, size_t grain,
                     void (*fn)(void *ctx, size_t begin, size_t end), void *ctx);

/* Process-wide lazily-created pool (online CPU count), for kernels that don't
 * thread an explicit pool through their public signature. Never destroyed (it
 * lives for the process). Returns NULL if creation failed. */
PgThreadPool *pg_global_threadpool(void);

#endif /* PEREGRINE_UTIL_THREAD_H */
