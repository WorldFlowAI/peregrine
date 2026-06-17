/*
 * peregrine - persistent pthread pool + blocking parallel-for.
 *
 * Workers wait on a generation counter; pg_parallel_for bumps the generation to
 * publish a job, then the calling thread and the woken workers all pull chunks
 * from a shared cursor (`next`) under the mutex until the range is exhausted.
 * The caller blocks until no worker is still active on the job, so the ctx it
 * passed stays alive for the whole call.
 */
/* _SC_NPROCESSORS_ONLN is a Darwin extension hidden under strict -std=c11 /
 * _POSIX_C_SOURCE; _DARWIN_C_SOURCE re-exposes it (and is ignored elsewhere).
 * glibc exposes the name unconditionally, so this also works on Linux. */
#define _DARWIN_C_SOURCE 1

#include "util/thread.h"

#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

static int online_cpus(void)
{
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return (n > 0) ? (int)n : 1;
}

struct PgThreadPool {
    pthread_t      *threads;
    int             n;            /* worker threads spawned                  */
    pthread_mutex_t mtx;
    pthread_cond_t  wake;         /* workers wait here for a new generation  */
    pthread_cond_t  done;         /* caller waits here for active -> 0        */

    /* current job (published under mtx) */
    void          (*fn)(void *, size_t, size_t);
    void           *ctx;
    size_t          n_items;
    size_t          grain;
    size_t          next;         /* next index to claim                     */
    int             active;       /* workers currently inside the job        */
    uint64_t        gen;          /* incremented once per published job      */
    int             shutdown;
};

/* Claim and run chunks until the range is exhausted. Caller holds mtx on entry
 * and on return. */
static void run_chunks(PgThreadPool *p)
{
    for (;;) {
        if (p->next >= p->n_items)
            return;
        size_t b = p->next;
        size_t e = b + p->grain;
        if (e > p->n_items) e = p->n_items;
        p->next = e;

        void (*fn)(void *, size_t, size_t) = p->fn;
        void *ctx = p->ctx;
        pthread_mutex_unlock(&p->mtx);
        fn(ctx, b, e);
        pthread_mutex_lock(&p->mtx);
    }
}

static void *worker_main(void *arg)
{
    PgThreadPool *p = arg;
    uint64_t seen = 0;

    pthread_mutex_lock(&p->mtx);
    for (;;) {
        while (p->gen == seen && !p->shutdown)
            pthread_cond_wait(&p->wake, &p->mtx);
        if (p->shutdown)
            break;
        seen = p->gen;

        p->active++;
        run_chunks(p);
        p->active--;
        if (p->active == 0)
            pthread_cond_signal(&p->done);
    }
    pthread_mutex_unlock(&p->mtx);
    return NULL;
}

PgThreadPool *pg_threadpool_create(int n_threads)
{
    if (n_threads <= 0)
        n_threads = online_cpus();

    PgThreadPool *p = calloc(1, sizeof *p);
    if (!p)
        return NULL;
    /* n workers besides the calling thread; with 1 CPU we spawn none and run
     * everything inline. */
    p->n = n_threads - 1;
    if (p->n < 0) p->n = 0;

    if (pthread_mutex_init(&p->mtx, NULL) != 0) { free(p); return NULL; }
    if (pthread_cond_init(&p->wake, NULL) != 0) {
        pthread_mutex_destroy(&p->mtx); free(p); return NULL;
    }
    if (pthread_cond_init(&p->done, NULL) != 0) {
        pthread_cond_destroy(&p->wake); pthread_mutex_destroy(&p->mtx); free(p); return NULL;
    }

    if (p->n > 0) {
        p->threads = calloc((size_t)p->n, sizeof *p->threads);
        if (!p->threads) {
            pthread_cond_destroy(&p->done); pthread_cond_destroy(&p->wake);
            pthread_mutex_destroy(&p->mtx); free(p); return NULL;
        }
        for (int i = 0; i < p->n; i++) {
            if (pthread_create(&p->threads[i], NULL, worker_main, p) != 0) {
                /* spawned fewer than asked: shrink and carry on */
                p->n = i;
                break;
            }
        }
    }
    return p;
}

void pg_threadpool_destroy(PgThreadPool *p)
{
    if (!p)
        return;
    pthread_mutex_lock(&p->mtx);
    p->shutdown = 1;
    pthread_cond_broadcast(&p->wake);
    pthread_mutex_unlock(&p->mtx);

    for (int i = 0; i < p->n; i++)
        pthread_join(p->threads[i], NULL);

    pthread_cond_destroy(&p->done);
    pthread_cond_destroy(&p->wake);
    pthread_mutex_destroy(&p->mtx);
    free(p->threads);
    free(p);
}

int pg_threadpool_size(const PgThreadPool *p)
{
    return p ? p->n + 1 : 1;
}

void pg_parallel_for(PgThreadPool *p, size_t n, size_t grain,
                     void (*fn)(void *, size_t, size_t), void *ctx)
{
    if (n == 0)
        return;
    if (grain == 0)
        grain = 1;

    /* No pool, or only the caller would run anyway: do it inline. */
    if (!p || p->n == 0) {
        fn(ctx, 0, n);
        return;
    }

    pthread_mutex_lock(&p->mtx);
    p->fn      = fn;
    p->ctx     = ctx;
    p->n_items = n;
    p->grain   = grain;
    p->next    = 0;
    p->gen++;
    pthread_cond_broadcast(&p->wake);

    /* The calling thread participates as a worker. */
    p->active++;
    run_chunks(p);
    p->active--;

    while (p->active > 0)
        pthread_cond_wait(&p->done, &p->mtx);
    pthread_mutex_unlock(&p->mtx);
}

/* ---- process-wide pool -------------------------------------------------- */
static pthread_once_t  g_once = PTHREAD_ONCE_INIT;
static PgThreadPool   *g_pool;
static int             g_requested_threads;

static void make_global_pool(void) { g_pool = pg_threadpool_create(g_requested_threads); }

PgThreadPool *pg_global_threadpool(void)
{
    pthread_once(&g_once, make_global_pool);
    return g_pool;
}

int pg_global_threadpool_configure(int n_threads)
{
    if (g_pool)
        return -1;
    g_requested_threads = n_threads;
    return 0;
}
