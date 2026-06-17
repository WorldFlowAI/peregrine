/*
 * peregrine - public C API (umbrella header)
 *
 * Stable C ABI surface for embedders. v0.1 exposes only capability + version;
 * the model/session API below is the v0.x target sketched in ROADMAP.md and is
 * commented out until implemented so this header always reflects reality.
 */
#ifndef PEREGRINE_H
#define PEREGRINE_H

#include <stddef.h>
#include "peregrine/model.h"
#include "peregrine/tokenizer.h"
#include "peregrine/version.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Bitmask of em CPU features the running binary detected. */
unsigned pg_get_cpu_flags(void);
const char *pg_cpu_flags_str(unsigned flags, char *buf, unsigned buflen);

/* ---- planned v0.x engine API (not yet implemented) -----------------------
 *
 * typedef struct PgModel   PgModel;     // weights + metadata (GGUF/safetensors)
 * typedef struct PgContext PgContext;   // KV cache + decode state
 *
 * PgModel   *pg_model_load(const char *path, PgError *err);
 * PgContext *pg_context_new(PgModel *m, const PgContextParams *p);
 * int        pg_eval(PgContext *c, const int32_t *tokens, size_t n_tokens);
 * int32_t    pg_sample(PgContext *c, const PgSamplerParams *p);
 * void       pg_context_free(PgContext *c);
 * void       pg_model_free(PgModel *m);
 *
 * ------------------------------------------------------------------------- */

#ifdef __cplusplus
}
#endif

#endif /* PEREGRINE_H */
