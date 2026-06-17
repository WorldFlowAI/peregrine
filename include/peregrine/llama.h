/*
 * peregrine - minimal Llama-class decoder API
 */
#ifndef PEREGRINE_LLAMA_H
#define PEREGRINE_LLAMA_H

#include <stddef.h>
#include <stdint.h>

#include "peregrine/tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PgLlamaModel PgLlamaModel;
typedef struct PgLlamaContext PgLlamaContext;

PgLlamaModel *pg_llama_model_load(const char *path, char *err, size_t err_len);
void pg_llama_model_free(PgLlamaModel *model);

const PgTokenizer *pg_llama_model_tokenizer(const PgLlamaModel *model);
size_t pg_llama_model_vocab_size(const PgLlamaModel *model);
size_t pg_llama_model_context_length(const PgLlamaModel *model);
size_t pg_llama_model_embedding_dim(const PgLlamaModel *model);
size_t pg_llama_model_layer_count(const PgLlamaModel *model);

PgLlamaContext *pg_llama_context_new(const PgLlamaModel *model,
                                     size_t context_length,
                                     char *err, size_t err_len);
void pg_llama_context_free(PgLlamaContext *ctx);

int pg_llama_eval_token(PgLlamaContext *ctx, int32_t token,
                        const float **logits,
                        char *err, size_t err_len);
const float *pg_llama_logits(const PgLlamaContext *ctx);
size_t pg_llama_context_position(const PgLlamaContext *ctx);

int32_t pg_llama_sample_greedy(const float *logits, size_t n_logits);

#ifdef __cplusplus
}
#endif

#endif /* PEREGRINE_LLAMA_H */
