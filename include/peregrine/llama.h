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
typedef struct PgSampler PgSampler;

typedef struct PgSamplerParams {
    float temperature;  /* <= 0: greedy */
    size_t top_k;       /* 0: disabled */
    float top_p;        /* <= 0 or >= 1: disabled */
    uint64_t seed;
} PgSamplerParams;

typedef struct PgLlamaProfile {
    size_t tokens;
    double token_embed_sec;
    double rope_sec;
    double attn_norm_sec;
    double qkv_sec;
    double attn_scores_sec;
    double attn_softmax_sec;
    double attn_mix_sec;
    double attn_output_sec;
    double attn_residual_sec;
    double ffn_norm_sec;
    double ffn_gate_up_sec;
    double ffn_act_sec;
    double ffn_down_sec;
    double ffn_residual_sec;
    double output_norm_sec;
    double logits_sec;
} PgLlamaProfile;

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
void pg_llama_context_profile_enable(PgLlamaContext *ctx, int enabled);
void pg_llama_context_profile_reset(PgLlamaContext *ctx);
const PgLlamaProfile *pg_llama_context_profile(const PgLlamaContext *ctx);

int pg_llama_eval_token(PgLlamaContext *ctx, int32_t token,
                        const float **logits,
                        char *err, size_t err_len);
const float *pg_llama_logits(const PgLlamaContext *ctx);
size_t pg_llama_context_position(const PgLlamaContext *ctx);

int32_t pg_llama_sample_greedy(const float *logits, size_t n_logits);
PgSampler *pg_sampler_new(const PgSamplerParams *params, char *err, size_t err_len);
void pg_sampler_free(PgSampler *sampler);
int32_t pg_llama_sample(const float *logits, size_t n_logits,
                        PgSampler *sampler, char *err, size_t err_len);

#ifdef __cplusplus
}
#endif

#endif /* PEREGRINE_LLAMA_H */
