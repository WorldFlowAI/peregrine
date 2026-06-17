/*
 * peregrine - f32 GGUF Llama-class decoder
 */
#include "peregrine/llama.h"

#include "peregrine/model.h"
#include "tensor/kernels/add/add.h"
#include "tensor/kernels/axpy/axpy.h"
#include "tensor/kernels/dot/dot.h"
#include "tensor/kernels/gemv/gemv.h"
#include "tensor/kernels/mul/mul.h"
#include "tensor/kernels/rmsnorm/rmsnorm.h"
#include "tensor/kernels/rope/rope.h"
#include "tensor/kernels/silu/silu.h"
#include "tensor/kernels/softmax/softmax.h"
#include "util/cpu.h"

#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct PgMatF32 {
    const float *data;
    size_t rows;
    size_t cols;
} PgMatF32;

typedef struct PgLlamaLayer {
    const float *attn_norm;
    const float *ffn_norm;
    PgMatF32 q;
    PgMatF32 k;
    PgMatF32 v;
    PgMatF32 o;
    PgMatF32 gate;
    PgMatF32 down;
    PgMatF32 up;
} PgLlamaLayer;

struct PgLlamaModel {
    PgModelFile *file;
    PgTokenizer *tokenizer;
    PgLlamaLayer *layers;

    const float *token_embd;
    const float *output_norm;
    PgMatF32 output;

    size_t vocab;
    size_t context_length;
    size_t n_embd;
    size_t n_ff;
    size_t n_layer;
    size_t n_head;
    size_t n_head_kv;
    size_t head_dim;
    size_t kv_dim;
    float norm_eps;
    float rope_theta;

    PgGemvDSP gemv;
    PgDotDSP dot;
    PgAxpyDSP axpy;
    PgRmsnormDSP rmsnorm;
    PgSoftmaxDSP softmax;
    PgRopeDSP rope;
    PgSiluDSP silu;
    PgMulDSP mul;
    PgAddDSP add;
};

struct PgLlamaContext {
    const PgLlamaModel *model;
    size_t context_length;
    size_t pos;

    float *key_cache;
    float *value_cache;

    float *x;
    float *norm;
    float *q;
    float *k;
    float *v;
    float *attn;
    float *proj;
    float *ff_gate;
    float *ff_up;
    float *ff_hidden;
    float *scores;
    float *probs;
    float *cos;
    float *sin;
    float *logits;
};

typedef struct PgSampleCandidate {
    int32_t id;
    float logit;
    double prob;
} PgSampleCandidate;

struct PgSampler {
    PgSamplerParams params;
    uint64_t state[4];
    PgSampleCandidate *candidates;
    size_t capacity;
};

static void set_err(char *err, size_t err_len, const char *fmt, ...)
{
    va_list ap;

    if (!err || err_len == 0)
        return;
    va_start(ap, fmt);
    vsnprintf(err, err_len, fmt, ap);
    va_end(ap);
}

static bool checked_mul_size(size_t a, size_t b, size_t *out)
{
    if (a != 0 && b > SIZE_MAX / a)
        return false;
    *out = a * b;
    return true;
}

static int string_view_eq(PgStringView s, const char *lit)
{
    size_t n = strlen(lit);
    return s.len == n && memcmp(s.data, lit, n) == 0;
}

static int metadata_u64(const PgModelFile *file, const char *key,
                        uint64_t *out, char *err, size_t err_len)
{
    const PgMetadataEntry *entry =
        pg_model_file_find_metadata(file, key, strlen(key));

    if (!entry || pg_metadata_as_u64(&entry->value, out) != 0) {
        set_err(err, err_len, "missing or invalid metadata '%s'", key);
        return -1;
    }
    return 0;
}

static int metadata_u64_default(const PgModelFile *file, const char *key,
                                uint64_t def, uint64_t *out)
{
    const PgMetadataEntry *entry =
        pg_model_file_find_metadata(file, key, strlen(key));

    if (!entry) {
        *out = def;
        return 0;
    }
    return pg_metadata_as_u64(&entry->value, out);
}

static int metadata_f64_default(const PgModelFile *file, const char *key,
                                double def, double *out)
{
    const PgMetadataEntry *entry =
        pg_model_file_find_metadata(file, key, strlen(key));

    if (!entry) {
        *out = def;
        return 0;
    }
    return pg_metadata_as_f64(&entry->value, out);
}

static int size_from_u64(uint64_t v, size_t *out)
{
    if (v > SIZE_MAX)
        return -1;
    *out = (size_t)v;
    return 0;
}

static int need_arch_llama(const PgModelFile *file, char *err, size_t err_len)
{
    const PgMetadataEntry *entry =
        pg_model_file_find_metadata(file, "general.architecture",
                                    strlen("general.architecture"));
    PgStringView arch;

    if (!entry || pg_metadata_as_string(&entry->value, &arch) != 0 ||
        !string_view_eq(arch, "llama")) {
        set_err(err, err_len, "only GGUF llama architecture is supported");
        return -1;
    }
    return 0;
}

static int tensor_is_f32_vec(const PgTensorView *t, size_t n)
{
    return t && t->type == PG_TENSOR_TYPE_F32 && t->n_dims == 1 &&
           t->dims[0] == n;
}

static int tensor_is_f32_mat(const PgTensorView *t, size_t rows, size_t cols)
{
    return t && t->type == PG_TENSOR_TYPE_F32 && t->n_dims == 2 &&
           t->dims[0] == cols && t->dims[1] == rows;
}

static const float *load_vec(const PgModelFile *file, const char *name,
                             size_t n, char *err, size_t err_len)
{
    const PgTensorView *t = pg_model_file_find_tensor(file, name, strlen(name));

    if (!tensor_is_f32_vec(t, n)) {
        set_err(err, err_len, "missing or unsupported f32 vector tensor '%s'", name);
        return NULL;
    }
    return (const float *)t->data;
}

static int load_mat(const PgModelFile *file, const char *name,
                    size_t rows, size_t cols, PgMatF32 *mat,
                    char *err, size_t err_len)
{
    const PgTensorView *t = pg_model_file_find_tensor(file, name, strlen(name));

    if (!tensor_is_f32_mat(t, rows, cols)) {
        set_err(err, err_len,
                "missing or unsupported f32 matrix tensor '%s' (%zux%zu)",
                name, rows, cols);
        return -1;
    }
    mat->data = (const float *)t->data;
    mat->rows = rows;
    mat->cols = cols;
    return 0;
}

static int layer_tensor_name(char *buf, size_t buf_len,
                             size_t layer, const char *suffix)
{
    int n = snprintf(buf, buf_len, "blk.%zu.%s", layer, suffix);
    return n > 0 && (size_t)n < buf_len ? 0 : -1;
}

static int load_layer(PgLlamaModel *m, size_t idx, char *err, size_t err_len)
{
    PgLlamaLayer *l = &m->layers[idx];
    char name[128];

    if (layer_tensor_name(name, sizeof(name), idx, "attn_norm.weight") != 0)
        return -1;
    l->attn_norm = load_vec(m->file, name, m->n_embd, err, err_len);
    if (!l->attn_norm)
        return -1;

    if (layer_tensor_name(name, sizeof(name), idx, "ffn_norm.weight") != 0)
        return -1;
    l->ffn_norm = load_vec(m->file, name, m->n_embd, err, err_len);
    if (!l->ffn_norm)
        return -1;

    if (layer_tensor_name(name, sizeof(name), idx, "attn_q.weight") != 0 ||
        load_mat(m->file, name, m->n_embd, m->n_embd, &l->q, err, err_len) != 0)
        return -1;
    if (layer_tensor_name(name, sizeof(name), idx, "attn_k.weight") != 0 ||
        load_mat(m->file, name, m->kv_dim, m->n_embd, &l->k, err, err_len) != 0)
        return -1;
    if (layer_tensor_name(name, sizeof(name), idx, "attn_v.weight") != 0 ||
        load_mat(m->file, name, m->kv_dim, m->n_embd, &l->v, err, err_len) != 0)
        return -1;
    if (layer_tensor_name(name, sizeof(name), idx, "attn_output.weight") != 0 ||
        load_mat(m->file, name, m->n_embd, m->n_embd, &l->o, err, err_len) != 0)
        return -1;
    if (layer_tensor_name(name, sizeof(name), idx, "ffn_gate.weight") != 0 ||
        load_mat(m->file, name, m->n_ff, m->n_embd, &l->gate, err, err_len) != 0)
        return -1;
    if (layer_tensor_name(name, sizeof(name), idx, "ffn_down.weight") != 0 ||
        load_mat(m->file, name, m->n_embd, m->n_ff, &l->down, err, err_len) != 0)
        return -1;
    if (layer_tensor_name(name, sizeof(name), idx, "ffn_up.weight") != 0 ||
        load_mat(m->file, name, m->n_ff, m->n_embd, &l->up, err, err_len) != 0)
        return -1;

    return 0;
}

static int parse_config(PgLlamaModel *m, char *err, size_t err_len)
{
    uint64_t v;
    double f;

    if (need_arch_llama(m->file, err, err_len) != 0)
        return -1;
    if (metadata_u64(m->file, "llama.context_length", &v, err, err_len) != 0 ||
        size_from_u64(v, &m->context_length) != 0)
        return -1;
    if (metadata_u64(m->file, "llama.embedding_length", &v, err, err_len) != 0 ||
        size_from_u64(v, &m->n_embd) != 0)
        return -1;
    if (metadata_u64(m->file, "llama.feed_forward_length", &v, err, err_len) != 0 ||
        size_from_u64(v, &m->n_ff) != 0)
        return -1;
    if (metadata_u64(m->file, "llama.block_count", &v, err, err_len) != 0 ||
        size_from_u64(v, &m->n_layer) != 0)
        return -1;
    if (metadata_u64(m->file, "llama.attention.head_count", &v, err, err_len) != 0 ||
        size_from_u64(v, &m->n_head) != 0)
        return -1;
    if (metadata_u64_default(m->file, "llama.attention.head_count_kv",
                             (uint64_t)m->n_head, &v) != 0 ||
        size_from_u64(v, &m->n_head_kv) != 0) {
        set_err(err, err_len, "invalid metadata 'llama.attention.head_count_kv'");
        return -1;
    }
    if (metadata_u64_default(m->file, "llama.rope.dimension_count", 0, &v) != 0) {
        set_err(err, err_len, "invalid metadata 'llama.rope.dimension_count'");
        return -1;
    }
    if (metadata_f64_default(m->file, "llama.attention.layer_norm_rms_epsilon",
                             1.0e-5, &f) != 0) {
        set_err(err, err_len,
                "invalid metadata 'llama.attention.layer_norm_rms_epsilon'");
        return -1;
    }
    m->norm_eps = (float)f;
    if (metadata_f64_default(m->file, "llama.rope.freq_base", 10000.0, &f) != 0) {
        set_err(err, err_len, "invalid metadata 'llama.rope.freq_base'");
        return -1;
    }
    m->rope_theta = (float)f;

    if (m->context_length == 0 || m->n_embd == 0 || m->n_ff == 0 ||
        m->n_layer == 0 || m->n_head == 0 || m->n_head_kv == 0 ||
        m->n_head % m->n_head_kv != 0 || m->n_embd % m->n_head != 0) {
        set_err(err, err_len, "invalid llama dimensions");
        return -1;
    }
    m->head_dim = m->n_embd / m->n_head;
    if ((m->head_dim & 1u) != 0) {
        set_err(err, err_len, "odd RoPE head dimension is not supported");
        return -1;
    }
    if (v == 0)
        v = (uint64_t)m->head_dim;
    if (v != m->head_dim) {
        set_err(err, err_len, "partial-RoPE models are not supported yet");
        return -1;
    }
    if (!checked_mul_size(m->n_head_kv, m->head_dim, &m->kv_dim)) {
        set_err(err, err_len, "invalid KV dimensions");
        return -1;
    }
    return 0;
}

static int load_tensors(PgLlamaModel *m, char *err, size_t err_len)
{
    PgMatF32 embd;
    size_t i;

    m->vocab = pg_tokenizer_vocab_size(m->tokenizer);
    if (m->vocab == 0) {
        set_err(err, err_len, "tokenizer has empty vocabulary");
        return -1;
    }
    if (load_mat(m->file, "token_embd.weight", m->vocab, m->n_embd,
                 &embd, err, err_len) != 0)
        return -1;
    m->token_embd = embd.data;

    if (load_mat(m->file, "output.weight", m->vocab, m->n_embd,
                 &m->output, NULL, 0) != 0) {
        m->output.data = m->token_embd;
        m->output.rows = m->vocab;
        m->output.cols = m->n_embd;
    }
    m->output_norm = load_vec(m->file, "output_norm.weight",
                              m->n_embd, err, err_len);
    if (!m->output_norm)
        return -1;

    m->layers = calloc(m->n_layer, sizeof(*m->layers));
    if (!m->layers) {
        set_err(err, err_len, "out of memory");
        return -1;
    }
    for (i = 0; i < m->n_layer; i++) {
        if (load_layer(m, i, err, err_len) != 0)
            return -1;
    }
    return 0;
}

PgLlamaModel *pg_llama_model_load(const char *path, char *err, size_t err_len)
{
    PgLlamaModel *m;
    unsigned cpu_flags;

    if (err && err_len)
        err[0] = '\0';
    m = calloc(1, sizeof(*m));
    if (!m) {
        set_err(err, err_len, "out of memory");
        return NULL;
    }
    m->file = pg_model_file_open(path, err, err_len);
    if (!m->file)
        goto fail;
    m->tokenizer = pg_tokenizer_from_model_file(m->file, err, err_len);
    if (!m->tokenizer)
        goto fail;
    if (parse_config(m, err, err_len) != 0)
        goto fail;
    if (load_tensors(m, err, err_len) != 0)
        goto fail;

    cpu_flags = pg_get_cpu_flags();
    pg_gemv_dsp_init(&m->gemv, cpu_flags);
    pg_dot_dsp_init(&m->dot, cpu_flags);
    pg_axpy_dsp_init(&m->axpy, cpu_flags);
    pg_rmsnorm_dsp_init(&m->rmsnorm, cpu_flags);
    pg_softmax_dsp_init(&m->softmax, cpu_flags);
    pg_rope_dsp_init(&m->rope, cpu_flags);
    pg_silu_dsp_init(&m->silu, cpu_flags);
    pg_mul_dsp_init(&m->mul, cpu_flags);
    pg_add_dsp_init(&m->add, cpu_flags);
    return m;

fail:
    pg_llama_model_free(m);
    return NULL;
}

void pg_llama_model_free(PgLlamaModel *m)
{
    if (!m)
        return;
    free(m->layers);
    pg_tokenizer_free(m->tokenizer);
    pg_model_file_free(m->file);
    free(m);
}

const PgTokenizer *pg_llama_model_tokenizer(const PgLlamaModel *m)
{
    return m ? m->tokenizer : NULL;
}

size_t pg_llama_model_vocab_size(const PgLlamaModel *m)
{
    return m ? m->vocab : 0;
}

size_t pg_llama_model_context_length(const PgLlamaModel *m)
{
    return m ? m->context_length : 0;
}

size_t pg_llama_model_embedding_dim(const PgLlamaModel *m)
{
    return m ? m->n_embd : 0;
}

size_t pg_llama_model_layer_count(const PgLlamaModel *m)
{
    return m ? m->n_layer : 0;
}

static float *alloc_f32(size_t n)
{
    return calloc(n ? n : 1, sizeof(float));
}

static int alloc_context_buffers(PgLlamaContext *ctx, char *err, size_t err_len)
{
    const PgLlamaModel *m = ctx->model;
    size_t kv_per_layer;
    size_t kv_total;
    size_t half_head = m->head_dim / 2;

    if (!checked_mul_size(ctx->context_length, m->kv_dim, &kv_per_layer) ||
        !checked_mul_size(kv_per_layer, m->n_layer, &kv_total)) {
        set_err(err, err_len, "KV cache too large");
        return -1;
    }

    ctx->key_cache = alloc_f32(kv_total);
    ctx->value_cache = alloc_f32(kv_total);
    ctx->x = alloc_f32(m->n_embd);
    ctx->norm = alloc_f32(m->n_embd);
    ctx->q = alloc_f32(m->n_embd);
    ctx->k = alloc_f32(m->kv_dim);
    ctx->v = alloc_f32(m->kv_dim);
    ctx->attn = alloc_f32(m->n_embd);
    ctx->proj = alloc_f32(m->n_embd);
    ctx->ff_gate = alloc_f32(m->n_ff);
    ctx->ff_up = alloc_f32(m->n_ff);
    ctx->ff_hidden = alloc_f32(m->n_ff);
    ctx->scores = alloc_f32(ctx->context_length);
    ctx->probs = alloc_f32(ctx->context_length);
    ctx->cos = alloc_f32(half_head);
    ctx->sin = alloc_f32(half_head);
    ctx->logits = alloc_f32(m->vocab);

    if (!ctx->key_cache || !ctx->value_cache || !ctx->x || !ctx->norm ||
        !ctx->q || !ctx->k || !ctx->v || !ctx->attn || !ctx->proj ||
        !ctx->ff_gate || !ctx->ff_up || !ctx->ff_hidden ||
        !ctx->scores || !ctx->probs || !ctx->cos || !ctx->sin || !ctx->logits) {
        set_err(err, err_len, "out of memory");
        return -1;
    }
    return 0;
}

PgLlamaContext *pg_llama_context_new(const PgLlamaModel *m,
                                     size_t context_length,
                                     char *err, size_t err_len)
{
    PgLlamaContext *ctx;

    if (err && err_len)
        err[0] = '\0';
    if (!m) {
        set_err(err, err_len, "missing model");
        return NULL;
    }
    if (context_length == 0 || context_length > m->context_length)
        context_length = m->context_length;

    ctx = calloc(1, sizeof(*ctx));
    if (!ctx) {
        set_err(err, err_len, "out of memory");
        return NULL;
    }
    ctx->model = m;
    ctx->context_length = context_length;
    if (alloc_context_buffers(ctx, err, err_len) != 0) {
        pg_llama_context_free(ctx);
        return NULL;
    }
    return ctx;
}

void pg_llama_context_free(PgLlamaContext *ctx)
{
    if (!ctx)
        return;
    free(ctx->key_cache);
    free(ctx->value_cache);
    free(ctx->x);
    free(ctx->norm);
    free(ctx->q);
    free(ctx->k);
    free(ctx->v);
    free(ctx->attn);
    free(ctx->proj);
    free(ctx->ff_gate);
    free(ctx->ff_up);
    free(ctx->ff_hidden);
    free(ctx->scores);
    free(ctx->probs);
    free(ctx->cos);
    free(ctx->sin);
    free(ctx->logits);
    free(ctx);
}

static float *cache_ptr(float *base, const PgLlamaContext *ctx,
                        size_t layer, size_t pos)
{
    const PgLlamaModel *m = ctx->model;
    return base + (layer * ctx->context_length + pos) * m->kv_dim;
}

static int eval_layer(PgLlamaContext *ctx, size_t layer_idx)
{
    const PgLlamaModel *m = ctx->model;
    const PgLlamaLayer *l = &m->layers[layer_idx];
    float *k_slot;
    float *v_slot;
    float scale = 1.0f / sqrtf((float)m->head_dim);
    size_t group = m->n_head / m->n_head_kv;
    size_t h;

    m->rmsnorm.rmsnorm_f32(ctx->norm, ctx->x, l->attn_norm,
                           m->n_embd, m->norm_eps);
    m->gemv.sgemv(l->q.rows, l->q.cols, l->q.data, l->q.cols,
                  ctx->norm, ctx->q);
    m->gemv.sgemv(l->k.rows, l->k.cols, l->k.data, l->k.cols,
                  ctx->norm, ctx->k);
    m->gemv.sgemv(l->v.rows, l->v.cols, l->v.data, l->v.cols,
                  ctx->norm, ctx->v);

    m->rope.apply(ctx->q, ctx->q, ctx->cos, ctx->sin, 1,
                  m->n_head, m->head_dim, PG_ROPE_INTERLEAVED);
    m->rope.apply(ctx->k, ctx->k, ctx->cos, ctx->sin, 1,
                  m->n_head_kv, m->head_dim, PG_ROPE_INTERLEAVED);

    k_slot = cache_ptr(ctx->key_cache, ctx, layer_idx, ctx->pos);
    v_slot = cache_ptr(ctx->value_cache, ctx, layer_idx, ctx->pos);
    memcpy(k_slot, ctx->k, m->kv_dim * sizeof(float));
    memcpy(v_slot, ctx->v, m->kv_dim * sizeof(float));

    memset(ctx->attn, 0, m->n_embd * sizeof(float));
    for (h = 0; h < m->n_head; h++) {
        size_t kv_h = h / group;
        const float *qh = ctx->q + h * m->head_dim;
        float *out_h = ctx->attn + h * m->head_dim;
        size_t t;

        for (t = 0; t <= ctx->pos; t++) {
            const float *kh = cache_ptr(ctx->key_cache, ctx, layer_idx, t) +
                              kv_h * m->head_dim;
            ctx->scores[t] = m->dot.dot_f32(qh, kh, m->head_dim) * scale;
        }
        m->softmax.softmax_f32(ctx->scores, ctx->probs, ctx->pos + 1);
        for (t = 0; t <= ctx->pos; t++) {
            const float *vh = cache_ptr(ctx->value_cache, ctx, layer_idx, t) +
                              kv_h * m->head_dim;
            m->axpy.axpy_f32(ctx->probs[t], vh, out_h, m->head_dim);
        }
    }

    m->gemv.sgemv(l->o.rows, l->o.cols, l->o.data, l->o.cols,
                  ctx->attn, ctx->proj);
    m->add.add_f32(ctx->x, ctx->proj, ctx->norm, m->n_embd);
    memcpy(ctx->x, ctx->norm, m->n_embd * sizeof(float));

    m->rmsnorm.rmsnorm_f32(ctx->norm, ctx->x, l->ffn_norm,
                           m->n_embd, m->norm_eps);
    m->gemv.sgemv(l->gate.rows, l->gate.cols, l->gate.data, l->gate.cols,
                  ctx->norm, ctx->ff_gate);
    m->gemv.sgemv(l->up.rows, l->up.cols, l->up.data, l->up.cols,
                  ctx->norm, ctx->ff_up);
    m->silu.silu_f32(ctx->ff_gate, ctx->ff_hidden, m->n_ff);
    m->mul.mul_f32(ctx->ff_hidden, ctx->ff_up, ctx->ff_gate, m->n_ff);
    m->gemv.sgemv(l->down.rows, l->down.cols, l->down.data, l->down.cols,
                  ctx->ff_gate, ctx->proj);
    m->add.add_f32(ctx->x, ctx->proj, ctx->norm, m->n_embd);
    memcpy(ctx->x, ctx->norm, m->n_embd * sizeof(float));
    return 0;
}

int pg_llama_eval_token(PgLlamaContext *ctx, int32_t token,
                        const float **logits,
                        char *err, size_t err_len)
{
    const PgLlamaModel *m;
    int32_t pos_i32;
    size_t i;

    if (err && err_len)
        err[0] = '\0';
    if (!ctx || !ctx->model) {
        set_err(err, err_len, "missing context");
        return -1;
    }
    m = ctx->model;
    if (token < 0 || (size_t)token >= m->vocab) {
        set_err(err, err_len, "token id out of range");
        return -1;
    }
    if (ctx->pos >= ctx->context_length) {
        set_err(err, err_len, "context length exceeded");
        return -1;
    }
    if (ctx->pos > (size_t)INT32_MAX) {
        set_err(err, err_len, "position exceeds RoPE int32 range");
        return -1;
    }

    memcpy(ctx->x, m->token_embd + (size_t)token * m->n_embd,
           m->n_embd * sizeof(float));
    pos_i32 = (int32_t)ctx->pos;
    m->rope.cache(ctx->cos, ctx->sin, &pos_i32, 1,
                  m->head_dim, m->rope_theta);

    for (i = 0; i < m->n_layer; i++) {
        if (eval_layer(ctx, i) != 0) {
            set_err(err, err_len, "layer evaluation failed");
            return -1;
        }
    }

    m->rmsnorm.rmsnorm_f32(ctx->norm, ctx->x, m->output_norm,
                           m->n_embd, m->norm_eps);
    m->gemv.sgemv(m->output.rows, m->output.cols, m->output.data,
                  m->output.cols, ctx->norm, ctx->logits);

    ctx->pos++;
    if (logits)
        *logits = ctx->logits;
    return 0;
}

const float *pg_llama_logits(const PgLlamaContext *ctx)
{
    return ctx ? ctx->logits : NULL;
}

size_t pg_llama_context_position(const PgLlamaContext *ctx)
{
    return ctx ? ctx->pos : 0;
}

static int candidate_cmp_desc(const void *a, const void *b)
{
    const PgSampleCandidate *ca = a;
    const PgSampleCandidate *cb = b;

    if (ca->logit < cb->logit)
        return 1;
    if (ca->logit > cb->logit)
        return -1;
    if (ca->id > cb->id)
        return 1;
    if (ca->id < cb->id)
        return -1;
    return 0;
}

int32_t pg_llama_sample_greedy(const float *logits, size_t n_logits)
{
    int32_t best = -1;
    float best_v = -INFINITY;
    size_t i;

    if (!logits || n_logits == 0 || n_logits > (size_t)INT32_MAX)
        return -1;
    for (i = 0; i < n_logits; i++) {
        float v = logits[i];
        if (!isfinite(v))
            continue;
        if (best < 0 || v > best_v) {
            best = (int32_t)i;
            best_v = v;
        }
    }
    return best;
}

static uint64_t splitmix64(uint64_t *x)
{
    uint64_t z;

    *x += 0x9e3779b97f4a7c15ull;
    z = *x;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

static void sampler_seed(PgSampler *sampler, uint64_t seed)
{
    uint64_t s = seed ? seed : 0x9e3779b97f4a7c15ull;
    size_t i;

    for (i = 0; i < 4; i++)
        sampler->state[i] = splitmix64(&s);
}

static uint64_t rotl64(uint64_t x, int k)
{
    return (x << k) | (x >> (64 - k));
}

static uint64_t sampler_rng(PgSampler *sampler)
{
    uint64_t *s = sampler->state;
    uint64_t result = rotl64(s[1] * 5, 7) * 9;
    uint64_t t = s[1] << 17;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl64(s[3], 45);
    return result;
}

static double sampler_uniform(PgSampler *sampler)
{
    return (double)(sampler_rng(sampler) >> 11) * (1.0 / 9007199254740992.0);
}

PgSampler *pg_sampler_new(const PgSamplerParams *params, char *err, size_t err_len)
{
    PgSamplerParams p;
    PgSampler *sampler;

    if (err && err_len)
        err[0] = '\0';
    if (params) {
        p = *params;
    } else {
        p.temperature = 0.0f;
        p.top_k = 0;
        p.top_p = 1.0f;
        p.seed = 0;
    }
    if (!isfinite(p.temperature) || p.temperature < 0.0f) {
        set_err(err, err_len, "invalid sampler temperature");
        return NULL;
    }
    if (!isfinite(p.top_p) || p.top_p < 0.0f) {
        set_err(err, err_len, "invalid sampler top-p");
        return NULL;
    }

    sampler = calloc(1, sizeof(*sampler));
    if (!sampler) {
        set_err(err, err_len, "out of memory");
        return NULL;
    }
    sampler->params = p;
    sampler_seed(sampler, p.seed);
    return sampler;
}

void pg_sampler_free(PgSampler *sampler)
{
    if (!sampler)
        return;
    free(sampler->candidates);
    free(sampler);
}

static int sampler_reserve(PgSampler *sampler, size_t n, char *err, size_t err_len)
{
    PgSampleCandidate *next;

    if (n <= sampler->capacity)
        return 0;
    next = realloc(sampler->candidates, n * sizeof(*next));
    if (!next) {
        set_err(err, err_len, "out of memory");
        return -1;
    }
    sampler->candidates = next;
    sampler->capacity = n;
    return 0;
}

int32_t pg_llama_sample(const float *logits, size_t n_logits,
                        PgSampler *sampler, char *err, size_t err_len)
{
    PgSampleCandidate *c;
    size_t count = 0;
    size_t keep;
    size_t i;
    double max_logit;
    double sum = 0.0;
    double mass;
    double r;

    if (err && err_len)
        err[0] = '\0';
    if (!sampler)
        return pg_llama_sample_greedy(logits, n_logits);
    if (!logits || n_logits == 0 || n_logits > (size_t)INT32_MAX) {
        set_err(err, err_len, "invalid logits");
        return -1;
    }
    if (sampler->params.temperature <= 0.0f)
        return pg_llama_sample_greedy(logits, n_logits);
    if (sampler_reserve(sampler, n_logits, err, err_len) != 0)
        return -1;

    c = sampler->candidates;
    for (i = 0; i < n_logits; i++) {
        if (!isfinite(logits[i]))
            continue;
        c[count].id = (int32_t)i;
        c[count].logit = logits[i];
        c[count].prob = 0.0;
        count++;
    }
    if (count == 0) {
        set_err(err, err_len, "no finite logits to sample");
        return -1;
    }

    qsort(c, count, sizeof(*c), candidate_cmp_desc);
    keep = count;
    if (sampler->params.top_k > 0 && sampler->params.top_k < keep)
        keep = sampler->params.top_k;

    max_logit = (double)c[0].logit / (double)sampler->params.temperature;
    for (i = 0; i < keep; i++) {
        double x = (double)c[i].logit / (double)sampler->params.temperature;
        c[i].prob = exp(x - max_logit);
        sum += c[i].prob;
    }
    if (sum <= 0.0 || !isfinite(sum)) {
        set_err(err, err_len, "invalid sampling probability mass");
        return -1;
    }
    for (i = 0; i < keep; i++)
        c[i].prob /= sum;

    if (sampler->params.top_p > 0.0f && sampler->params.top_p < 1.0f) {
        double p = 0.0;
        size_t nucleus = 0;

        while (nucleus < keep) {
            p += c[nucleus].prob;
            nucleus++;
            if (p >= (double)sampler->params.top_p)
                break;
        }
        keep = nucleus ? nucleus : 1;
    }

    mass = 0.0;
    for (i = 0; i < keep; i++)
        mass += c[i].prob;
    if (mass <= 0.0 || !isfinite(mass)) {
        set_err(err, err_len, "invalid sampling probability mass");
        return -1;
    }

    r = sampler_uniform(sampler) * mass;
    for (i = 0; i < keep; i++) {
        if (r <= c[i].prob)
            return c[i].id;
        r -= c[i].prob;
    }
    return c[keep - 1].id;
}
