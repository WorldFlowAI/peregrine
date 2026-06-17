/*
 * peregrine - Llama decoder smoke tests
 */
#include "peregrine/llama.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int failures;

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); \
        failures++; \
    } \
} while (0)

typedef struct TensorSpec {
    const char *name;
    uint32_t n_dims;
    uint64_t dims[2];
} TensorSpec;

static const TensorSpec tensors[] = {
    { "token_embd.weight", 2, { 4, 4 } },
    { "output_norm.weight", 1, { 4, 0 } },
    { "output.weight", 2, { 4, 4 } },
    { "blk.0.attn_q.weight", 2, { 4, 4 } },
    { "blk.0.attn_k.weight", 2, { 4, 4 } },
    { "blk.0.attn_v.weight", 2, { 4, 4 } },
    { "blk.0.attn_output.weight", 2, { 4, 4 } },
    { "blk.0.attn_norm.weight", 1, { 4, 0 } },
    { "blk.0.ffn_gate.weight", 2, { 4, 4 } },
    { "blk.0.ffn_down.weight", 2, { 4, 4 } },
    { "blk.0.ffn_up.weight", 2, { 4, 4 } },
    { "blk.0.ffn_norm.weight", 1, { 4, 0 } },
};

static void write_all(FILE *f, const void *ptr, size_t n)
{
    if (fwrite(ptr, 1, n, f) != n) {
        fprintf(stderr, "fwrite failed: %s\n", strerror(errno));
        exit(2);
    }
}

static void write_u8(FILE *f, uint8_t v)
{
    write_all(f, &v, sizeof(v));
}

static void write_u32(FILE *f, uint32_t v)
{
    unsigned char b[4];
    b[0] = (unsigned char)v;
    b[1] = (unsigned char)(v >> 8);
    b[2] = (unsigned char)(v >> 16);
    b[3] = (unsigned char)(v >> 24);
    write_all(f, b, sizeof(b));
}

static void write_u64(FILE *f, uint64_t v)
{
    unsigned char b[8];
    b[0] = (unsigned char)v;
    b[1] = (unsigned char)(v >> 8);
    b[2] = (unsigned char)(v >> 16);
    b[3] = (unsigned char)(v >> 24);
    b[4] = (unsigned char)(v >> 32);
    b[5] = (unsigned char)(v >> 40);
    b[6] = (unsigned char)(v >> 48);
    b[7] = (unsigned char)(v >> 56);
    write_all(f, b, sizeof(b));
}

static void write_f32(FILE *f, float v)
{
    uint32_t bits;

    memcpy(&bits, &v, sizeof(bits));
    write_u32(f, bits);
}

static void write_str(FILE *f, const char *s)
{
    write_u64(f, strlen(s));
    write_all(f, s, strlen(s));
}

static void write_meta_u32(FILE *f, const char *key, uint32_t v)
{
    write_str(f, key);
    write_u32(f, 4);
    write_u32(f, v);
}

static void write_meta_f32(FILE *f, const char *key, float v)
{
    write_str(f, key);
    write_u32(f, 6);
    write_f32(f, v);
}

static void write_meta_string(FILE *f, const char *key, const char *value)
{
    write_str(f, key);
    write_u32(f, 8);
    write_str(f, value);
}

static void write_meta_tokens(FILE *f)
{
    static const char *tokens[] = { "<unk>", "<s>", "</s>", "x" };
    size_t i;

    write_str(f, "tokenizer.ggml.tokens");
    write_u32(f, 9);
    write_u32(f, 8);
    write_u64(f, sizeof(tokens) / sizeof(tokens[0]));
    for (i = 0; i < sizeof(tokens) / sizeof(tokens[0]); i++)
        write_str(f, tokens[i]);
}

static size_t tensor_float_count(const TensorSpec *t)
{
    size_t n = 1;
    uint32_t i;

    for (i = 0; i < t->n_dims; i++)
        n *= (size_t)t->dims[i];
    return n;
}

static int is_norm_tensor(const char *name)
{
    return !strcmp(name, "output_norm.weight") ||
           !strcmp(name, "blk.0.attn_norm.weight") ||
           !strcmp(name, "blk.0.ffn_norm.weight");
}

static float tensor_value(const TensorSpec *t, size_t row, size_t col)
{
    if (!strcmp(t->name, "token_embd.weight"))
        return row == 1 && col == 0 ? 1.0f : 0.0f;
    if (!strcmp(t->name, "output.weight")) {
        if (col != 0)
            return 0.0f;
        if (row == 1)
            return 0.1f;
        if (row == 2)
            return 0.2f;
        if (row == 3)
            return 1.0f;
    }
    if (is_norm_tensor(t->name))
        return 1.0f;
    return 0.0f;
}

static void write_tensor_data(FILE *f, const TensorSpec *t)
{
    size_t rows = t->n_dims == 2 ? (size_t)t->dims[1] : 1;
    size_t cols = (size_t)t->dims[0];
    size_t r, c;

    for (r = 0; r < rows; r++) {
        for (c = 0; c < cols; c++)
            write_f32(f, tensor_value(t, r, c));
    }
}

static char *make_temp_path(void)
{
    char tmpl[256];
    int fd;
    char *path;

    snprintf(tmpl, sizeof(tmpl), "/tmp/peregrine-llama-XXXXXX");
    fd = mkstemp(tmpl);
    if (fd < 0) {
        fprintf(stderr, "mkstemp failed: %s\n", strerror(errno));
        exit(2);
    }
    close(fd);
    path = malloc(strlen(tmpl) + 1);
    if (!path)
        exit(2);
    strcpy(path, tmpl);
    return path;
}

static void write_tiny_llama_gguf(const char *path)
{
    FILE *f = fopen(path, "wb");
    uint64_t offset = 0;
    size_t i;
    long pos;
    unsigned pad;

    if (!f) {
        fprintf(stderr, "%s: %s\n", path, strerror(errno));
        exit(2);
    }

    write_all(f, "GGUF", 4);
    write_u32(f, 3);
    write_u64(f, sizeof(tensors) / sizeof(tensors[0]));
    write_u64(f, 13);

    write_meta_string(f, "general.architecture", "llama");
    write_meta_string(f, "tokenizer.ggml.model", "llama");
    write_meta_tokens(f);
    write_meta_u32(f, "tokenizer.ggml.unknown_token_id", 0);
    write_meta_u32(f, "tokenizer.ggml.bos_token_id", 1);
    write_meta_u32(f, "tokenizer.ggml.eos_token_id", 2);
    write_meta_u32(f, "llama.context_length", 4);
    write_meta_u32(f, "llama.embedding_length", 4);
    write_meta_u32(f, "llama.feed_forward_length", 4);
    write_meta_u32(f, "llama.block_count", 1);
    write_meta_u32(f, "llama.attention.head_count", 1);
    write_meta_u32(f, "llama.rope.dimension_count", 4);
    write_meta_f32(f, "llama.attention.layer_norm_rms_epsilon", 1.0e-5f);

    for (i = 0; i < sizeof(tensors) / sizeof(tensors[0]); i++) {
        uint32_t d;

        write_str(f, tensors[i].name);
        write_u32(f, tensors[i].n_dims);
        for (d = 0; d < tensors[i].n_dims; d++)
            write_u64(f, tensors[i].dims[d]);
        write_u32(f, 0);
        write_u64(f, offset);
        offset += (uint64_t)(tensor_float_count(&tensors[i]) * sizeof(float));
    }

    pos = ftell(f);
    if (pos < 0)
        exit(2);
    pad = (unsigned)((32 - ((unsigned)pos & 31u)) & 31u);
    while (pad--)
        write_u8(f, 0);

    for (i = 0; i < sizeof(tensors) / sizeof(tensors[0]); i++)
        write_tensor_data(f, &tensors[i]);
    fclose(f);
}

int main(void)
{
    char err[256];
    char *path = make_temp_path();
    PgLlamaModel *model;
    PgLlamaContext *ctx;
    const float *logits = NULL;
    int32_t next;

    write_tiny_llama_gguf(path);
    model = pg_llama_model_load(path, err, sizeof(err));
    CHECK(model != NULL);
    if (!model) {
        fprintf(stderr, "load failed: %s\n", err);
        goto done;
    }
    CHECK(pg_llama_model_vocab_size(model) == 4);
    CHECK(pg_llama_model_embedding_dim(model) == 4);
    CHECK(pg_llama_model_layer_count(model) == 1);

    ctx = pg_llama_context_new(model, 0, err, sizeof(err));
    CHECK(ctx != NULL);
    if (!ctx) {
        fprintf(stderr, "context failed: %s\n", err);
        goto free_model;
    }
    CHECK(pg_llama_eval_token(ctx, 1, &logits, err, sizeof(err)) == 0);
    CHECK(logits != NULL);
    CHECK(pg_llama_context_position(ctx) == 1);
    next = pg_llama_sample_greedy(logits, pg_llama_model_vocab_size(model));
    CHECK(next == 3);
    CHECK(logits && logits[3] > logits[2]);

    pg_llama_context_free(ctx);
free_model:
    pg_llama_model_free(model);
done:
    unlink(path);
    free(path);
    return failures ? 1 : 0;
}
