/*
 * peregrine - command line front-end (the `ffmpeg` analog).
 *
 * v0.1 reports build/capability info and is the home for the inference
 * pipeline (load -> tokenize -> eval -> sample) as it lands. Subcommands are
 * intentionally tiny stubs so the binary always builds and runs.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "peregrine/llama.h"
#include "peregrine/model.h"
#include "peregrine/tokenizer.h"
#include "peregrine/version.h"
#include "util/cpu.h"
#include "util/thread.h"

static int cmd_info(void)
{
    char buf[128];
    unsigned flags = pg_get_cpu_flags();
    printf("peregrine %s\n", PEREGRINE_VERSION_STRING);
    printf("cpu features: %s\n", pg_cpu_flags_str(flags, buf, sizeof buf));
    return 0;
}

static void print_tensor_name(const PgTensorView *t)
{
    fwrite(t->name, 1, t->name_len, stdout);
}

static int cmd_inspect(int argc, char **argv)
{
    const char *path = NULL;
    PgModelFile *model;
    char err[256];
    size_t i;

    if (argc == 3) {
        path = argv[2];
    } else if (argc == 4 && !strcmp(argv[2], "-m")) {
        path = argv[3];
    } else {
        fprintf(stderr, "usage: %s inspect [-m] <model.gguf|model.safetensors>\n", argv[0]);
        return 2;
    }

    model = pg_model_file_open(path, err, sizeof(err));
    if (!model) {
        fprintf(stderr, "peregrine: %s\n", err);
        return 1;
    }

    printf("%s: %s, %zu tensors\n",
           path,
           pg_model_format_name(pg_model_file_format(model)),
           pg_model_file_tensor_count(model));
    for (i = 0; i < pg_model_file_tensor_count(model); i++) {
        const PgTensorView *t = pg_model_file_tensor(model, i);
        unsigned d;

        printf("  ");
        print_tensor_name(t);
        printf("  %s  [", pg_tensor_type_name(t->type));
        for (d = 0; d < t->n_dims; d++) {
            if (d)
                printf(", ");
            printf("%llu", (unsigned long long)t->dims[d]);
        }
        printf("]  %zu bytes\n", t->nbytes);
    }

    pg_model_file_free(model);
    return 0;
}

static int parse_size_arg(const char *s, size_t *out)
{
    char *end = NULL;
    unsigned long v;

    if (!s || !*s)
        return -1;
    v = strtoul(s, &end, 10);
    if (!end || *end != '\0')
        return -1;
    *out = (size_t)v;
    return 0;
}

static int parse_u64_arg(const char *s, uint64_t *out)
{
    char *end = NULL;
    unsigned long long v;

    if (!s || !*s)
        return -1;
    v = strtoull(s, &end, 0);
    if (!end || *end != '\0')
        return -1;
    *out = (uint64_t)v;
    return 0;
}

static int parse_float_arg(const char *s, float *out)
{
    char *end = NULL;
    double v;

    if (!s || !*s)
        return -1;
    v = strtod(s, &end);
    if (!end || *end != '\0')
        return -1;
    *out = (float)v;
    return 0;
}

static double now_sec(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static void print_escaped(const char *s, size_t n)
{
    size_t i;

    putchar('"');
    for (i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];

        if (c == '\\' || c == '"') {
            putchar('\\');
            putchar(c);
        } else if (c == '\n') {
            fputs("\\n", stdout);
        } else if (c == '\r') {
            fputs("\\r", stdout);
        } else if (c == '\t') {
            fputs("\\t", stdout);
        } else if (c < 0x20 || c == 0x7f) {
            printf("\\x%02x", c);
        } else {
            putchar(c);
        }
    }
    putchar('"');
}

static int cmd_tokenize(int argc, char **argv)
{
    const char *path = NULL;
    const char *prompt = NULL;
    int add_bos = 1;
    int add_eos = 0;
    int ids_only = 0;
    PgModelFile *model = NULL;
    PgTokenizer *tok = NULL;
    PgTokenBuffer tokens = { 0 };
    char err[256];
    size_t i;
    int rc = 1;

    for (i = 2; i < (size_t)argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < (size_t)argc) {
            path = argv[++i];
        } else if (!strcmp(argv[i], "-p") && i + 1 < (size_t)argc) {
            prompt = argv[++i];
        } else if (!strcmp(argv[i], "--no-bos")) {
            add_bos = 0;
        } else if (!strcmp(argv[i], "--eos")) {
            add_eos = 1;
        } else if (!strcmp(argv[i], "--ids-only")) {
            ids_only = 1;
        } else {
            fprintf(stderr,
                    "usage: %s tokenize -m <model.gguf> -p <prompt> [--no-bos] [--eos] [--ids-only]\n",
                    argv[0]);
            return 2;
        }
    }
    if (!path || !prompt) {
        fprintf(stderr,
                "usage: %s tokenize -m <model.gguf> -p <prompt> [--no-bos] [--eos] [--ids-only]\n",
                argv[0]);
        return 2;
    }

    model = pg_model_file_open(path, err, sizeof(err));
    if (!model) {
        fprintf(stderr, "peregrine: %s\n", err);
        goto done;
    }
    tok = pg_tokenizer_from_model_file(model, err, sizeof(err));
    if (!tok) {
        fprintf(stderr, "peregrine: %s\n", err);
        goto done;
    }
    if (pg_tokenizer_encode(tok, prompt, add_bos, add_eos, &tokens) != 0) {
        fprintf(stderr, "peregrine: failed to tokenize prompt\n");
        goto done;
    }

    if (ids_only) {
        for (i = 0; i < tokens.count; i++) {
            if (i)
                putchar(' ');
            printf("%d", tokens.data[i]);
        }
        putchar('\n');
    } else {
        printf("count: %zu\nids:", tokens.count);
        for (i = 0; i < tokens.count; i++)
            printf(" %d", tokens.data[i]);
        putchar('\n');
        for (i = 0; i < tokens.count; i++) {
            char text[4096];
            size_t written = 0;

            printf("%4zu  %8d  ", i, tokens.data[i]);
            if (pg_tokenizer_decode_token(tok, tokens.data[i],
                                          text, sizeof(text), &written) == 0) {
                print_escaped(text, written);
            } else {
                fputs("<decode-error>", stdout);
            }
            putchar('\n');
        }
    }
    rc = 0;

done:
    pg_token_buffer_free(&tokens);
    pg_tokenizer_free(tok);
    pg_model_file_free(model);
    return rc;
}

static int token_selected(const int32_t *ids, size_t n, int32_t id)
{
    size_t i;

    for (i = 0; i < n; i++) {
        if (ids[i] == id)
            return 1;
    }
    return 0;
}

static int cmd_logits(int argc, char **argv)
{
    const char *path = NULL;
    const char *prompt = NULL;
    size_t top_n = 16;
    int add_bos = 1;
    PgLlamaModel *model = NULL;
    PgLlamaContext *ctx = NULL;
    const PgTokenizer *tok;
    PgTokenBuffer prompt_tokens = { 0 };
    const float *logits = NULL;
    int32_t *top_ids = NULL;
    char err[256];
    size_t i;
    int rc = 1;

    for (i = 2; i < (size_t)argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < (size_t)argc) {
            path = argv[++i];
        } else if (!strcmp(argv[i], "-p") && i + 1 < (size_t)argc) {
            prompt = argv[++i];
        } else if (!strcmp(argv[i], "--top") && i + 1 < (size_t)argc) {
            if (parse_size_arg(argv[++i], &top_n) != 0 || top_n == 0) {
                fprintf(stderr, "peregrine: invalid --top value\n");
                return 2;
            }
        } else if (!strcmp(argv[i], "--no-bos")) {
            add_bos = 0;
        } else {
            fprintf(stderr,
                    "usage: %s logits -m <model.gguf> -p <prompt> [--top n] [--no-bos]\n",
                    argv[0]);
            return 2;
        }
    }
    if (!path || !prompt) {
        fprintf(stderr,
                "usage: %s logits -m <model.gguf> -p <prompt> [--top n] [--no-bos]\n",
                argv[0]);
        return 2;
    }

    model = pg_llama_model_load(path, err, sizeof(err));
    if (!model) {
        fprintf(stderr, "peregrine: %s\n", err);
        goto done;
    }
    ctx = pg_llama_context_new(model, 0, err, sizeof(err));
    if (!ctx) {
        fprintf(stderr, "peregrine: %s\n", err);
        goto done;
    }
    tok = pg_llama_model_tokenizer(model);
    if (pg_tokenizer_encode(tok, prompt, add_bos, 0, &prompt_tokens) != 0 ||
        prompt_tokens.count == 0) {
        fprintf(stderr, "peregrine: failed to tokenize prompt\n");
        goto done;
    }
    for (i = 0; i < prompt_tokens.count; i++) {
        if (pg_llama_eval_token(ctx, prompt_tokens.data[i],
                                &logits, err, sizeof(err)) != 0) {
            fprintf(stderr, "peregrine: %s\n", err);
            goto done;
        }
    }

    if (top_n > pg_llama_model_vocab_size(model))
        top_n = pg_llama_model_vocab_size(model);
    top_ids = malloc(top_n * sizeof(*top_ids));
    if (!top_ids) {
        fprintf(stderr, "peregrine: out of memory\n");
        goto done;
    }

    printf("prompt_tokens:");
    for (i = 0; i < prompt_tokens.count; i++)
        printf(" %d", prompt_tokens.data[i]);
    printf("\nposition: %zu\n", pg_llama_context_position(ctx));
    printf("top_logits:\n");
    for (i = 0; i < top_n; i++) {
        int32_t best_id = -1;
        float best = 0.0f;
        size_t j;

        for (j = 0; j < pg_llama_model_vocab_size(model); j++) {
            if (token_selected(top_ids, i, (int32_t)j))
                continue;
            if (best_id < 0 || logits[j] > best) {
                best_id = (int32_t)j;
                best = logits[j];
            }
        }
        top_ids[i] = best_id;
        printf("%4zu  %8d  %.9g  ", i, best_id, best);
        if (best_id >= 0) {
            char text[4096];
            size_t written = 0;

            if (pg_tokenizer_decode_token(tok, best_id,
                                          text, sizeof(text), &written) == 0)
                print_escaped(text, written);
            else
                fputs("<decode-error>", stdout);
        }
        putchar('\n');
    }
    rc = 0;

done:
    free(top_ids);
    pg_token_buffer_free(&prompt_tokens);
    pg_llama_context_free(ctx);
    pg_llama_model_free(model);
    return rc;
}

static int eval_prompt_tokens(PgLlamaContext *ctx, const PgTokenBuffer *tokens,
                              const float **logits, char *err, size_t err_len)
{
    size_t i;

    for (i = 0; i < tokens->count; i++) {
        if (pg_llama_eval_token(ctx, tokens->data[i], logits, err, err_len) != 0)
            return -1;
    }
    return 0;
}

static int eval_decode_tokens(PgLlamaContext *ctx, const PgLlamaModel *model,
                              size_t n_predict, int32_t fixed_token,
                              const float **logits, double *sample_sec,
                              char *err, size_t err_len)
{
    size_t i;

    for (i = 0; i < n_predict; i++) {
        int32_t next = fixed_token;

        if (next < 0) {
            double t0 = sample_sec ? now_sec() : 0.0;

            next = pg_llama_sample_greedy(*logits,
                                          pg_llama_model_vocab_size(model));
            if (sample_sec)
                *sample_sec += now_sec() - t0;
            if (next < 0) {
                snprintf(err, err_len, "failed to sample next token");
                return -1;
            }
        }
        if (pg_llama_eval_token(ctx, next, logits, err, err_len) != 0)
            return -1;
    }
    return 0;
}

static void profile_add(PgLlamaProfile *dst, const PgLlamaProfile *src)
{
    dst->tokens += src->tokens;
    dst->token_embed_sec += src->token_embed_sec;
    dst->rope_sec += src->rope_sec;
    dst->attn_norm_sec += src->attn_norm_sec;
    dst->qkv_sec += src->qkv_sec;
    dst->attn_scores_sec += src->attn_scores_sec;
    dst->attn_softmax_sec += src->attn_softmax_sec;
    dst->attn_mix_sec += src->attn_mix_sec;
    dst->attn_output_sec += src->attn_output_sec;
    dst->attn_residual_sec += src->attn_residual_sec;
    dst->ffn_norm_sec += src->ffn_norm_sec;
    dst->ffn_gate_up_sec += src->ffn_gate_up_sec;
    dst->ffn_act_sec += src->ffn_act_sec;
    dst->ffn_down_sec += src->ffn_down_sec;
    dst->ffn_residual_sec += src->ffn_residual_sec;
    dst->output_norm_sec += src->output_norm_sec;
    dst->logits_sec += src->logits_sec;
}

static void print_profile_row(const char *name, double sec,
                              double total_sec, size_t tokens)
{
    double pct = total_sec > 0.0 ? 100.0 * sec / total_sec : 0.0;
    double us = tokens ? 1.0e6 * sec / (double)tokens : 0.0;

    printf("profile.%s_sec: %.6f\n", name, sec);
    printf("profile.%s_us_per_tok: %.2f\n", name, us);
    printf("profile.%s_pct: %.2f\n", name, pct);
}

static double profile_total_sec(const PgLlamaProfile *p, double sample_sec)
{
    return p->token_embed_sec + p->rope_sec + p->attn_norm_sec + p->qkv_sec +
           p->attn_scores_sec + p->attn_softmax_sec + p->attn_mix_sec +
           p->attn_output_sec + p->attn_residual_sec + p->ffn_norm_sec +
           p->ffn_gate_up_sec + p->ffn_act_sec + p->ffn_down_sec +
           p->ffn_residual_sec + p->output_norm_sec + p->logits_sec +
           sample_sec;
}

static void print_profile(const PgLlamaProfile *p, double sample_sec)
{
    double total = profile_total_sec(p, sample_sec);

    printf("profile.tokens: %zu\n", p->tokens);
    printf("profile.total_sec: %.6f\n", total);
    print_profile_row("token_embed", p->token_embed_sec, total, p->tokens);
    print_profile_row("rope", p->rope_sec, total, p->tokens);
    print_profile_row("attn_norm", p->attn_norm_sec, total, p->tokens);
    print_profile_row("qkv", p->qkv_sec, total, p->tokens);
    print_profile_row("attn_scores", p->attn_scores_sec, total, p->tokens);
    print_profile_row("attn_softmax", p->attn_softmax_sec, total, p->tokens);
    print_profile_row("attn_mix", p->attn_mix_sec, total, p->tokens);
    print_profile_row("attn_output", p->attn_output_sec, total, p->tokens);
    print_profile_row("attn_residual", p->attn_residual_sec, total, p->tokens);
    print_profile_row("ffn_norm", p->ffn_norm_sec, total, p->tokens);
    print_profile_row("ffn_gate_up", p->ffn_gate_up_sec, total, p->tokens);
    print_profile_row("ffn_act", p->ffn_act_sec, total, p->tokens);
    print_profile_row("ffn_down", p->ffn_down_sec, total, p->tokens);
    print_profile_row("ffn_residual", p->ffn_residual_sec, total, p->tokens);
    print_profile_row("output_norm", p->output_norm_sec, total, p->tokens);
    print_profile_row("logits", p->logits_sec, total, p->tokens);
    print_profile_row("sample", sample_sec, total, p->tokens);
}

static int cmd_bench(int argc, char **argv)
{
    const char *path = NULL;
    const char *prompt = NULL;
    size_t n_predict = 128;
    size_t repeat = 5;
    size_t threads = 0;
    int32_t fixed_token = -1;
    int warmup = 1;
    int profile = 0;
    PgLlamaModel *model = NULL;
    PgLlamaContext *ctx = NULL;
    const PgTokenizer *tok;
    PgTokenBuffer prompt_tokens = { 0 };
    const float *logits = NULL;
    char err[256];
    double load_sec;
    double tokenize_sec;
    double prompt_total = 0.0;
    double decode_total = 0.0;
    double sample_total = 0.0;
    double t0;
    PgLlamaProfile profile_total = { 0 };
    size_t i;
    int rc = 1;

    for (i = 2; i < (size_t)argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < (size_t)argc) {
            path = argv[++i];
        } else if (!strcmp(argv[i], "-p") && i + 1 < (size_t)argc) {
            prompt = argv[++i];
        } else if (!strcmp(argv[i], "-n") && i + 1 < (size_t)argc) {
            if (parse_size_arg(argv[++i], &n_predict) != 0) {
                fprintf(stderr, "peregrine: invalid -n value\n");
                return 2;
            }
        } else if ((!strcmp(argv[i], "-r") || !strcmp(argv[i], "--repeat")) &&
                   i + 1 < (size_t)argc) {
            if (parse_size_arg(argv[++i], &repeat) != 0 || repeat == 0) {
                fprintf(stderr, "peregrine: invalid repeat value\n");
                return 2;
            }
        } else if (!strcmp(argv[i], "--threads") && i + 1 < (size_t)argc) {
            if (parse_size_arg(argv[++i], &threads) != 0 || threads > 1024) {
                fprintf(stderr, "peregrine: invalid thread count\n");
                return 2;
            }
        } else if (!strcmp(argv[i], "--fixed-token") && i + 1 < (size_t)argc) {
            size_t token;

            if (parse_size_arg(argv[++i], &token) != 0 || token > (size_t)INT32_MAX) {
                fprintf(stderr, "peregrine: invalid fixed token\n");
                return 2;
            }
            fixed_token = (int32_t)token;
        } else if (!strcmp(argv[i], "--no-warmup")) {
            warmup = 0;
        } else if (!strcmp(argv[i], "--profile")) {
            profile = 1;
        } else {
            fprintf(stderr,
                    "usage: %s bench -m <model.gguf> -p <prompt> [-n tokens] [-r repeat] [--threads n] [--fixed-token id] [--profile] [--no-warmup]\n",
                    argv[0]);
            return 2;
        }
    }
    if (!path || !prompt) {
        fprintf(stderr,
                "usage: %s bench -m <model.gguf> -p <prompt> [-n tokens] [-r repeat] [--threads n] [--fixed-token id] [--profile] [--no-warmup]\n",
                argv[0]);
        return 2;
    }
    if (threads > 0 &&
        pg_global_threadpool_configure((int)threads) != 0) {
        fprintf(stderr, "peregrine: thread pool already initialized\n");
        return 2;
    }

    t0 = now_sec();
    model = pg_llama_model_load(path, err, sizeof(err));
    load_sec = now_sec() - t0;
    if (!model) {
        fprintf(stderr, "peregrine: %s\n", err);
        goto done;
    }

    tok = pg_llama_model_tokenizer(model);
    t0 = now_sec();
    if (pg_tokenizer_encode(tok, prompt, 1, 0, &prompt_tokens) != 0 ||
        prompt_tokens.count == 0) {
        fprintf(stderr, "peregrine: failed to tokenize prompt\n");
        goto done;
    }
    tokenize_sec = now_sec() - t0;
    if (prompt_tokens.count + n_predict > pg_llama_model_context_length(model)) {
        fprintf(stderr, "peregrine: benchmark exceeds model context length\n");
        goto done;
    }
    if (fixed_token >= 0 && (size_t)fixed_token >= pg_llama_model_vocab_size(model)) {
        fprintf(stderr, "peregrine: fixed token out of range\n");
        goto done;
    }

    if (warmup) {
        ctx = pg_llama_context_new(model, 0, err, sizeof(err));
        if (!ctx) {
            fprintf(stderr, "peregrine: %s\n", err);
            goto done;
        }
        if (eval_prompt_tokens(ctx, &prompt_tokens, &logits, err, sizeof(err)) != 0 ||
            eval_decode_tokens(ctx, model, n_predict, fixed_token,
                               &logits, NULL, err, sizeof(err)) != 0) {
            fprintf(stderr, "peregrine: %s\n", err);
            goto done;
        }
        pg_llama_context_free(ctx);
        ctx = NULL;
    }

    for (i = 0; i < repeat; i++) {
        double prompt_sec;
        double decode_sec;

        ctx = pg_llama_context_new(model, 0, err, sizeof(err));
        if (!ctx) {
            fprintf(stderr, "peregrine: %s\n", err);
            goto done;
        }

        t0 = now_sec();
        if (eval_prompt_tokens(ctx, &prompt_tokens, &logits, err, sizeof(err)) != 0) {
            fprintf(stderr, "peregrine: %s\n", err);
            goto done;
        }
        prompt_sec = now_sec() - t0;

        if (profile) {
            pg_llama_context_profile_enable(ctx, 1);
            pg_llama_context_profile_reset(ctx);
        }
        t0 = now_sec();
        if (eval_decode_tokens(ctx, model, n_predict, fixed_token,
                               &logits, profile ? &sample_total : NULL,
                               err, sizeof(err)) != 0) {
            fprintf(stderr, "peregrine: %s\n", err);
            goto done;
        }
        decode_sec = now_sec() - t0;
        if (profile) {
            const PgLlamaProfile *p = pg_llama_context_profile(ctx);

            if (p)
                profile_add(&profile_total, p);
        }

        prompt_total += prompt_sec;
        decode_total += decode_sec;
        pg_llama_context_free(ctx);
        ctx = NULL;
    }

    {
        PgThreadPool *pool = pg_global_threadpool();
        double prompt_avg = prompt_total / (double)repeat;
        double decode_avg = decode_total / (double)repeat;
        double prompt_tps = prompt_avg > 0.0 ?
                            (double)prompt_tokens.count / prompt_avg : 0.0;
        double decode_tps = decode_avg > 0.0 ?
                            (double)n_predict / decode_avg : 0.0;

        printf("model: %s\n", path);
        printf("threads: %d\n", pg_threadpool_size(pool));
        printf("prompt_tokens: %zu\n", prompt_tokens.count);
        printf("decode_tokens: %zu\n", n_predict);
        printf("decode_mode: %s\n", fixed_token >= 0 ? "fixed" : "greedy");
        printf("repeat: %zu\n", repeat);
        printf("load_sec: %.6f\n", load_sec);
        printf("tokenize_sec: %.6f\n", tokenize_sec);
        printf("prompt_eval_sec_avg: %.6f\n", prompt_avg);
        printf("prompt_eval_tok_s: %.2f\n", prompt_tps);
        printf("decode_sec_avg: %.6f\n", decode_avg);
        printf("decode_tok_s: %.2f\n", decode_tps);
        if (profile)
            print_profile(&profile_total, sample_total);
    }
    rc = 0;

done:
    pg_llama_context_free(ctx);
    pg_token_buffer_free(&prompt_tokens);
    pg_llama_model_free(model);
    return rc;
}

static int cmd_run(int argc, char **argv)
{
    const char *path = NULL;
    const char *prompt = NULL;
    size_t n_predict = 32;
    PgSamplerParams sampler_params = { 0.0f, 0, 1.0f, 0 };
    PgSampler *sampler = NULL;
    PgLlamaModel *model = NULL;
    PgLlamaContext *ctx = NULL;
    const PgTokenizer *tok;
    PgTokenBuffer prompt_tokens = { 0 };
    const float *logits = NULL;
    char err[256];
    size_t i;
    int rc = 1;

    for (i = 2; i < (size_t)argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < (size_t)argc) {
            path = argv[++i];
        } else if (!strcmp(argv[i], "-p") && i + 1 < (size_t)argc) {
            prompt = argv[++i];
        } else if (!strcmp(argv[i], "-n") && i + 1 < (size_t)argc) {
            if (parse_size_arg(argv[++i], &n_predict) != 0) {
                fprintf(stderr, "peregrine: invalid -n value\n");
                return 2;
            }
        } else if ((!strcmp(argv[i], "--temp") || !strcmp(argv[i], "--temperature")) &&
                   i + 1 < (size_t)argc) {
            if (parse_float_arg(argv[++i], &sampler_params.temperature) != 0) {
                fprintf(stderr, "peregrine: invalid temperature value\n");
                return 2;
            }
        } else if (!strcmp(argv[i], "--top-k") && i + 1 < (size_t)argc) {
            if (parse_size_arg(argv[++i], &sampler_params.top_k) != 0) {
                fprintf(stderr, "peregrine: invalid top-k value\n");
                return 2;
            }
        } else if (!strcmp(argv[i], "--top-p") && i + 1 < (size_t)argc) {
            if (parse_float_arg(argv[++i], &sampler_params.top_p) != 0) {
                fprintf(stderr, "peregrine: invalid top-p value\n");
                return 2;
            }
        } else if (!strcmp(argv[i], "--seed") && i + 1 < (size_t)argc) {
            if (parse_u64_arg(argv[++i], &sampler_params.seed) != 0) {
                fprintf(stderr, "peregrine: invalid seed value\n");
                return 2;
            }
        } else {
            fprintf(stderr,
                    "usage: %s run -m <model.gguf> -p <prompt> [-n tokens] [--temp t] [--top-k k] [--top-p p] [--seed s]\n",
                    argv[0]);
            return 2;
        }
    }
    if (!path || !prompt) {
        fprintf(stderr,
                "usage: %s run -m <model.gguf> -p <prompt> [-n tokens] [--temp t] [--top-k k] [--top-p p] [--seed s]\n",
                argv[0]);
        return 2;
    }

    sampler = pg_sampler_new(&sampler_params, err, sizeof(err));
    if (!sampler) {
        fprintf(stderr, "peregrine: %s\n", err);
        goto done;
    }
    model = pg_llama_model_load(path, err, sizeof(err));
    if (!model) {
        fprintf(stderr, "peregrine: %s\n", err);
        goto done;
    }
    ctx = pg_llama_context_new(model, 0, err, sizeof(err));
    if (!ctx) {
        fprintf(stderr, "peregrine: %s\n", err);
        goto done;
    }
    tok = pg_llama_model_tokenizer(model);
    if (pg_tokenizer_encode(tok, prompt, 1, 0, &prompt_tokens) != 0 ||
        prompt_tokens.count == 0) {
        fprintf(stderr, "peregrine: failed to tokenize prompt\n");
        goto done;
    }

    for (i = 0; i < prompt_tokens.count; i++) {
        if (pg_llama_eval_token(ctx, prompt_tokens.data[i],
                                &logits, err, sizeof(err)) != 0) {
            fprintf(stderr, "peregrine: %s\n", err);
            goto done;
        }
    }

    fputs(prompt, stdout);
    fflush(stdout);
    for (i = 0; i < n_predict; i++) {
        char text[4096];
        size_t written = 0;
        int32_t next = pg_llama_sample(logits, pg_llama_model_vocab_size(model),
                                       sampler, err, sizeof(err));

        if (next < 0) {
            fprintf(stderr, "\nperegrine: %s\n", err[0] ? err : "failed to sample next token");
            goto done;
        }
        if (next == pg_tokenizer_eos_id(tok))
            break;
        if (pg_tokenizer_decode_token(tok, next, text, sizeof(text), &written) != 0) {
            fprintf(stderr, "\nperegrine: failed to decode token %d\n", next);
            goto done;
        }
        fwrite(text, 1, written, stdout);
        fflush(stdout);
        if (pg_llama_eval_token(ctx, next, &logits, err, sizeof(err)) != 0) {
            fprintf(stderr, "\nperegrine: %s\n", err);
            goto done;
        }
    }
    fputc('\n', stdout);
    rc = 0;

done:
    pg_sampler_free(sampler);
    pg_token_buffer_free(&prompt_tokens);
    pg_llama_context_free(ctx);
    pg_llama_model_free(model);
    return rc;
}

static int usage(const char *argv0)
{
    fprintf(stderr,
        "peregrine %s - FFmpeg for AI inference (pre-alpha)\n\n"
        "usage:\n"
        "  %s info                         show build + detected CPU features\n"
        "  %s inspect -m <model>            show model tensor directory\n"
        "  %s tokenize -m <model> -p <txt>  print tokenizer token IDs\n"
        "  %s logits -m <model> -p <txt>    print top logits after prompt\n"
        "  %s bench -m <model> -p <txt>     benchmark prompt/decode eval\n"
        "  %s run -m <model.gguf> -p <txt>  run f32 GGUF inference\n",
        PEREGRINE_VERSION_STRING, argv0, argv0, argv0, argv0, argv0, argv0);
    return 2;
}

int main(int argc, char **argv)
{
    if (argc < 2)
        return cmd_info();
    if (!strcmp(argv[1], "info") || !strcmp(argv[1], "--version"))
        return cmd_info();
    if (!strcmp(argv[1], "inspect"))
        return cmd_inspect(argc, argv);
    if (!strcmp(argv[1], "tokenize"))
        return cmd_tokenize(argc, argv);
    if (!strcmp(argv[1], "logits"))
        return cmd_logits(argc, argv);
    if (!strcmp(argv[1], "bench"))
        return cmd_bench(argc, argv);
    if (!strcmp(argv[1], "run"))
        return cmd_run(argc, argv);
    return usage(argv[0]);
}
