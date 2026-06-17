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

#include "peregrine/llama.h"
#include "peregrine/model.h"
#include "peregrine/tokenizer.h"
#include "peregrine/version.h"
#include "util/cpu.h"

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

static int cmd_run(int argc, char **argv)
{
    const char *path = NULL;
    const char *prompt = NULL;
    size_t n_predict = 32;
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
        } else {
            fprintf(stderr, "usage: %s run -m <model.gguf> -p <prompt> [-n tokens]\n", argv[0]);
            return 2;
        }
    }
    if (!path || !prompt) {
        fprintf(stderr, "usage: %s run -m <model.gguf> -p <prompt> [-n tokens]\n", argv[0]);
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
        int32_t next = pg_llama_sample_greedy(logits, pg_llama_model_vocab_size(model));

        if (next < 0) {
            fprintf(stderr, "\nperegrine: failed to sample next token\n");
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
        "  %s run -m <model.gguf> -p <txt>  run greedy f32 GGUF inference\n",
        PEREGRINE_VERSION_STRING, argv0, argv0, argv0);
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
    if (!strcmp(argv[1], "run"))
        return cmd_run(argc, argv);
    return usage(argv[0]);
}
