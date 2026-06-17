/*
 * peregrine - command line front-end (the `ffmpeg` analog).
 *
 * v0.1 reports build/capability info and is the home for the inference
 * pipeline (load -> tokenize -> eval -> sample) as it lands. Subcommands are
 * intentionally tiny stubs so the binary always builds and runs.
 */
#include <stdio.h>
#include <string.h>

#include "peregrine/model.h"
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

static int usage(const char *argv0)
{
    fprintf(stderr,
        "peregrine %s - FFmpeg for AI inference (pre-alpha)\n\n"
        "usage:\n"
        "  %s info                         show build + detected CPU features\n"
        "  %s inspect -m <model>            show model tensor directory\n"
        "  %s run -m <model.gguf> -p <txt>  run inference   (not yet implemented)\n",
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
    if (!strcmp(argv[1], "run")) {
        fprintf(stderr, "peregrine: `run` not implemented yet\n");
        return 1;
    }
    return usage(argv[0]);
}
