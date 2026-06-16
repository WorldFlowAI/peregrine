/*
 * peregrine - command line front-end (the `ffmpeg` analog).
 *
 * v0.1 reports build/capability info and is the home for the inference
 * pipeline (load -> tokenize -> eval -> sample) as it lands. Subcommands are
 * intentionally tiny stubs so the binary always builds and runs.
 */
#include <stdio.h>
#include <string.h>

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

static int usage(const char *argv0)
{
    fprintf(stderr,
        "peregrine %s - FFmpeg for AI inference (pre-alpha)\n\n"
        "usage:\n"
        "  %s info                         show build + detected CPU features\n"
        "  %s run -m <model.gguf> -p <txt>  run inference   (not yet implemented)\n",
        PEREGRINE_VERSION_STRING, argv0, argv0);
    return 2;
}

int main(int argc, char **argv)
{
    if (argc < 2)
        return cmd_info();
    if (!strcmp(argv[1], "info") || !strcmp(argv[1], "--version"))
        return cmd_info();
    if (!strcmp(argv[1], "run")) {
        fprintf(stderr, "peregrine: `run` not implemented yet (see ROADMAP.md)\n");
        return 1;
    }
    return usage(argv[0]);
}
