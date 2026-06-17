/*
 * peregrine - tokenizer tests
 */
#include "peregrine/tokenizer.h"

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

static void write_all(FILE *f, const void *ptr, size_t n)
{
    if (fwrite(ptr, 1, n, f) != n) {
        fprintf(stderr, "fwrite failed: %s\n", strerror(errno));
        exit(2);
    }
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

static void write_strn(FILE *f, const void *s, size_t n)
{
    write_u64(f, n);
    write_all(f, s, n);
}

static char *make_temp_path(void)
{
    char tmpl[] = "/tmp/peregrine-tokenizer-XXXXXX";
    char *path;
    int fd = mkstemp(tmpl);

    if (fd < 0)
        exit(2);
    close(fd);
    path = malloc(strlen(tmpl) + 1);
    if (!path)
        exit(2);
    strcpy(path, tmpl);
    return path;
}

static void write_tokenizer_fixture(const char *path)
{
    static const char sp[] = "\xE2\x96\x81";
    FILE *f = fopen(path, "wb");
    long pos;
    unsigned pad;

    if (!f)
        exit(2);

    write_all(f, "GGUF", 4);
    write_u32(f, 3);
    write_u64(f, 1);  /* tensors */
    write_u64(f, 8);  /* metadata */

    write_str(f, "general.alignment");
    write_u32(f, 4);
    write_u32(f, 32);

    write_str(f, "tokenizer.ggml.model");
    write_u32(f, 8);
    write_str(f, "llama");

    write_str(f, "tokenizer.ggml.bos_token_id");
    write_u32(f, 5);
    write_u32(f, 1);

    write_str(f, "tokenizer.ggml.eos_token_id");
    write_u32(f, 5);
    write_u32(f, 2);

    write_str(f, "tokenizer.ggml.unknown_token_id");
    write_u32(f, 5);
    write_u32(f, 0);

    write_str(f, "tokenizer.ggml.tokens");
    write_u32(f, 9);
    write_u32(f, 8);
    write_u64(f, 6);
    write_str(f, "<unk>");
    write_str(f, "<s>");
    write_str(f, "</s>");
    {
        char token[16];

        memcpy(token, sp, 3);
        memcpy(token + 3, "hello", 5);
        write_strn(f, token, 8);
        memcpy(token, sp, 3);
        memcpy(token + 3, "world", 5);
        write_strn(f, token, 8);
    }
    write_str(f, "<0x21>");

    write_str(f, "tokenizer.ggml.scores");
    write_u32(f, 9);
    write_u32(f, 6);
    write_u64(f, 6);
    write_f32(f, -10.0f);
    write_f32(f, 0.0f);
    write_f32(f, 0.0f);
    write_f32(f, 5.0f);
    write_f32(f, 5.0f);
    write_f32(f, 0.0f);

    write_str(f, "llama.context_length");
    write_u32(f, 4);
    write_u32(f, 16);

    write_str(f, "dummy.weight");
    write_u32(f, 1);
    write_u64(f, 1);
    write_u32(f, 0);
    write_u64(f, 0);

    pos = ftell(f);
    if (pos < 0)
        exit(2);
    pad = (unsigned)((32 - ((unsigned)pos & 31u)) & 31u);
    while (pad--)
        fputc(0, f);
    write_u32(f, 0);
    fclose(f);
}

static void test_tokenizer(void)
{
    char *path = make_temp_path();
    char err[256];
    PgModelFile *model;
    PgTokenizer *tok;
    PgTokenBuffer buf = { 0 };
    char out[16];
    size_t written = 0;

    write_tokenizer_fixture(path);
    model = pg_model_file_open(path, err, sizeof(err));
    CHECK(model != NULL);
    tok = pg_tokenizer_from_model_file(model, err, sizeof(err));
    CHECK(tok != NULL);
    if (tok) {
        CHECK(pg_tokenizer_vocab_size(tok) == 6);
        CHECK(pg_tokenizer_bos_id(tok) == 1);
        CHECK(pg_tokenizer_eos_id(tok) == 2);
        CHECK(pg_tokenizer_encode(tok, "hello world!", 1, 1, &buf) == 0);
        CHECK(buf.count == 5);
        CHECK(buf.count >= 5 && buf.data[0] == 1);
        CHECK(buf.count >= 5 && buf.data[1] == 3);
        CHECK(buf.count >= 5 && buf.data[2] == 4);
        CHECK(buf.count >= 5 && buf.data[3] == 5);
        CHECK(buf.count >= 5 && buf.data[4] == 2);
        CHECK(pg_tokenizer_decode_token(tok, 3, out, sizeof(out), &written) == 0);
        CHECK(written == 6);
        CHECK(memcmp(out, " hello", 6) == 0);
        pg_token_buffer_free(&buf);
        pg_tokenizer_free(tok);
    } else {
        fprintf(stderr, "tokenizer load failed: %s\n", err);
    }
    pg_model_file_free(model);
    unlink(path);
    free(path);
}

int main(void)
{
    test_tokenizer();
    return failures ? 1 : 0;
}
