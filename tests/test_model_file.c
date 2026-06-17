/*
 * peregrine - model file loader tests
 */
#include "peregrine/model.h"

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

static void write_str(FILE *f, const char *s)
{
    write_u64(f, strlen(s));
    write_all(f, s, strlen(s));
}

static void write_f32_bits(FILE *f, uint32_t bits)
{
    write_u32(f, bits);
}

static char *make_temp_path(const char *suffix)
{
    char tmpl[256];
    int fd;
    char *path;

    snprintf(tmpl, sizeof(tmpl), "/tmp/peregrine-%s-XXXXXX", suffix);
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

static void write_minimal_gguf(const char *path)
{
    FILE *f = fopen(path, "wb");
    long pos;
    unsigned pad;

    if (!f) {
        fprintf(stderr, "%s: %s\n", path, strerror(errno));
        exit(2);
    }

    write_all(f, "GGUF", 4);
    write_u32(f, 3);
    write_u64(f, 1); /* tensors */
    write_u64(f, 2); /* metadata */

    write_str(f, "general.alignment");
    write_u32(f, 4); /* uint32 */
    write_u32(f, 32);

    write_str(f, "general.architecture");
    write_u32(f, 8); /* string */
    write_str(f, "llama");

    write_str(f, "tok_embeddings.weight");
    write_u32(f, 2); /* dims */
    write_u64(f, 2);
    write_u64(f, 2);
    write_u32(f, 0); /* f32 */
    write_u64(f, 0); /* offset from tensor data base */

    pos = ftell(f);
    if (pos < 0)
        exit(2);
    pad = (unsigned)((32 - ((unsigned)pos & 31u)) & 31u);
    while (pad--)
        fputc(0, f);

    write_f32_bits(f, 0x3f800000u);
    write_f32_bits(f, 0x40000000u);
    write_f32_bits(f, 0x40400000u);
    write_f32_bits(f, 0x40800000u);

    fclose(f);
}

static void write_minimal_safetensors(const char *path)
{
    static const char header[] =
        "{\"w\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]},"
        "\"__metadata__\":{\"format\":\"pt\"}}";
    FILE *f = fopen(path, "wb");

    if (!f) {
        fprintf(stderr, "%s: %s\n", path, strerror(errno));
        exit(2);
    }
    write_u64(f, sizeof(header) - 1);
    write_all(f, header, sizeof(header) - 1);
    write_f32_bits(f, 0x3f800000u);
    write_f32_bits(f, 0x40000000u);
    write_f32_bits(f, 0x40400000u);
    write_f32_bits(f, 0x40800000u);
    fclose(f);
}

static float load_f32(const void *ptr)
{
    float f;
    memcpy(&f, ptr, sizeof(f));
    return f;
}

static void test_gguf(void)
{
    char err[256];
    char *path = make_temp_path("gguf");
    PgModelFile *m;
    const PgTensorView *t;

    write_minimal_gguf(path);
    m = pg_model_file_open(path, err, sizeof(err));
    CHECK(m != NULL);
    if (!m) {
        fprintf(stderr, "gguf load failed: %s\n", err);
        goto done;
    }

    CHECK(pg_model_file_format(m) == PG_MODEL_FORMAT_GGUF);
    CHECK(pg_model_file_tensor_count(m) == 1);
    t = pg_model_file_find_tensor(m, "tok_embeddings.weight", 21);
    CHECK(t != NULL);
    if (t) {
        CHECK(t->type == PG_TENSOR_TYPE_F32);
        CHECK(t->n_dims == 2);
        CHECK(t->dims[0] == 2 && t->dims[1] == 2);
        CHECK(t->nbytes == 16);
        CHECK(load_f32(t->data) == 1.0f);
        CHECK(load_f32((const unsigned char *)t->data + 12) == 4.0f);
    }
    pg_model_file_free(m);

done:
    unlink(path);
    free(path);
}

static void test_safetensors(void)
{
    char err[256];
    char *path = make_temp_path("safetensors");
    PgModelFile *m;
    const PgTensorView *t;

    write_minimal_safetensors(path);
    m = pg_model_file_open(path, err, sizeof(err));
    CHECK(m != NULL);
    if (!m) {
        fprintf(stderr, "safetensors load failed: %s\n", err);
        goto done;
    }

    CHECK(pg_model_file_format(m) == PG_MODEL_FORMAT_SAFETENSORS);
    CHECK(pg_model_file_tensor_count(m) == 1);
    t = pg_model_file_tensor(m, 0);
    CHECK(t != NULL);
    CHECK(pg_model_file_find_tensor(m, "w", 1) == t);
    if (t) {
        CHECK(t->type == PG_TENSOR_TYPE_F32);
        CHECK(t->n_dims == 2);
        CHECK(t->dims[0] == 2 && t->dims[1] == 2);
        CHECK(t->nbytes == 16);
        CHECK(load_f32(t->data) == 1.0f);
        CHECK(load_f32((const unsigned char *)t->data + 12) == 4.0f);
    }
    pg_model_file_free(m);

done:
    unlink(path);
    free(path);
}

static void test_bad_safetensors_bounds(void)
{
    static const char header[] =
        "{\"w\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,20]}}";
    char err[256];
    char *path = make_temp_path("badst");
    PgModelFile *m;
    FILE *f = fopen(path, "wb");

    if (!f)
        exit(2);
    write_u64(f, sizeof(header) - 1);
    write_all(f, header, sizeof(header) - 1);
    write_f32_bits(f, 0x3f800000u);
    fclose(f);

    m = pg_model_file_open(path, err, sizeof(err));
    CHECK(m == NULL);

    unlink(path);
    free(path);
}

static void test_bad_safetensors_hole(void)
{
    static const char header[] =
        "{\"w\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[4,20]}}";
    char err[256];
    char *path = make_temp_path("holest");
    PgModelFile *m;
    FILE *f = fopen(path, "wb");
    int i;

    if (!f)
        exit(2);
    write_u64(f, sizeof(header) - 1);
    write_all(f, header, sizeof(header) - 1);
    for (i = 0; i < 5; i++)
        write_f32_bits(f, 0x3f800000u);
    fclose(f);

    m = pg_model_file_open(path, err, sizeof(err));
    CHECK(m == NULL);

    unlink(path);
    free(path);
}

int main(void)
{
    test_gguf();
    test_safetensors();
    test_bad_safetensors_bounds();
    test_bad_safetensors_hole();
    return failures ? 1 : 0;
}
