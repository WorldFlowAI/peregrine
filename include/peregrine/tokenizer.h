/*
 * peregrine - tokenizer API
 */
#ifndef PEREGRINE_TOKENIZER_H
#define PEREGRINE_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

#include "peregrine/model.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PgTokenizer PgTokenizer;

typedef struct PgTokenBuffer {
    int32_t *data;
    size_t count;
    size_t capacity;
} PgTokenBuffer;

PgTokenizer *pg_tokenizer_from_model_file(const PgModelFile *file,
                                          char *err, size_t err_len);
void pg_tokenizer_free(PgTokenizer *tok);

int pg_tokenizer_encode(const PgTokenizer *tok, const char *text,
                        int add_bos, int add_eos, PgTokenBuffer *out);
int pg_tokenizer_decode_token(const PgTokenizer *tok, int32_t token,
                              char *out, size_t out_len, size_t *written);

int32_t pg_tokenizer_bos_id(const PgTokenizer *tok);
int32_t pg_tokenizer_eos_id(const PgTokenizer *tok);
int32_t pg_tokenizer_unk_id(const PgTokenizer *tok);
size_t pg_tokenizer_vocab_size(const PgTokenizer *tok);

void pg_token_buffer_free(PgTokenBuffer *buf);

#ifdef __cplusplus
}
#endif

#endif /* PEREGRINE_TOKENIZER_H */
