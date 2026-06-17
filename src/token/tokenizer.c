/*
 * peregrine - GGUF tokenizer support
 */
#include "peregrine/tokenizer.h"

#include <ctype.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PG_TOKENIZER_HASH_LOAD_NUM 2
#define PG_TOKENIZER_HASH_LOAD_DEN 3

typedef struct TokenEntry {
    PgStringView text;
    double score;
} TokenEntry;

struct PgTokenizer {
    TokenEntry *vocab;
    size_t vocab_size;
    int32_t bos_id;
    int32_t eos_id;
    int32_t unk_id;
    int32_t byte_id[256];
    int32_t *hash_ids;
    size_t hash_size;
    size_t max_token_len;
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

static uint64_t hash_bytes(const char *s, size_t n)
{
    uint64_t h = 1469598103934665603ull;
    size_t i;

    for (i = 0; i < n; i++) {
        h ^= (unsigned char)s[i];
        h *= 1099511628211ull;
    }
    return h;
}

static bool string_eq(PgStringView a, const char *b, size_t n)
{
    return a.len == n && memcmp(a.data, b, n) == 0;
}

static int token_buffer_push(PgTokenBuffer *buf, int32_t token)
{
    int32_t *next;
    size_t cap;

    if (buf->count == buf->capacity) {
        cap = buf->capacity ? buf->capacity * 2 : 16;
        next = realloc(buf->data, cap * sizeof(*next));
        if (!next)
            return -1;
        buf->data = next;
        buf->capacity = cap;
    }
    buf->data[buf->count++] = token;
    return 0;
}

void pg_token_buffer_free(PgTokenBuffer *buf)
{
    if (!buf)
        return;
    free(buf->data);
    buf->data = NULL;
    buf->count = 0;
    buf->capacity = 0;
}

static size_t next_pow2(size_t v)
{
    size_t p = 1;

    while (p < v && p <= SIZE_MAX / 2)
        p <<= 1;
    return p;
}

static int tokenizer_find(const PgTokenizer *tok, const char *s, size_t n)
{
    uint64_t h;
    size_t mask;
    size_t pos;

    if (!tok || tok->hash_size == 0)
        return -1;
    h = hash_bytes(s, n);
    mask = tok->hash_size - 1;
    pos = (size_t)h & mask;
    for (;;) {
        int32_t id = tok->hash_ids[pos];
        if (id < 0)
            return -1;
        if (string_eq(tok->vocab[id].text, s, n))
            return id;
        pos = (pos + 1) & mask;
    }
}

static int tokenizer_build_hash(PgTokenizer *tok)
{
    size_t need = tok->vocab_size * PG_TOKENIZER_HASH_LOAD_DEN /
                  PG_TOKENIZER_HASH_LOAD_NUM + 1;
    size_t i;

    tok->hash_size = next_pow2(need < 16 ? 16 : need);
    tok->hash_ids = malloc(tok->hash_size * sizeof(*tok->hash_ids));
    if (!tok->hash_ids)
        return -1;
    for (i = 0; i < tok->hash_size; i++)
        tok->hash_ids[i] = -1;

    for (i = 0; i < tok->vocab_size; i++) {
        size_t mask = tok->hash_size - 1;
        size_t pos = (size_t)hash_bytes(tok->vocab[i].text.data,
                                        tok->vocab[i].text.len) & mask;
        while (tok->hash_ids[pos] >= 0)
            pos = (pos + 1) & mask;
        tok->hash_ids[pos] = (int32_t)i;
        if (tok->vocab[i].text.len > tok->max_token_len)
            tok->max_token_len = tok->vocab[i].text.len;
    }
    return 0;
}

static int hex_digit(int c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static void tokenizer_build_byte_ids(PgTokenizer *tok)
{
    size_t i;

    for (i = 0; i < 256; i++)
        tok->byte_id[i] = tok->unk_id;
    for (i = 0; i < tok->vocab_size; i++) {
        PgStringView s = tok->vocab[i].text;
        int hi, lo;

        if (s.len == 6 && memcmp(s.data, "<0x", 3) == 0 && s.data[5] == '>') {
            hi = hex_digit((unsigned char)s.data[3]);
            lo = hex_digit((unsigned char)s.data[4]);
            if (hi >= 0 && lo >= 0)
                tok->byte_id[(hi << 4) | lo] = (int32_t)i;
        }
    }
}

static const PgMetadataEntry *need_meta(const PgModelFile *file,
                                        const char *key, char *err, size_t err_len)
{
    const PgMetadataEntry *entry = pg_model_file_find_metadata(file, key, strlen(key));

    if (!entry)
        set_err(err, err_len, "missing tokenizer metadata '%s'", key);
    return entry;
}

static int metadata_i32(const PgModelFile *file, const char *key,
                        int32_t *out, char *err, size_t err_len)
{
    const PgMetadataEntry *entry = need_meta(file, key, err, err_len);
    int64_t v;

    if (!entry)
        return -1;
    if (pg_metadata_as_i64(&entry->value, &v) != 0 || v < INT32_MIN || v > INT32_MAX) {
        set_err(err, err_len, "invalid tokenizer metadata '%s'", key);
        return -1;
    }
    *out = (int32_t)v;
    return 0;
}

PgTokenizer *pg_tokenizer_from_model_file(const PgModelFile *file,
                                          char *err, size_t err_len)
{
    const PgMetadataEntry *model_entry;
    const PgMetadataEntry *tokens_entry;
    const PgMetadataEntry *scores_entry;
    PgStringView model_name;
    PgTokenizer *tok;
    size_t i;

    if (err && err_len)
        err[0] = '\0';
    if (!file || pg_model_file_format(file) != PG_MODEL_FORMAT_GGUF) {
        set_err(err, err_len, "tokenizer requires a GGUF model file");
        return NULL;
    }

    model_entry = need_meta(file, "tokenizer.ggml.model", err, err_len);
    tokens_entry = need_meta(file, "tokenizer.ggml.tokens", err, err_len);
    if (!model_entry || !tokens_entry)
        return NULL;
    if (pg_metadata_as_string(&model_entry->value, &model_name) != 0 ||
        !string_eq(model_name, "llama", 5)) {
        set_err(err, err_len, "only GGUF llama tokenizers are supported");
        return NULL;
    }
    if (tokens_entry->value.type != PG_METADATA_TYPE_ARRAY ||
        tokens_entry->value.elem_type != PG_METADATA_TYPE_STRING) {
        set_err(err, err_len, "invalid tokenizer token array");
        return NULL;
    }

    tok = calloc(1, sizeof(*tok));
    if (!tok) {
        set_err(err, err_len, "out of memory");
        return NULL;
    }
    tok->vocab_size = tokens_entry->value.count;
    tok->bos_id = 1;
    tok->eos_id = 2;
    tok->unk_id = 0;
    (void)metadata_i32(file, "tokenizer.ggml.bos_token_id", &tok->bos_id, NULL, 0);
    (void)metadata_i32(file, "tokenizer.ggml.eos_token_id", &tok->eos_id, NULL, 0);
    (void)metadata_i32(file, "tokenizer.ggml.unknown_token_id", &tok->unk_id, NULL, 0);

    tok->vocab = calloc(tok->vocab_size ? tok->vocab_size : 1, sizeof(*tok->vocab));
    if (!tok->vocab) {
        pg_tokenizer_free(tok);
        set_err(err, err_len, "out of memory");
        return NULL;
    }

    scores_entry = pg_model_file_find_metadata(file, "tokenizer.ggml.scores",
                                               strlen("tokenizer.ggml.scores"));
    if (scores_entry && (scores_entry->value.type != PG_METADATA_TYPE_ARRAY ||
                         scores_entry->value.elem_type != PG_METADATA_TYPE_F64 ||
                         scores_entry->value.count != tok->vocab_size)) {
        pg_tokenizer_free(tok);
        set_err(err, err_len, "invalid tokenizer score array");
        return NULL;
    }

    for (i = 0; i < tok->vocab_size; i++) {
        tok->vocab[i].text = tokens_entry->value.as.string_array[i];
        tok->vocab[i].score = scores_entry ? scores_entry->value.as.f64_array[i] : 0.0;
    }
    if (tokenizer_build_hash(tok) != 0) {
        pg_tokenizer_free(tok);
        set_err(err, err_len, "out of memory");
        return NULL;
    }
    tokenizer_build_byte_ids(tok);
    return tok;
}

void pg_tokenizer_free(PgTokenizer *tok)
{
    if (!tok)
        return;
    free(tok->vocab);
    free(tok->hash_ids);
    free(tok);
}

static int append_normalized(char **buf, size_t *len, size_t *cap,
                             const char *src)
{
    static const char spm_space[] = "\xE2\x96\x81";
    size_t i;

    for (i = 0; src[i]; i++) {
        const char *p = &src[i];
        size_t n = 1;

        if (src[i] == ' ') {
            p = spm_space;
            n = sizeof(spm_space) - 1;
        }
        if (*len + n > *cap) {
            size_t next_cap = *cap ? *cap * 2 : 64;
            char *next;

            while (*len + n > next_cap)
                next_cap *= 2;
            next = realloc(*buf, next_cap);
            if (!next)
                return -1;
            *buf = next;
            *cap = next_cap;
        }
        memcpy(*buf + *len, p, n);
        *len += n;
    }
    return 0;
}

static int normalize_llama_text(const char *text, char **out, size_t *out_len)
{
    char *buf = NULL;
    size_t len = 0;
    size_t cap = 0;

    if (append_normalized(&buf, &len, &cap, " ") != 0 ||
        append_normalized(&buf, &len, &cap, text) != 0) {
        free(buf);
        return -1;
    }
    *out = buf;
    *out_len = len;
    return 0;
}

typedef struct SpmSymbol {
    size_t off;
    size_t len;
    int prev;
    int next;
} SpmSymbol;

typedef struct SpmBigram {
    int left;
    int right;
    double score;
    size_t size;
} SpmBigram;

typedef struct SpmHeap {
    SpmBigram *data;
    size_t count;
    size_t capacity;
} SpmHeap;

static size_t utf8_char_len(const char *s, size_t remaining)
{
    unsigned char c;
    size_t n = 1;

    if (remaining == 0)
        return 0;
    c = (unsigned char)s[0];
    if ((c & 0x80) == 0x00)
        n = 1;
    else if ((c & 0xe0) == 0xc0)
        n = 2;
    else if ((c & 0xf0) == 0xe0)
        n = 3;
    else if ((c & 0xf8) == 0xf0)
        n = 4;
    return n <= remaining ? n : 1;
}

static int spm_bigram_better(const SpmBigram *a, const SpmBigram *b)
{
    if (a->score > b->score)
        return 1;
    if (a->score < b->score)
        return 0;
    return a->left < b->left;
}

static int spm_heap_push(SpmHeap *heap, SpmBigram bigram)
{
    size_t i;

    if (heap->count == heap->capacity) {
        size_t cap = heap->capacity ? heap->capacity * 2 : 32;
        SpmBigram *next;

        if (cap < heap->capacity)
            return -1;
        if (cap > SIZE_MAX / sizeof(*next))
            return -1;
        next = realloc(heap->data, cap * sizeof(*next));
        if (!next)
            return -1;
        heap->data = next;
        heap->capacity = cap;
    }

    i = heap->count++;
    while (i > 0) {
        size_t parent = (i - 1) >> 1;

        if (!spm_bigram_better(&bigram, &heap->data[parent]))
            break;
        heap->data[i] = heap->data[parent];
        i = parent;
    }
    heap->data[i] = bigram;
    return 0;
}

static int spm_heap_pop(SpmHeap *heap, SpmBigram *out)
{
    SpmBigram tail;
    size_t i = 0;

    if (heap->count == 0)
        return 0;
    *out = heap->data[0];
    tail = heap->data[--heap->count];
    while (1) {
        size_t left = i * 2 + 1;
        size_t right = left + 1;
        size_t best = left;

        if (left >= heap->count)
            break;
        if (right < heap->count &&
            spm_bigram_better(&heap->data[right], &heap->data[left]))
            best = right;
        if (!spm_bigram_better(&heap->data[best], &tail))
            break;
        heap->data[i] = heap->data[best];
        i = best;
    }
    if (heap->count > 0)
        heap->data[i] = tail;
    return 1;
}

static int spm_try_add_bigram(const PgTokenizer *tok, const char *norm,
                              const SpmSymbol *symbols, int left, int right,
                              SpmHeap *heap)
{
    SpmBigram bigram;
    int id;

    if (left < 0 || right < 0)
        return 0;
    if (symbols[left].len == 0 || symbols[right].len == 0)
        return 0;

    bigram.left = left;
    bigram.right = right;
    bigram.size = symbols[left].len + symbols[right].len;
    id = tokenizer_find(tok, norm + symbols[left].off, bigram.size);
    if (id < 0)
        return 0;
    bigram.score = tok->vocab[id].score;
    return spm_heap_push(heap, bigram);
}

static int spm_build_symbols(const char *norm, size_t n,
                             SpmSymbol **out_symbols, size_t *out_count)
{
    SpmSymbol *symbols;
    size_t count = 0;
    size_t pos = 0;

    symbols = malloc((n ? n : 1) * sizeof(*symbols));
    if (!symbols)
        return -1;
    while (pos < n) {
        size_t len = utf8_char_len(norm + pos, n - pos);

        symbols[count].off = pos;
        symbols[count].len = len;
        symbols[count].prev = count ? (int)count - 1 : -1;
        symbols[count].next = -1;
        if (count > 0)
            symbols[count - 1].next = (int)count;
        count++;
        pos += len;
    }
    *out_symbols = symbols;
    *out_count = count;
    return 0;
}

static int spm_emit_symbol(const PgTokenizer *tok, const char *norm,
                           const SpmSymbol *symbol, PgTokenBuffer *out)
{
    int id = tokenizer_find(tok, norm + symbol->off, symbol->len);
    size_t i;

    if (id >= 0)
        return token_buffer_push(out, id);
    for (i = 0; i < symbol->len; i++) {
        int32_t byte_id = tok->byte_id[(unsigned char)norm[symbol->off + i]];

        if (token_buffer_push(out, byte_id) != 0)
            return -1;
    }
    return 0;
}

int pg_tokenizer_encode(const PgTokenizer *tok, const char *text,
                        int add_bos, int add_eos, PgTokenBuffer *out)
{
    char *norm = NULL;
    SpmSymbol *symbols = NULL;
    SpmHeap heap = { 0 };
    size_t n;
    size_t n_symbols = 0;
    size_t i;

    if (!tok || !text || !out)
        return -1;
    if (normalize_llama_text(text, &norm, &n) != 0)
        return -1;
    if (spm_build_symbols(norm, n, &symbols, &n_symbols) != 0)
        goto fail;

    for (i = 1; i < n_symbols; i++) {
        if (spm_try_add_bigram(tok, norm, symbols, (int)i - 1, (int)i,
                               &heap) != 0)
            goto fail;
    }
    {
        SpmBigram bigram;

        while (spm_heap_pop(&heap, &bigram)) {
            SpmSymbol *left;
            SpmSymbol *right;
            int next;

            if (symbols[bigram.left].len == 0 ||
                symbols[bigram.right].len == 0 ||
                symbols[bigram.left].next != bigram.right ||
                symbols[bigram.right].prev != bigram.left ||
                symbols[bigram.left].len + symbols[bigram.right].len != bigram.size)
                continue;

            left = &symbols[bigram.left];
            right = &symbols[bigram.right];
            next = right->next;
            left->len += right->len;
            left->next = next;
            right->len = 0;
            if (next >= 0)
                symbols[next].prev = bigram.left;

            if (spm_try_add_bigram(tok, norm, symbols, left->prev,
                                   bigram.left, &heap) != 0 ||
                spm_try_add_bigram(tok, norm, symbols, bigram.left,
                                   left->next, &heap) != 0)
                goto fail;
        }
    }

    if (add_bos && token_buffer_push(out, tok->bos_id) != 0)
        goto fail;
    for (i = 0; i < n_symbols; i++) {
        if (symbols[i].prev < 0 && symbols[i].len > 0) {
            int cursor = (int)i;

            while (cursor >= 0) {
                if (spm_emit_symbol(tok, norm, &symbols[cursor], out) != 0)
                    goto fail;
                cursor = symbols[cursor].next;
            }
            break;
        }
    }
    if (n_symbols == 0 && tok->unk_id >= 0) {
        if (token_buffer_push(out, tok->unk_id) != 0)
            goto fail;
    }
    if (add_eos && token_buffer_push(out, tok->eos_id) != 0)
        goto fail;

    free(heap.data);
    free(symbols);
    free(norm);
    return 0;

fail:
    free(heap.data);
    free(symbols);
    free(norm);
    return -1;
}

int pg_tokenizer_decode_token(const PgTokenizer *tok, int32_t token,
                              char *out, size_t out_len, size_t *written)
{
    static const unsigned char spm_space[] = { 0xE2, 0x96, 0x81 };
    PgStringView s;
    size_t i;
    size_t n = 0;

    if (!tok || token < 0 || (size_t)token >= tok->vocab_size)
        return -1;
    s = tok->vocab[token].text;
    if (s.len == 6 && memcmp(s.data, "<0x", 3) == 0 && s.data[5] == '>') {
        int hi = hex_digit((unsigned char)s.data[3]);
        int lo = hex_digit((unsigned char)s.data[4]);

        if (hi < 0 || lo < 0)
            return -1;
        if (out && out_len > 0)
            out[0] = (char)((hi << 4) | lo);
        if (written)
            *written = 1;
        return out_len >= 1 ? 0 : -1;
    }

    for (i = 0; i < s.len; ) {
        char c;
        size_t copy = 1;

        if (i + 3 <= s.len && memcmp(s.data + i, spm_space, 3) == 0) {
            c = ' ';
            copy = 3;
        } else {
            c = s.data[i];
        }
        if (out && n < out_len)
            out[n] = c;
        n++;
        i += copy;
    }
    if (written)
        *written = n;
    return n <= out_len ? 0 : -1;
}

int32_t pg_tokenizer_bos_id(const PgTokenizer *tok)
{
    return tok ? tok->bos_id : -1;
}

int32_t pg_tokenizer_eos_id(const PgTokenizer *tok)
{
    return tok ? tok->eos_id : -1;
}

int32_t pg_tokenizer_unk_id(const PgTokenizer *tok)
{
    return tok ? tok->unk_id : -1;
}

size_t pg_tokenizer_vocab_size(const PgTokenizer *tok)
{
    return tok ? tok->vocab_size : 0;
}
