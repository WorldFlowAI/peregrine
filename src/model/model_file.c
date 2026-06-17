/*
 * peregrine - zero-copy GGUF / safetensors tensor directory loaders
 */
#include "peregrine/model.h"

#include "util/filemap.h"

#include <ctype.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GGUF_DEFAULT_ALIGNMENT 32u

struct PgModelFile {
    PgModelFormat format;
    PgFileMap map;
    PgMetadataEntry *metadata;
    size_t metadata_count;
    PgTensorView *tensors;
    size_t tensor_count;
};

typedef struct Reader {
    const unsigned char *base;
    const unsigned char *p;
    const unsigned char *end;
    char *err;
    size_t err_len;
} Reader;

typedef struct Json {
    const char *p;
    const char *end;
    char *err;
    size_t err_len;
} Json;

typedef struct TensorInterval {
    size_t begin;
    size_t end;
} TensorInterval;

static void set_err(char *err, size_t err_len, const char *fmt, ...)
{
    va_list ap;

    if (!err || err_len == 0)
        return;
    va_start(ap, fmt);
    vsnprintf(err, err_len, fmt, ap);
    va_end(ap);
}

static void reader_err(Reader *r, const char *fmt, ...)
{
    va_list ap;

    if (!r->err || r->err_len == 0 || r->err[0])
        return;
    va_start(ap, fmt);
    vsnprintf(r->err, r->err_len, fmt, ap);
    va_end(ap);
}

static void json_err(Json *j, const char *fmt, ...)
{
    va_list ap;

    if (!j->err || j->err_len == 0 || j->err[0])
        return;
    va_start(ap, fmt);
    vsnprintf(j->err, j->err_len, fmt, ap);
    va_end(ap);
}

static bool checked_mul_size(size_t a, size_t b, size_t *out)
{
    if (a != 0 && b > SIZE_MAX / a)
        return false;
    *out = a * b;
    return true;
}

static bool checked_add_size(size_t a, size_t b, size_t *out)
{
    if (b > SIZE_MAX - a)
        return false;
    *out = a + b;
    return true;
}

static bool tensor_nbytes(PgTensorType type, unsigned n_dims,
                          const uint64_t dims[PG_MAX_TENSOR_DIMS],
                          size_t *nbytes)
{
    size_t elem_size = pg_tensor_type_size(type);
    size_t count = 1;
    unsigned i;

    if (elem_size == 0)
        return false;
    for (i = 0; i < n_dims; i++) {
        if (dims[i] > SIZE_MAX)
            return false;
        if (!checked_mul_size(count, (size_t)dims[i], &count))
            return false;
    }
    return checked_mul_size(count, elem_size, nbytes);
}

static bool rd_bytes(Reader *r, size_t n, const unsigned char **out)
{
    if ((size_t)(r->end - r->p) < n) {
        reader_err(r, "unexpected end of file");
        return false;
    }
    *out = r->p;
    r->p += n;
    return true;
}

static bool rd_u32(Reader *r, uint32_t *out)
{
    const unsigned char *p;

    if (!rd_bytes(r, 4, &p))
        return false;
    *out = ((uint32_t)p[0]) | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    return true;
}

static bool rd_u8(Reader *r, uint8_t *out)
{
    const unsigned char *p;

    if (!rd_bytes(r, 1, &p))
        return false;
    *out = p[0];
    return true;
}

static bool rd_u16(Reader *r, uint16_t *out)
{
    const unsigned char *p;

    if (!rd_bytes(r, 2, &p))
        return false;
    *out = ((uint16_t)p[0]) | ((uint16_t)p[1] << 8);
    return true;
}

static bool rd_u64(Reader *r, uint64_t *out)
{
    const unsigned char *p;

    if (!rd_bytes(r, 8, &p))
        return false;
    *out = ((uint64_t)p[0]) | ((uint64_t)p[1] << 8) |
           ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24) |
           ((uint64_t)p[4] << 32) | ((uint64_t)p[5] << 40) |
           ((uint64_t)p[6] << 48) | ((uint64_t)p[7] << 56);
    return true;
}

static bool rd_f32(Reader *r, float *out)
{
    uint32_t bits;

    if (!rd_u32(r, &bits))
        return false;
    memcpy(out, &bits, sizeof(bits));
    return true;
}

static bool rd_f64(Reader *r, double *out)
{
    uint64_t bits;

    if (!rd_u64(r, &bits))
        return false;
    memcpy(out, &bits, sizeof(bits));
    return true;
}

static bool rd_string(Reader *r, const char **str, size_t *len)
{
    uint64_t n;
    const unsigned char *p;

    if (!rd_u64(r, &n))
        return false;
    if (n > SIZE_MAX) {
        reader_err(r, "string too large");
        return false;
    }
    if (!rd_bytes(r, (size_t)n, &p))
        return false;
    *str = (const char *)p;
    *len = (size_t)n;
    return true;
}

static PgMetadataType gguf_metadata_type(uint32_t raw_type)
{
    switch (raw_type) {
    case 0:
    case 2:
    case 4:
    case 10:
        return PG_METADATA_TYPE_U64;
    case 1:
    case 3:
    case 5:
    case 11:
        return PG_METADATA_TYPE_I64;
    case 6:
    case 12:
        return PG_METADATA_TYPE_F64;
    case 7:
        return PG_METADATA_TYPE_BOOL;
    case 8:
        return PG_METADATA_TYPE_STRING;
    case 9:
        return PG_METADATA_TYPE_ARRAY;
    default:
        return PG_METADATA_TYPE_UNKNOWN;
    }
}

static void metadata_value_free(PgMetadataValue *value)
{
    if (!value)
        return;
    if (value->type == PG_METADATA_TYPE_ARRAY) {
        switch (value->elem_type) {
        case PG_METADATA_TYPE_U64:
            free((void *)value->as.u64_array);
            break;
        case PG_METADATA_TYPE_I64:
            free((void *)value->as.i64_array);
            break;
        case PG_METADATA_TYPE_F64:
            free((void *)value->as.f64_array);
            break;
        case PG_METADATA_TYPE_BOOL:
            free((void *)value->as.bool_array);
            break;
        case PG_METADATA_TYPE_STRING:
            free((void *)value->as.string_array);
            break;
        default:
            break;
        }
    }
    memset(value, 0, sizeof(*value));
}

static void metadata_free(PgModelFile *file)
{
    size_t i;

    if (!file || !file->metadata)
        return;
    for (i = 0; i < file->metadata_count; i++)
        metadata_value_free(&file->metadata[i].value);
    free(file->metadata);
    file->metadata = NULL;
    file->metadata_count = 0;
}

static bool gguf_read_scalar(Reader *r, uint32_t raw_type, PgMetadataValue *value)
{
    uint8_t u8;
    uint16_t u16;
    uint32_t u32;
    uint64_t u64;
    float f32;
    double f64;
    const char *str;
    size_t len;

    memset(value, 0, sizeof(*value));
    value->type = gguf_metadata_type(raw_type);
    switch (raw_type) {
    case 0:
        if (!rd_u8(r, &u8)) return false;
        value->as.u64 = u8;
        return true;
    case 1:
        if (!rd_u8(r, &u8)) return false;
        value->as.i64 = (int8_t)u8;
        return true;
    case 2:
        if (!rd_u16(r, &u16)) return false;
        value->as.u64 = u16;
        return true;
    case 3:
        if (!rd_u16(r, &u16)) return false;
        value->as.i64 = (int16_t)u16;
        return true;
    case 4:
        if (!rd_u32(r, &u32)) return false;
        value->as.u64 = u32;
        return true;
    case 5:
        if (!rd_u32(r, &u32)) return false;
        value->as.i64 = (int32_t)u32;
        return true;
    case 6:
        if (!rd_f32(r, &f32)) return false;
        value->as.f64 = f32;
        return true;
    case 7:
        if (!rd_u8(r, &u8)) return false;
        value->as.boolean = !!u8;
        return true;
    case 8:
        if (!rd_string(r, &str, &len)) return false;
        value->as.string.data = str;
        value->as.string.len = len;
        return true;
    case 10:
        if (!rd_u64(r, &u64)) return false;
        value->as.u64 = u64;
        return true;
    case 11:
        if (!rd_u64(r, &u64)) return false;
        value->as.i64 = (int64_t)u64;
        return true;
    case 12:
        if (!rd_f64(r, &f64)) return false;
        value->as.f64 = f64;
        return true;
    default:
        reader_err(r, "unsupported GGUF metadata value type %u", raw_type);
        return false;
    }
}

static bool gguf_read_value(Reader *r, uint32_t raw_type, PgMetadataValue *value)
{
    uint32_t elem_raw_type;
    PgMetadataType elem_type;
    uint64_t len;
    size_t count;
    size_t i;

    memset(value, 0, sizeof(*value));
    if (raw_type != 9)
        return gguf_read_scalar(r, raw_type, value);

    if (!rd_u32(r, &elem_raw_type) || !rd_u64(r, &len))
        return false;
    elem_type = gguf_metadata_type(elem_raw_type);
    if (elem_type == PG_METADATA_TYPE_UNKNOWN || elem_type == PG_METADATA_TYPE_ARRAY) {
        reader_err(r, "unsupported GGUF metadata array type %u", elem_raw_type);
        return false;
    }
    if (len > SIZE_MAX) {
        reader_err(r, "metadata array too large");
        return false;
    }

    count = (size_t)len;
    value->type = PG_METADATA_TYPE_ARRAY;
    value->elem_type = elem_type;
    value->count = count;

    switch (elem_type) {
    case PG_METADATA_TYPE_U64: {
        uint64_t *arr = calloc(count ? count : 1, sizeof(*arr));
        if (!arr) {
            reader_err(r, "out of memory");
            return false;
        }
        for (i = 0; i < count; i++) {
            PgMetadataValue elem;
            if (!gguf_read_scalar(r, elem_raw_type, &elem)) {
                free(arr);
                return false;
            }
            arr[i] = elem.as.u64;
        }
        value->as.u64_array = arr;
        return true;
    }
    case PG_METADATA_TYPE_I64: {
        int64_t *arr = calloc(count ? count : 1, sizeof(*arr));
        if (!arr) {
            reader_err(r, "out of memory");
            return false;
        }
        for (i = 0; i < count; i++) {
            PgMetadataValue elem;
            if (!gguf_read_scalar(r, elem_raw_type, &elem)) {
                free(arr);
                return false;
            }
            arr[i] = elem.as.i64;
        }
        value->as.i64_array = arr;
        return true;
    }
    case PG_METADATA_TYPE_F64: {
        double *arr = calloc(count ? count : 1, sizeof(*arr));
        if (!arr) {
            reader_err(r, "out of memory");
            return false;
        }
        for (i = 0; i < count; i++) {
            PgMetadataValue elem;
            if (!gguf_read_scalar(r, elem_raw_type, &elem)) {
                free(arr);
                return false;
            }
            arr[i] = elem.as.f64;
        }
        value->as.f64_array = arr;
        return true;
    }
    case PG_METADATA_TYPE_BOOL: {
        int *arr = calloc(count ? count : 1, sizeof(*arr));
        if (!arr) {
            reader_err(r, "out of memory");
            return false;
        }
        for (i = 0; i < count; i++) {
            PgMetadataValue elem;
            if (!gguf_read_scalar(r, elem_raw_type, &elem)) {
                free(arr);
                return false;
            }
            arr[i] = elem.as.boolean;
        }
        value->as.bool_array = arr;
        return true;
    }
    case PG_METADATA_TYPE_STRING: {
        PgStringView *arr = calloc(count ? count : 1, sizeof(*arr));
        if (!arr) {
            reader_err(r, "out of memory");
            return false;
        }
        for (i = 0; i < count; i++) {
            PgMetadataValue elem;
            if (!gguf_read_scalar(r, elem_raw_type, &elem)) {
                free(arr);
                return false;
            }
            arr[i] = elem.as.string;
        }
        value->as.string_array = arr;
        return true;
    }
    default:
        break;
    }
    reader_err(r, "unsupported GGUF metadata array type %u", elem_raw_type);
    return false;
}

static bool metadata_value_as_u64(const PgMetadataValue *value, uint64_t *out)
{
    if (!value || value->type != PG_METADATA_TYPE_U64)
        return false;
    *out = value->as.u64;
    return true;
}

static PgTensorType gguf_tensor_type(uint32_t type)
{
    switch (type) {
    case 0:  return PG_TENSOR_TYPE_F32;
    case 1:  return PG_TENSOR_TYPE_F16;
    case 24: return PG_TENSOR_TYPE_I8;
    case 25: return PG_TENSOR_TYPE_I16;
    case 26: return PG_TENSOR_TYPE_I32;
    case 27: return PG_TENSOR_TYPE_I64;
    case 28: return PG_TENSOR_TYPE_F64;
    case 30: return PG_TENSOR_TYPE_BF16;
    default: return PG_TENSOR_TYPE_UNKNOWN;
    }
}

static bool parse_gguf(PgModelFile *file, char *err, size_t err_len)
{
    Reader r = { file->map.data, file->map.data, file->map.data + file->map.size, err, err_len };
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_count;
    uint64_t i;
    uint32_t alignment = GGUF_DEFAULT_ALIGNMENT;
    PgMetadataEntry *metadata;
    PgTensorView *tensors;
    uint64_t *offsets;
    size_t tensor_data_base;
    size_t align_add;

    if (file->map.size < 24 || memcmp(file->map.data, "GGUF", 4) != 0) {
        set_err(err, err_len, "not a GGUF file");
        return false;
    }
    r.p += 4;
    if (!rd_u32(&r, &version) || !rd_u64(&r, &tensor_count) || !rd_u64(&r, &metadata_count))
        return false;
    if (version < 2 || version > 3) {
        set_err(err, err_len, "unsupported GGUF version %u", version);
        return false;
    }
    if (tensor_count > SIZE_MAX / sizeof(*tensors)) {
        set_err(err, err_len, "too many GGUF tensors");
        return false;
    }
    if (metadata_count > SIZE_MAX / sizeof(*metadata)) {
        set_err(err, err_len, "too many GGUF metadata entries");
        return false;
    }

    metadata = calloc((size_t)metadata_count ? (size_t)metadata_count : 1, sizeof(*metadata));
    if (!metadata) {
        set_err(err, err_len, "out of memory");
        return false;
    }
    file->metadata = metadata;
    file->metadata_count = 0;
    for (i = 0; i < metadata_count; i++) {
        const char *key;
        size_t key_len;
        uint32_t type;
        PgMetadataValue value;

        if (!rd_string(&r, &key, &key_len) || !rd_u32(&r, &type) ||
            !gguf_read_value(&r, type, &value))
            return false;
        metadata[i].key.data = key;
        metadata[i].key.len = key_len;
        metadata[i].value = value;
        file->metadata_count = (size_t)i + 1;

        if (key_len == 17 && memcmp(key, "general.alignment", 17) == 0) {
            uint64_t align64;

            if (!metadata_value_as_u64(&value, &align64) ||
                align64 == 0 || align64 > UINT32_MAX ||
                (align64 & (align64 - 1)) != 0) {
                set_err(err, err_len, "invalid GGUF alignment");
                return false;
            }
            alignment = (uint32_t)align64;
        }
    }

    tensors = calloc((size_t)tensor_count ? (size_t)tensor_count : 1, sizeof(*tensors));
    offsets = calloc((size_t)tensor_count ? (size_t)tensor_count : 1, sizeof(*offsets));
    if (!tensors) {
        set_err(err, err_len, "out of memory");
        return false;
    }
    if (!offsets) {
        set_err(err, err_len, "out of memory");
        free(tensors);
        return false;
    }

    for (i = 0; i < tensor_count; i++) {
        PgTensorView *t = &tensors[i];
        uint32_t n_dims;
        uint32_t raw_type;
        uint64_t offset;
        unsigned d;

        if (!rd_string(&r, &t->name, &t->name_len) || !rd_u32(&r, &n_dims)) {
            free(offsets);
            free(tensors);
            return false;
        }
        if (n_dims > PG_MAX_TENSOR_DIMS) {
            set_err(err, err_len, "GGUF tensor has too many dimensions");
            free(offsets);
            free(tensors);
            return false;
        }
        t->n_dims = n_dims;
        for (d = 0; d < n_dims; d++) {
            if (!rd_u64(&r, &t->dims[d])) {
                free(offsets);
                free(tensors);
                return false;
            }
        }
        if (!rd_u32(&r, &raw_type) || !rd_u64(&r, &offset)) {
            free(offsets);
            free(tensors);
            return false;
        }
        t->type = gguf_tensor_type(raw_type);
        if (t->type == PG_TENSOR_TYPE_UNKNOWN || !tensor_nbytes(t->type, t->n_dims, t->dims, &t->nbytes)) {
            set_err(err, err_len, "unsupported GGUF tensor type %u", raw_type);
            free(offsets);
            free(tensors);
            return false;
        }
        offsets[i] = offset;
    }

    tensor_data_base = (size_t)(r.p - r.base);
    if (!checked_add_size(tensor_data_base, (size_t)alignment - 1, &align_add)) {
        set_err(err, err_len, "GGUF tensor data base overflow");
        free(offsets);
        free(tensors);
        return false;
    }
    tensor_data_base = align_add & ~(size_t)(alignment - 1);
    if (tensor_data_base > file->map.size) {
        set_err(err, err_len, "GGUF tensor data base is past end of file");
        free(offsets);
        free(tensors);
        return false;
    }

    for (i = 0; i < tensor_count; i++) {
        PgTensorView *t = &tensors[i];
        size_t off;
        size_t abs;
        size_t end;

        if (offsets[i] > SIZE_MAX) {
            set_err(err, err_len, "GGUF tensor offset too large");
            free(offsets);
            free(tensors);
            return false;
        }
        off = (size_t)offsets[i];
        if (!checked_add_size(tensor_data_base, off, &abs) ||
            !checked_add_size(abs, t->nbytes, &end) ||
            end > file->map.size) {
            set_err(err, err_len, "GGUF tensor data is out of bounds");
            free(offsets);
            free(tensors);
            return false;
        }
        t->data = file->map.data + abs;
    }

    free(offsets);
    file->format = PG_MODEL_FORMAT_GGUF;
    file->tensors = tensors;
    file->tensor_count = (size_t)tensor_count;
    return true;
}

static uint64_t json_read_u64_le(const unsigned char *p)
{
    return ((uint64_t)p[0]) | ((uint64_t)p[1] << 8) |
           ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24) |
           ((uint64_t)p[4] << 32) | ((uint64_t)p[5] << 40) |
           ((uint64_t)p[6] << 48) | ((uint64_t)p[7] << 56);
}

static void json_ws(Json *j)
{
    while (j->p < j->end && isspace((unsigned char)*j->p))
        j->p++;
}

static bool json_char(Json *j, char c)
{
    json_ws(j);
    if (j->p >= j->end || *j->p != c) {
        json_err(j, "expected '%c' in safetensors header", c);
        return false;
    }
    j->p++;
    return true;
}

static bool json_string(Json *j, const char **str, size_t *len)
{
    const char *start;

    json_ws(j);
    if (j->p >= j->end || *j->p != '"') {
        json_err(j, "expected string in safetensors header");
        return false;
    }
    j->p++;
    start = j->p;
    while (j->p < j->end && *j->p != '"') {
        if ((unsigned char)*j->p < 0x20 || *j->p == '\\') {
            json_err(j, "escaped/control JSON strings are not supported yet");
            return false;
        }
        j->p++;
    }
    if (j->p >= j->end) {
        json_err(j, "unterminated string in safetensors header");
        return false;
    }
    *str = start;
    *len = (size_t)(j->p - start);
    j->p++;
    return true;
}

static bool json_u64(Json *j, uint64_t *out)
{
    uint64_t v = 0;
    bool any = false;

    json_ws(j);
    while (j->p < j->end && *j->p >= '0' && *j->p <= '9') {
        unsigned digit = (unsigned)(*j->p - '0');
        if (v > (UINT64_MAX - digit) / 10) {
            json_err(j, "integer overflow in safetensors header");
            return false;
        }
        v = v * 10 + digit;
        any = true;
        j->p++;
    }
    if (!any) {
        json_err(j, "expected unsigned integer in safetensors header");
        return false;
    }
    *out = v;
    return true;
}

static bool json_skip_value(Json *j, unsigned depth);

static bool json_skip_array(Json *j, unsigned depth)
{
    if (!json_char(j, '['))
        return false;
    json_ws(j);
    if (j->p < j->end && *j->p == ']') {
        j->p++;
        return true;
    }
    for (;;) {
        if (!json_skip_value(j, depth + 1))
            return false;
        json_ws(j);
        if (j->p < j->end && *j->p == ',') {
            j->p++;
            continue;
        }
        return json_char(j, ']');
    }
}

static bool json_skip_object(Json *j, unsigned depth)
{
    if (!json_char(j, '{'))
        return false;
    json_ws(j);
    if (j->p < j->end && *j->p == '}') {
        j->p++;
        return true;
    }
    for (;;) {
        const char *key;
        size_t key_len;

        if (!json_string(j, &key, &key_len) || !json_char(j, ':') ||
            !json_skip_value(j, depth + 1))
            return false;
        (void)key;
        (void)key_len;
        json_ws(j);
        if (j->p < j->end && *j->p == ',') {
            j->p++;
            continue;
        }
        return json_char(j, '}');
    }
}

static bool json_skip_literal(Json *j, const char *lit)
{
    size_t n = strlen(lit);

    json_ws(j);
    if ((size_t)(j->end - j->p) < n || memcmp(j->p, lit, n) != 0) {
        json_err(j, "invalid literal in safetensors header");
        return false;
    }
    j->p += n;
    return true;
}

static bool json_skip_number(Json *j)
{
    json_ws(j);
    if (j->p < j->end && *j->p == '-')
        j->p++;
    if (j->p >= j->end || !isdigit((unsigned char)*j->p)) {
        json_err(j, "invalid number in safetensors header");
        return false;
    }
    while (j->p < j->end && isdigit((unsigned char)*j->p))
        j->p++;
    if (j->p < j->end && *j->p == '.') {
        j->p++;
        while (j->p < j->end && isdigit((unsigned char)*j->p))
            j->p++;
    }
    if (j->p < j->end && (*j->p == 'e' || *j->p == 'E')) {
        j->p++;
        if (j->p < j->end && (*j->p == '+' || *j->p == '-'))
            j->p++;
        while (j->p < j->end && isdigit((unsigned char)*j->p))
            j->p++;
    }
    return true;
}

static bool json_skip_value(Json *j, unsigned depth)
{
    if (depth > 16) {
        json_err(j, "JSON nesting too deep in safetensors header");
        return false;
    }
    json_ws(j);
    if (j->p >= j->end) {
        json_err(j, "unexpected end of safetensors header");
        return false;
    }
    switch (*j->p) {
    case '"': {
        const char *s;
        size_t n;
        return json_string(j, &s, &n);
    }
    case '[':
        return json_skip_array(j, depth);
    case '{':
        return json_skip_object(j, depth);
    case 't':
        return json_skip_literal(j, "true");
    case 'f':
        return json_skip_literal(j, "false");
    case 'n':
        return json_skip_literal(j, "null");
    default:
        return json_skip_number(j);
    }
}

static PgTensorType safetensors_dtype(const char *s, size_t n)
{
    if (n == 3 && memcmp(s, "F64", 3) == 0) return PG_TENSOR_TYPE_F64;
    if (n == 3 && memcmp(s, "F32", 3) == 0) return PG_TENSOR_TYPE_F32;
    if (n == 3 && memcmp(s, "F16", 3) == 0) return PG_TENSOR_TYPE_F16;
    if (n == 4 && memcmp(s, "BF16", 4) == 0) return PG_TENSOR_TYPE_BF16;
    if (n == 3 && memcmp(s, "I64", 3) == 0) return PG_TENSOR_TYPE_I64;
    if (n == 3 && memcmp(s, "I32", 3) == 0) return PG_TENSOR_TYPE_I32;
    if (n == 3 && memcmp(s, "I16", 3) == 0) return PG_TENSOR_TYPE_I16;
    if (n == 2 && memcmp(s, "I8", 2) == 0) return PG_TENSOR_TYPE_I8;
    if (n == 3 && memcmp(s, "U64", 3) == 0) return PG_TENSOR_TYPE_U64;
    if (n == 3 && memcmp(s, "U32", 3) == 0) return PG_TENSOR_TYPE_U32;
    if (n == 3 && memcmp(s, "U16", 3) == 0) return PG_TENSOR_TYPE_U16;
    if (n == 2 && memcmp(s, "U8", 2) == 0) return PG_TENSOR_TYPE_U8;
    if (n == 4 && memcmp(s, "BOOL", 4) == 0) return PG_TENSOR_TYPE_BOOL;
    return PG_TENSOR_TYPE_UNKNOWN;
}

static bool json_shape(Json *j, PgTensorView *t)
{
    json_ws(j);
    if (!json_char(j, '['))
        return false;
    t->n_dims = 0;
    json_ws(j);
    if (j->p < j->end && *j->p == ']') {
        j->p++;
        return true;
    }
    for (;;) {
        uint64_t dim;

        if (t->n_dims >= PG_MAX_TENSOR_DIMS) {
            json_err(j, "safetensors tensor has too many dimensions");
            return false;
        }
        if (!json_u64(j, &dim))
            return false;
        t->dims[t->n_dims++] = dim;
        json_ws(j);
        if (j->p < j->end && *j->p == ',') {
            j->p++;
            continue;
        }
        return json_char(j, ']');
    }
}

static bool json_offsets(Json *j, uint64_t *begin, uint64_t *end)
{
    if (!json_char(j, '[') || !json_u64(j, begin) || !json_char(j, ',') ||
        !json_u64(j, end) || !json_char(j, ']'))
        return false;
    return true;
}

static bool parse_safetensors_tensor(Json *j, PgTensorView *t,
                                     const unsigned char *data_base,
                                     size_t data_size)
{
    bool have_dtype = false;
    bool have_shape = false;
    bool have_offsets = false;
    uint64_t begin = 0;
    uint64_t end = 0;
    size_t expected;

    if (!json_char(j, '{'))
        return false;
    json_ws(j);
    if (j->p < j->end && *j->p == '}') {
        json_err(j, "empty safetensors tensor entry");
        return false;
    }

    for (;;) {
        const char *key;
        size_t key_len;

        if (!json_string(j, &key, &key_len) || !json_char(j, ':'))
            return false;
        if (key_len == 5 && memcmp(key, "dtype", 5) == 0) {
            const char *dtype;
            size_t dtype_len;

            if (!json_string(j, &dtype, &dtype_len))
                return false;
            t->type = safetensors_dtype(dtype, dtype_len);
            if (t->type == PG_TENSOR_TYPE_UNKNOWN) {
                json_err(j, "unsupported safetensors dtype");
                return false;
            }
            have_dtype = true;
        } else if (key_len == 5 && memcmp(key, "shape", 5) == 0) {
            if (!json_shape(j, t))
                return false;
            have_shape = true;
        } else if (key_len == 12 && memcmp(key, "data_offsets", 12) == 0) {
            if (!json_offsets(j, &begin, &end))
                return false;
            have_offsets = true;
        } else {
            if (!json_skip_value(j, 0))
                return false;
        }

        json_ws(j);
        if (j->p < j->end && *j->p == ',') {
            j->p++;
            continue;
        }
        if (!json_char(j, '}'))
            return false;
        break;
    }

    if (!have_dtype || !have_shape || !have_offsets) {
        json_err(j, "incomplete safetensors tensor entry");
        return false;
    }
    if (begin > end || end > data_size) {
        json_err(j, "safetensors tensor data is out of bounds");
        return false;
    }
    t->nbytes = (size_t)(end - begin);
    if (!tensor_nbytes(t->type, t->n_dims, t->dims, &expected))
        return false;
    if (expected != t->nbytes) {
        json_err(j, "safetensors tensor byte size does not match dtype/shape");
        return false;
    }
    t->data = data_base + begin;
    return true;
}

static bool append_tensor(PgModelFile *file, const PgTensorView *tensor,
                          char *err, size_t err_len)
{
    PgTensorView *next;

    if (file->tensor_count == SIZE_MAX / sizeof(*file->tensors)) {
        set_err(err, err_len, "too many tensors");
        return false;
    }
    next = realloc(file->tensors, (file->tensor_count + 1) * sizeof(*file->tensors));
    if (!next) {
        set_err(err, err_len, "out of memory");
        return false;
    }
    file->tensors = next;
    file->tensors[file->tensor_count++] = *tensor;
    return true;
}

static int interval_cmp(const void *a, const void *b)
{
    const TensorInterval *ia = a;
    const TensorInterval *ib = b;

    if (ia->begin < ib->begin) return -1;
    if (ia->begin > ib->begin) return 1;
    if (ia->end < ib->end) return -1;
    if (ia->end > ib->end) return 1;
    return 0;
}

static bool safetensors_validate_coverage(const PgModelFile *file,
                                          const unsigned char *data_base,
                                          size_t data_size,
                                          char *err, size_t err_len)
{
    TensorInterval *intervals;
    size_t cursor = 0;
    size_t i;

    if (file->tensor_count == 0) {
        if (data_size != 0) {
            set_err(err, err_len, "safetensors byte buffer has no tensor index");
            return false;
        }
        return true;
    }

    intervals = malloc(file->tensor_count * sizeof(*intervals));
    if (!intervals) {
        set_err(err, err_len, "out of memory");
        return false;
    }

    for (i = 0; i < file->tensor_count; i++) {
        const PgTensorView *t = &file->tensors[i];
        const unsigned char *ptr = t->data;

        intervals[i].begin = (size_t)(ptr - data_base);
        intervals[i].end = intervals[i].begin + t->nbytes;
    }
    qsort(intervals, file->tensor_count, sizeof(*intervals), interval_cmp);

    for (i = 0; i < file->tensor_count; i++) {
        if (intervals[i].begin != cursor) {
            set_err(err, err_len, "safetensors byte buffer has holes or overlaps");
            free(intervals);
            return false;
        }
        cursor = intervals[i].end;
    }
    free(intervals);

    if (cursor != data_size) {
        set_err(err, err_len, "safetensors byte buffer is not fully indexed");
        return false;
    }
    return true;
}

static bool parse_safetensors(PgModelFile *file, char *err, size_t err_len)
{
    uint64_t header_len64;
    size_t header_len;
    size_t data_off;
    const unsigned char *data_base;
    size_t data_size;
    Json j;

    if (file->map.size < 9) {
        set_err(err, err_len, "not a safetensors file");
        return false;
    }
    header_len64 = json_read_u64_le(file->map.data);
    if (header_len64 > SIZE_MAX || !checked_add_size(8, (size_t)header_len64, &data_off) ||
        data_off > file->map.size) {
        set_err(err, err_len, "invalid safetensors header length");
        return false;
    }
    header_len = (size_t)header_len64;
    if (header_len == 0 || file->map.data[8] != '{') {
        set_err(err, err_len, "not a safetensors file");
        return false;
    }

    data_base = file->map.data + data_off;
    data_size = file->map.size - data_off;
    j.p = (const char *)file->map.data + 8;
    j.end = j.p + header_len;
    j.err = err;
    j.err_len = err_len;

    if (!json_char(&j, '{'))
        return false;
    json_ws(&j);
    if (j.p < j.end && *j.p == '}') {
        j.p++;
    } else {
        for (;;) {
            const char *name;
            size_t name_len;
            PgTensorView t;

            memset(&t, 0, sizeof(t));
            if (!json_string(&j, &name, &name_len) || !json_char(&j, ':'))
                return false;
            if (name_len == 12 && memcmp(name, "__metadata__", 12) == 0) {
                if (!json_skip_value(&j, 0))
                    return false;
            } else {
                t.name = name;
                t.name_len = name_len;
                if (!parse_safetensors_tensor(&j, &t, data_base, data_size) ||
                    !append_tensor(file, &t, err, err_len))
                    return false;
            }
            json_ws(&j);
            if (j.p < j.end && *j.p == ',') {
                j.p++;
                continue;
            }
            if (!json_char(&j, '}'))
                return false;
            break;
        }
    }
    json_ws(&j);
    if (j.p != j.end) {
        json_err(&j, "trailing non-whitespace in safetensors header");
        return false;
    }
    if (!safetensors_validate_coverage(file, data_base, data_size, err, err_len))
        return false;
    file->format = PG_MODEL_FORMAT_SAFETENSORS;
    return true;
}

PgModelFile *pg_model_file_open(const char *path, char *err, size_t err_len)
{
    PgModelFile *file;

    if (err && err_len)
        err[0] = '\0';
    file = calloc(1, sizeof(*file));
    if (!file) {
        set_err(err, err_len, "out of memory");
        return NULL;
    }
    if (pg_file_map_open(&file->map, path, err, err_len) != 0) {
        free(file);
        return NULL;
    }

    if (file->map.size >= 4 && memcmp(file->map.data, "GGUF", 4) == 0) {
        if (!parse_gguf(file, err, err_len)) {
            pg_model_file_free(file);
            return NULL;
        }
        return file;
    }
    if (file->map.size >= 9 && file->map.data[8] == '{') {
        if (!parse_safetensors(file, err, err_len)) {
            pg_model_file_free(file);
            return NULL;
        }
        return file;
    }

    set_err(err, err_len, "unsupported model file format");
    pg_model_file_free(file);
    return NULL;
}

void pg_model_file_free(PgModelFile *file)
{
    if (!file)
        return;
    metadata_free(file);
    free(file->tensors);
    pg_file_map_close(&file->map);
    free(file);
}

PgModelFormat pg_model_file_format(const PgModelFile *file)
{
    return file ? file->format : PG_MODEL_FORMAT_UNKNOWN;
}

size_t pg_model_file_metadata_count(const PgModelFile *file)
{
    return file ? file->metadata_count : 0;
}

const PgMetadataEntry *pg_model_file_metadata(const PgModelFile *file, size_t idx)
{
    if (!file || idx >= file->metadata_count)
        return NULL;
    return &file->metadata[idx];
}

const PgMetadataEntry *pg_model_file_find_metadata(const PgModelFile *file,
                                                   const char *key, size_t key_len)
{
    size_t i;

    if (!file || !key)
        return NULL;
    for (i = 0; i < file->metadata_count; i++) {
        const PgMetadataEntry *entry = &file->metadata[i];
        if (entry->key.len == key_len && memcmp(entry->key.data, key, key_len) == 0)
            return entry;
    }
    return NULL;
}

size_t pg_model_file_tensor_count(const PgModelFile *file)
{
    return file ? file->tensor_count : 0;
}

const PgTensorView *pg_model_file_tensor(const PgModelFile *file, size_t idx)
{
    if (!file || idx >= file->tensor_count)
        return NULL;
    return &file->tensors[idx];
}

const PgTensorView *pg_model_file_find_tensor(const PgModelFile *file,
                                              const char *name, size_t name_len)
{
    size_t i;

    if (!file || !name)
        return NULL;
    for (i = 0; i < file->tensor_count; i++) {
        const PgTensorView *t = &file->tensors[i];
        if (t->name_len == name_len && memcmp(t->name, name, name_len) == 0)
            return t;
    }
    return NULL;
}

const char *pg_model_format_name(PgModelFormat format)
{
    switch (format) {
    case PG_MODEL_FORMAT_GGUF: return "gguf";
    case PG_MODEL_FORMAT_SAFETENSORS: return "safetensors";
    default: return "unknown";
    }
}

const char *pg_metadata_type_name(PgMetadataType type)
{
    switch (type) {
    case PG_METADATA_TYPE_U64: return "u64";
    case PG_METADATA_TYPE_I64: return "i64";
    case PG_METADATA_TYPE_F64: return "f64";
    case PG_METADATA_TYPE_BOOL: return "bool";
    case PG_METADATA_TYPE_STRING: return "string";
    case PG_METADATA_TYPE_ARRAY: return "array";
    default: return "unknown";
    }
}

const char *pg_tensor_type_name(PgTensorType type)
{
    switch (type) {
    case PG_TENSOR_TYPE_BOOL: return "bool";
    case PG_TENSOR_TYPE_U8: return "u8";
    case PG_TENSOR_TYPE_I8: return "i8";
    case PG_TENSOR_TYPE_U16: return "u16";
    case PG_TENSOR_TYPE_I16: return "i16";
    case PG_TENSOR_TYPE_U32: return "u32";
    case PG_TENSOR_TYPE_I32: return "i32";
    case PG_TENSOR_TYPE_U64: return "u64";
    case PG_TENSOR_TYPE_I64: return "i64";
    case PG_TENSOR_TYPE_F16: return "f16";
    case PG_TENSOR_TYPE_BF16: return "bf16";
    case PG_TENSOR_TYPE_F32: return "f32";
    case PG_TENSOR_TYPE_F64: return "f64";
    default: return "unknown";
    }
}

size_t pg_tensor_type_size(PgTensorType type)
{
    switch (type) {
    case PG_TENSOR_TYPE_BOOL:
    case PG_TENSOR_TYPE_U8:
    case PG_TENSOR_TYPE_I8:
        return 1;
    case PG_TENSOR_TYPE_U16:
    case PG_TENSOR_TYPE_I16:
    case PG_TENSOR_TYPE_F16:
    case PG_TENSOR_TYPE_BF16:
        return 2;
    case PG_TENSOR_TYPE_U32:
    case PG_TENSOR_TYPE_I32:
    case PG_TENSOR_TYPE_F32:
        return 4;
    case PG_TENSOR_TYPE_U64:
    case PG_TENSOR_TYPE_I64:
    case PG_TENSOR_TYPE_F64:
        return 8;
    default:
        return 0;
    }
}

int pg_metadata_as_u64(const PgMetadataValue *value, uint64_t *out)
{
    if (!value || !out)
        return -1;
    if (value->type == PG_METADATA_TYPE_U64) {
        *out = value->as.u64;
        return 0;
    }
    if (value->type == PG_METADATA_TYPE_I64 && value->as.i64 >= 0) {
        *out = (uint64_t)value->as.i64;
        return 0;
    }
    return -1;
}

int pg_metadata_as_i64(const PgMetadataValue *value, int64_t *out)
{
    if (!value || !out)
        return -1;
    if (value->type == PG_METADATA_TYPE_I64) {
        *out = value->as.i64;
        return 0;
    }
    if (value->type == PG_METADATA_TYPE_U64 && value->as.u64 <= INT64_MAX) {
        *out = (int64_t)value->as.u64;
        return 0;
    }
    return -1;
}

int pg_metadata_as_f64(const PgMetadataValue *value, double *out)
{
    if (!value || !out)
        return -1;
    if (value->type == PG_METADATA_TYPE_F64) {
        *out = value->as.f64;
        return 0;
    }
    if (value->type == PG_METADATA_TYPE_U64) {
        *out = (double)value->as.u64;
        return 0;
    }
    if (value->type == PG_METADATA_TYPE_I64) {
        *out = (double)value->as.i64;
        return 0;
    }
    return -1;
}

int pg_metadata_as_string(const PgMetadataValue *value, PgStringView *out)
{
    if (!value || !out || value->type != PG_METADATA_TYPE_STRING)
        return -1;
    *out = value->as.string;
    return 0;
}

int pg_metadata_array_string(const PgMetadataValue *value, size_t idx,
                             PgStringView *out)
{
    if (!value || !out || value->type != PG_METADATA_TYPE_ARRAY ||
        value->elem_type != PG_METADATA_TYPE_STRING || idx >= value->count)
        return -1;
    *out = value->as.string_array[idx];
    return 0;
}
