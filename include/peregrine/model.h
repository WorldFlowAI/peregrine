/*
 * peregrine - model file tensor directory API
 *
 * M3 starts with zero-copy model loading: tensors are views into an mmap'd
 * model file, not copied buffers. Names are not NUL-terminated; use name_len.
 */
#ifndef PEREGRINE_MODEL_H
#define PEREGRINE_MODEL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PG_MAX_TENSOR_DIMS 8

typedef struct PgStringView {
    const char *data;
    size_t len;
} PgStringView;

typedef enum PgModelFormat {
    PG_MODEL_FORMAT_UNKNOWN = 0,
    PG_MODEL_FORMAT_GGUF,
    PG_MODEL_FORMAT_SAFETENSORS,
} PgModelFormat;

typedef enum PgTensorType {
    PG_TENSOR_TYPE_UNKNOWN = 0,
    PG_TENSOR_TYPE_BOOL,
    PG_TENSOR_TYPE_U8,
    PG_TENSOR_TYPE_I8,
    PG_TENSOR_TYPE_U16,
    PG_TENSOR_TYPE_I16,
    PG_TENSOR_TYPE_U32,
    PG_TENSOR_TYPE_I32,
    PG_TENSOR_TYPE_U64,
    PG_TENSOR_TYPE_I64,
    PG_TENSOR_TYPE_F16,
    PG_TENSOR_TYPE_BF16,
    PG_TENSOR_TYPE_F32,
    PG_TENSOR_TYPE_F64,
} PgTensorType;

typedef enum PgMetadataType {
    PG_METADATA_TYPE_UNKNOWN = 0,
    PG_METADATA_TYPE_U64,
    PG_METADATA_TYPE_I64,
    PG_METADATA_TYPE_F64,
    PG_METADATA_TYPE_BOOL,
    PG_METADATA_TYPE_STRING,
    PG_METADATA_TYPE_ARRAY,
} PgMetadataType;

typedef struct PgMetadataValue {
    PgMetadataType type;
    PgMetadataType elem_type;
    size_t count;
    union {
        uint64_t u64;
        int64_t i64;
        double f64;
        int boolean;
        PgStringView string;
        const uint64_t *u64_array;
        const int64_t *i64_array;
        const double *f64_array;
        const int *bool_array;
        const PgStringView *string_array;
    } as;
} PgMetadataValue;

typedef struct PgMetadataEntry {
    PgStringView key;
    PgMetadataValue value;
} PgMetadataEntry;

typedef struct PgTensorView {
    const char *name;
    size_t name_len;
    PgTensorType type;
    unsigned n_dims;
    uint64_t dims[PG_MAX_TENSOR_DIMS];
    const void *data;
    size_t nbytes;
} PgTensorView;

typedef struct PgModelFile PgModelFile;

PgModelFile *pg_model_file_open(const char *path, char *err, size_t err_len);
void pg_model_file_free(PgModelFile *file);

PgModelFormat pg_model_file_format(const PgModelFile *file);
size_t pg_model_file_metadata_count(const PgModelFile *file);
const PgMetadataEntry *pg_model_file_metadata(const PgModelFile *file, size_t idx);
const PgMetadataEntry *pg_model_file_find_metadata(const PgModelFile *file,
                                                   const char *key, size_t key_len);
size_t pg_model_file_tensor_count(const PgModelFile *file);
const PgTensorView *pg_model_file_tensor(const PgModelFile *file, size_t idx);
const PgTensorView *pg_model_file_find_tensor(const PgModelFile *file,
                                              const char *name, size_t name_len);

const char *pg_model_format_name(PgModelFormat format);
const char *pg_metadata_type_name(PgMetadataType type);
const char *pg_tensor_type_name(PgTensorType type);
size_t pg_tensor_type_size(PgTensorType type);

int pg_metadata_as_u64(const PgMetadataValue *value, uint64_t *out);
int pg_metadata_as_i64(const PgMetadataValue *value, int64_t *out);
int pg_metadata_as_f64(const PgMetadataValue *value, double *out);
int pg_metadata_as_string(const PgMetadataValue *value, PgStringView *out);
int pg_metadata_array_string(const PgMetadataValue *value, size_t idx,
                             PgStringView *out);

#ifdef __cplusplus
}
#endif

#endif /* PEREGRINE_MODEL_H */
