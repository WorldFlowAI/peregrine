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
size_t pg_model_file_tensor_count(const PgModelFile *file);
const PgTensorView *pg_model_file_tensor(const PgModelFile *file, size_t idx);
const PgTensorView *pg_model_file_find_tensor(const PgModelFile *file,
                                              const char *name, size_t name_len);

const char *pg_model_format_name(PgModelFormat format);
const char *pg_tensor_type_name(PgTensorType type);
size_t pg_tensor_type_size(PgTensorType type);

#ifdef __cplusplus
}
#endif

#endif /* PEREGRINE_MODEL_H */
