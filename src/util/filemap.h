/*
 * peregrine - POSIX read-only file mapping helper
 */
#ifndef PEREGRINE_UTIL_FILEMAP_H
#define PEREGRINE_UTIL_FILEMAP_H

#include <stddef.h>

typedef struct PgFileMap {
    const unsigned char *data;
    size_t size;
    int fd;
} PgFileMap;

int pg_file_map_open(PgFileMap *map, const char *path, char *err, size_t err_len);
void pg_file_map_close(PgFileMap *map);

#endif /* PEREGRINE_UTIL_FILEMAP_H */
