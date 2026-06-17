/*
 * peregrine - POSIX read-only file mapping helper
 */
#include "util/filemap.h"

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static void set_err(char *err, size_t err_len, const char *fmt, ...)
{
    va_list ap;

    if (!err || err_len == 0)
        return;
    va_start(ap, fmt);
    vsnprintf(err, err_len, fmt, ap);
    va_end(ap);
}

int pg_file_map_open(PgFileMap *map, const char *path, char *err, size_t err_len)
{
    struct stat st;
    int fd;
    void *ptr;

    map->data = NULL;
    map->size = 0;
    map->fd = -1;

    fd = open(path, O_RDONLY);
    if (fd < 0) {
        set_err(err, err_len, "%s: %s", path, strerror(errno));
        return -1;
    }

    if (fstat(fd, &st) != 0) {
        set_err(err, err_len, "%s: fstat failed: %s", path, strerror(errno));
        close(fd);
        return -1;
    }
    if (st.st_size <= 0) {
        set_err(err, err_len, "%s: empty file", path);
        close(fd);
        return -1;
    }
    if ((uintmax_t)st.st_size > (uintmax_t)SIZE_MAX) {
        set_err(err, err_len, "%s: file too large for this platform", path);
        close(fd);
        return -1;
    }

    ptr = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (ptr == MAP_FAILED) {
        set_err(err, err_len, "%s: mmap failed: %s", path, strerror(errno));
        close(fd);
        return -1;
    }

    map->data = ptr;
    map->size = (size_t)st.st_size;
    map->fd = fd;
    return 0;
}

void pg_file_map_close(PgFileMap *map)
{
    if (!map)
        return;
    if (map->data)
        munmap((void *)map->data, map->size);
    if (map->fd >= 0)
        close(map->fd);
    map->data = NULL;
    map->size = 0;
    map->fd = -1;
}
