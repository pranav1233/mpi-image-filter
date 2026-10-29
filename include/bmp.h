#ifndef BMP_H
#define BMP_H

#include <stddef.h>
#include <stdint.h>

/* In-memory image: contiguous, top-down, no row padding, BGR order,
 * width * height * 3 bytes. */
typedef struct {
    int32_t width;
    int32_t height;     /* always positive in memory */
    int32_t x_ppm;      /* resolution, carried through so round-trips are exact */
    int32_t y_ppm;
    uint8_t *data;
} Image;

/* Header facts needed to locate pixel rows inside a file (also used by MPI-IO). */
typedef struct {
    int32_t width;
    int32_t height;     /* absolute value */
    int top_down;       /* 1 if biHeight was negative */
    uint32_t data_offset;
    size_t stride;      /* bytes per row on disk, including padding */
    int32_t x_ppm;
    int32_t y_ppm;
} BmpInfo;

/* Bytes per row on disk: width * 3 rounded up to a multiple of 4. */
size_t bmp_stride(int32_t width);

/* Parse and validate the headers. Returns 0 on success, -1 with a message in err. */
int bmp_read_info(const char *path, BmpInfo *info, char *err, size_t errlen);

/* Read a 24-bit BI_RGB BMP into img (allocates img->data). Returns 0 or -1. */
int bmp_read(const char *path, Image *img, char *err, size_t errlen);

/* Write img as a bottom-up 24-bit BI_RGB BMP. Returns 0 or -1. */
int bmp_write(const char *path, const Image *img, char *err, size_t errlen);

void image_free(Image *img);

#endif
