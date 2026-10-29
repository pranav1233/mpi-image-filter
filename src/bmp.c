#include "bmp.h"

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FILE_HEADER_SIZE 14
#define INFO_HEADER_SIZE 40
#define BI_RGB 0

static void set_err(char *err, size_t errlen, const char *fmt, ...)
{
    if (!err || errlen == 0)
        return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, errlen, fmt, ap);
    va_end(ap);
}

/* Headers are parsed byte by byte (little-endian) so struct layout and
 * host endianness never matter. */
static uint16_t get_u16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }

static uint32_t get_u32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static void put_u16(uint8_t *p, uint16_t v)
{
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
}

static void put_u32(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}

size_t bmp_stride(int32_t width)
{
    /* Each row on disk is padded to a 4-byte boundary. */
    return (((size_t)width * 3 + 3) / 4) * 4;
}

static int parse_headers(FILE *f, const char *path, BmpInfo *info, char *err, size_t errlen)
{
    uint8_t h[FILE_HEADER_SIZE + INFO_HEADER_SIZE];
    if (fread(h, 1, sizeof h, f) != sizeof h) {
        set_err(err, errlen, "%s: not a BMP file (too short)", path);
        return -1;
    }
    if (h[0] != 'B' || h[1] != 'M') {
        set_err(err, errlen, "%s: not a BMP file (bad magic)", path);
        return -1;
    }

    const uint8_t *ih = h + FILE_HEADER_SIZE;
    uint32_t data_offset = get_u32(h + 10);
    uint32_t info_size = get_u32(ih + 0);
    int32_t width = (int32_t)get_u32(ih + 4);
    int32_t height = (int32_t)get_u32(ih + 8);
    uint16_t planes = get_u16(ih + 12);
    uint16_t bpp = get_u16(ih + 14);
    uint32_t compression = get_u32(ih + 16);

    /* BITMAPINFOHEADER (40) and its V4/V5 extensions share the first 40 bytes;
     * with BI_RGB the extra fields do not affect pixel layout. */
    if (info_size != 40 && info_size != 108 && info_size != 124) {
        set_err(err, errlen, "%s: unsupported BMP header size %u (need BITMAPINFOHEADER)", path, info_size);
        return -1;
    }
    if (planes != 1) {
        set_err(err, errlen, "%s: invalid BMP (planes = %u)", path, planes);
        return -1;
    }
    if (bpp != 24) {
        set_err(err, errlen, "%s: unsupported bit depth %u (only 24-bit BMP is supported)", path, bpp);
        return -1;
    }
    if (compression != BI_RGB) {
        set_err(err, errlen, "%s: unsupported compression %u (only uncompressed BI_RGB)", path, compression);
        return -1;
    }
    if (width <= 0 || height == 0 || height == INT32_MIN) {
        set_err(err, errlen, "%s: invalid dimensions %d x %d", path, width, height);
        return -1;
    }
    if (data_offset < FILE_HEADER_SIZE + info_size) {
        set_err(err, errlen, "%s: invalid pixel data offset %u", path, data_offset);
        return -1;
    }

    int32_t abs_h = height < 0 ? -height : height;
    if ((size_t)abs_h > SIZE_MAX / 3 / (size_t)width) {
        set_err(err, errlen, "%s: image too large", path);
        return -1;
    }

    info->width = width;
    info->height = abs_h;
    info->top_down = height < 0;
    info->data_offset = data_offset;
    info->stride = bmp_stride(width);
    info->x_ppm = (int32_t)get_u32(ih + 24);
    info->y_ppm = (int32_t)get_u32(ih + 28);
    return 0;
}

int bmp_read_info(const char *path, BmpInfo *info, char *err, size_t errlen)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        set_err(err, errlen, "%s: %s", path, strerror(errno));
        return -1;
    }
    int rc = parse_headers(f, path, info, err, errlen);
    fclose(f);
    return rc;
}

int bmp_read(const char *path, Image *img, char *err, size_t errlen)
{
    memset(img, 0, sizeof *img);
    FILE *f = fopen(path, "rb");
    if (!f) {
        set_err(err, errlen, "%s: %s", path, strerror(errno));
        return -1;
    }

    BmpInfo info;
    uint8_t *row = NULL;
    uint8_t *data = NULL;
    if (parse_headers(f, path, &info, err, errlen) != 0)
        goto fail;
    if (fseek(f, (long)info.data_offset, SEEK_SET) != 0) {
        set_err(err, errlen, "%s: seek to pixel data failed", path);
        goto fail;
    }

    size_t row_bytes = (size_t)info.width * 3;
    row = malloc(info.stride);
    data = malloc(row_bytes * (size_t)info.height);
    if (!row || !data) {
        set_err(err, errlen, "%s: out of memory", path);
        goto fail;
    }

    for (int32_t i = 0; i < info.height; i++) {
        if (fread(row, 1, info.stride, f) != info.stride) {
            set_err(err, errlen, "%s: truncated pixel data (row %d of %d)", path, i, info.height);
            goto fail;
        }
        /* Positive height = bottom-up on disk: file row i is image row h-1-i. */
        int32_t y = info.top_down ? i : info.height - 1 - i;
        memcpy(data + (size_t)y * row_bytes, row, row_bytes);
    }

    free(row);
    fclose(f);
    img->width = info.width;
    img->height = info.height;
    img->x_ppm = info.x_ppm;
    img->y_ppm = info.y_ppm;
    img->data = data;
    return 0;

fail:
    free(row);
    free(data);
    fclose(f);
    return -1;
}

int bmp_write(const char *path, const Image *img, char *err, size_t errlen)
{
    size_t stride = bmp_stride(img->width);
    size_t row_bytes = (size_t)img->width * 3;
    size_t image_size = stride * (size_t)img->height;
    size_t file_size = FILE_HEADER_SIZE + INFO_HEADER_SIZE + image_size;
    if (file_size > UINT32_MAX) {
        set_err(err, errlen, "%s: image too large for BMP (%zu bytes)", path, file_size);
        return -1;
    }

    uint8_t h[FILE_HEADER_SIZE + INFO_HEADER_SIZE] = {0};
    h[0] = 'B';
    h[1] = 'M';
    put_u32(h + 2, (uint32_t)file_size);
    put_u32(h + 10, FILE_HEADER_SIZE + INFO_HEADER_SIZE);
    uint8_t *ih = h + FILE_HEADER_SIZE;
    put_u32(ih + 0, INFO_HEADER_SIZE);
    put_u32(ih + 4, (uint32_t)img->width);
    put_u32(ih + 8, (uint32_t)img->height);   /* positive: bottom-up */
    put_u16(ih + 12, 1);
    put_u16(ih + 14, 24);
    put_u32(ih + 16, BI_RGB);
    put_u32(ih + 20, (uint32_t)image_size);
    put_u32(ih + 24, (uint32_t)img->x_ppm);
    put_u32(ih + 28, (uint32_t)img->y_ppm);

    FILE *f = fopen(path, "wb");
    if (!f) {
        set_err(err, errlen, "%s: %s", path, strerror(errno));
        return -1;
    }
    uint8_t *row = calloc(1, stride);   /* padding bytes stay zero */
    if (!row) {
        set_err(err, errlen, "%s: out of memory", path);
        fclose(f);
        return -1;
    }

    int ok = fwrite(h, 1, sizeof h, f) == sizeof h;
    for (int32_t i = 0; ok && i < img->height; i++) {
        /* Bottom-up: the first row written is the last image row. */
        memcpy(row, img->data + (size_t)(img->height - 1 - i) * row_bytes, row_bytes);
        ok = fwrite(row, 1, stride, f) == stride;
    }
    free(row);
    if (fclose(f) != 0)
        ok = 0;
    if (!ok) {
        set_err(err, errlen, "%s: write failed", path);
        return -1;
    }
    return 0;
}

void image_free(Image *img)
{
    free(img->data);
    img->data = NULL;
}
