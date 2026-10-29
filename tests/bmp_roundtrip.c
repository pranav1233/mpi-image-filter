/* Test helper: read a BMP and write it back unchanged. */
#include <stdio.h>
#include <stdlib.h>

#include "bmp.h"

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s in.bmp out.bmp\n", argv[0]);
        return EXIT_FAILURE;
    }
    char err[512];
    Image img;
    if (bmp_read(argv[1], &img, err, sizeof err) != 0 ||
        bmp_write(argv[2], &img, err, sizeof err) != 0) {
        fprintf(stderr, "error: %s\n", err);
        return EXIT_FAILURE;
    }
    image_free(&img);
    return EXIT_SUCCESS;
}
