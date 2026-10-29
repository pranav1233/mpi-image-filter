/* Serial baseline: same CLI and same kernels as the MPI driver, no MPI.
 * The whole image is treated as a single band whose halos are always the
 * replicated edge rows. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bmp.h"
#include "cli.h"
#include "filters.h"

/* Run all iterations, ping-ponging between two halo-extended buffers.
 * On success img->data holds the result. */
static int run_filter(const Options *opt, Image *img, char *err, size_t errlen)
{
    FilterPlan plan;
    if (filter_plan_init(&plan, opt->filter, opt->radius, opt->sigma, img->width) != 0) {
        snprintf(err, errlen, "out of memory");
        return -1;
    }
    size_t rb = (size_t)img->width * 3;
    size_t band_rows = (size_t)img->height + 2 * (size_t)plan.halo;
    uint8_t *buf[2] = {malloc(band_rows * rb), malloc(band_rows * rb)};
    if (!buf[0] || !buf[1]) {
        snprintf(err, errlen, "out of memory");
        free(buf[0]);
        free(buf[1]);
        filter_plan_free(&plan);
        return -1;
    }

    /* Owned rows start after `halo` halo rows. */
    size_t off = (size_t)plan.halo * rb;
    memcpy(buf[0] + off, img->data, (size_t)img->height * rb);
    int cur = 0;
    for (int it = 0; it < opt->iterations; it++) {
        filter_fill_edge_halo(buf[cur] + off, img->width, img->height, plan.halo, 1, 1);
        filter_apply(&plan, buf[cur] + off, buf[1 - cur] + off, img->height);
        cur = 1 - cur;
    }
    memcpy(img->data, buf[cur] + off, (size_t)img->height * rb);

    free(buf[0]);
    free(buf[1]);
    filter_plan_free(&plan);
    return 0;
}

int main(int argc, char **argv)
{
    char err[512];
    Options opt;
    int rc = cli_parse(argc, argv, &opt, err, sizeof err);
    if (rc != 0) {
        fprintf(rc > 0 ? stdout : stderr, rc > 0 ? "%s" : "error: %s\n", err);
        return rc > 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    Image img;
    if (bmp_read(opt.input, &img, err, sizeof err) != 0) {
        fprintf(stderr, "error: %s\n", err);
        return EXIT_FAILURE;
    }
    if (run_filter(&opt, &img, err, sizeof err) != 0 ||
        bmp_write(opt.output, &img, err, sizeof err) != 0) {
        fprintf(stderr, "error: %s\n", err);
        image_free(&img);
        return EXIT_FAILURE;
    }
    image_free(&img);
    return EXIT_SUCCESS;
}
