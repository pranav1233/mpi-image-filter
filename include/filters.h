#ifndef FILTERS_H
#define FILTERS_H

#include <stdint.h>

/* Pure pixel kernels: no MPI calls here, so the serial and MPI builds run
 * exactly the same arithmetic and their outputs can be compared byte for byte.
 *
 * Buffers are BGR, top-down, width * 3 bytes per row, no padding.
 *
 * Halo convention: a band of `rows` rows is stored with `halo` extra rows
 * above and below it. Kernels receive `src` pointing at the first *owned* row
 * and may read rows -halo .. rows+halo-1 relative to it. Columns are clamped
 * inside the kernels; rows outside the image must already be filled in the
 * halo (see filter_fill_edge_halo). */

typedef enum {
    FILTER_GRAYSCALE,
    FILTER_BOX,
    FILTER_GAUSSIAN,
    FILTER_SOBEL
} FilterKind;

typedef struct {
    FilterKind kind;
    int width;
    int halo;          /* rows needed above/below each band */
    float *weights;    /* gaussian: 2*halo+1 taps, normalized */
    void *scratch;     /* per-row work buffers, sized for `width` */
} FilterPlan;

/* Precompute weights and allocate scratch for images `width` pixels wide.
 * radius is used by box, sigma by gaussian (halo = ceil(3 * sigma)).
 * Returns 0, or -1 if out of memory. */
int filter_plan_init(FilterPlan *p, FilterKind kind, int radius, double sigma, int width);
void filter_plan_free(FilterPlan *p);

/* Halo rows the filter needs, without allocating anything. */
int filter_halo_rows(FilterKind kind, int radius, double sigma);

/* Apply one pass to `rows` owned rows. src and dst must not overlap. */
void filter_apply(const FilterPlan *p, const uint8_t *src, uint8_t *dst, int rows);

/* Clamp-to-edge boundary: copy the band's first row into the `halo` rows
 * above it (if top) and its last row into the rows below it (if bottom). */
void filter_fill_edge_halo(uint8_t *band, int width, int rows, int halo, int top, int bottom);

/* Individual kernels (filter_apply dispatches to these). */
void filter_grayscale(const uint8_t *src, uint8_t *dst, int width, int rows);
void filter_box(const uint8_t *src, uint8_t *dst, int width, int rows, int radius, uint32_t *colsum);
void filter_gaussian(const uint8_t *src, uint8_t *dst, int width, int rows, int radius,
                     const float *weights, float *ring, float *acc);
void filter_sobel(const uint8_t *src, uint8_t *dst, int width, int rows, uint8_t *gray_ring);

#endif
