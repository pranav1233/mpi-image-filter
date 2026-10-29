#include "filters.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

/* Integer form of 0.299R + 0.587G + 0.114B, rounded half up. Exact, so there is
 * no floating-point ordering to worry about between builds. */
static inline uint8_t luma(const uint8_t *bgr)
{
    unsigned y = 114u * bgr[0] + 587u * bgr[1] + 299u * bgr[2];
    return (uint8_t)((y + 500u) / 1000u);
}

/* ------------------------------------------------------------------ plan */

int filter_halo_rows(FilterKind kind, int radius, double sigma)
{
    switch (kind) {
    case FILTER_GRAYSCALE: return 0;
    case FILTER_BOX:       return radius;
    case FILTER_GAUSSIAN:  return (int)ceil(3.0 * sigma);
    case FILTER_SOBEL:     return 1;
    }
    return 0;
}

int filter_plan_init(FilterPlan *p, FilterKind kind, int radius, double sigma, int width)
{
    memset(p, 0, sizeof *p);
    p->kind = kind;
    p->width = width;
    p->halo = filter_halo_rows(kind, radius, sigma);

    size_t row_vals = (size_t)width * 3;
    size_t taps = 2 * (size_t)p->halo + 1;
    size_t scratch_bytes = 0;
    switch (kind) {
    case FILTER_GRAYSCALE: break;
    case FILTER_BOX:       scratch_bytes = row_vals * sizeof(uint32_t); break;
    case FILTER_GAUSSIAN:  scratch_bytes = (taps + 1) * row_vals * sizeof(float); break;
    case FILTER_SOBEL:     scratch_bytes = 3 * (size_t)width; break;
    }
    if (scratch_bytes && !(p->scratch = malloc(scratch_bytes)))
        return -1;

    if (kind == FILTER_GAUSSIAN) {
        /* Weights computed in double, normalized, then stored as float once so
         * every rank and the serial build use the very same taps. */
        p->weights = malloc(taps * sizeof(float));
        double *w = malloc(taps * sizeof(double));
        if (!p->weights || !w) {
            free(w);
            filter_plan_free(p);
            return -1;
        }
        double sum = 0.0;
        for (int k = -p->halo; k <= p->halo; k++) {
            w[k + p->halo] = exp(-(double)(k * k) / (2.0 * sigma * sigma));
            sum += w[k + p->halo];
        }
        for (size_t i = 0; i < taps; i++)
            p->weights[i] = (float)(w[i] / sum);
        free(w);
    }
    return 0;
}

void filter_plan_free(FilterPlan *p)
{
    free(p->weights);
    free(p->scratch);
    p->weights = NULL;
    p->scratch = NULL;
}

void filter_apply(const FilterPlan *p, const uint8_t *src, uint8_t *dst, int rows)
{
    size_t row_vals = (size_t)p->width * 3;
    switch (p->kind) {
    case FILTER_GRAYSCALE:
        filter_grayscale(src, dst, p->width, rows);
        break;
    case FILTER_BOX:
        filter_box(src, dst, p->width, rows, p->halo, p->scratch);
        break;
    case FILTER_GAUSSIAN: {
        float *ring = p->scratch;
        float *acc = ring + (2 * (size_t)p->halo + 1) * row_vals;
        filter_gaussian(src, dst, p->width, rows, p->halo, p->weights, ring, acc);
        break;
    }
    case FILTER_SOBEL:
        filter_sobel(src, dst, p->width, rows, p->scratch);
        break;
    }
}

void filter_fill_edge_halo(uint8_t *band, int width, int rows, int halo, int top, int bottom)
{
    size_t rb = (size_t)width * 3;
    for (int k = 1; k <= halo; k++) {
        if (top)
            memcpy(band - (size_t)k * rb, band, rb);
        if (bottom)
            memcpy(band + (size_t)(rows - 1 + k) * rb, band + (size_t)(rows - 1) * rb, rb);
    }
}

/* ------------------------------------------------------------------ kernels */

void filter_grayscale(const uint8_t *src, uint8_t *dst, int width, int rows)
{
    size_t n = (size_t)width * (size_t)rows;
    for (size_t i = 0; i < n; i++) {
        uint8_t y = luma(src + 3 * i);
        dst[3 * i + 0] = y;
        dst[3 * i + 1] = y;
        dst[3 * i + 2] = y;
    }
}

/* Box mean, computed separably with exact integer sums (so summation order
 * cannot change the result). colsum[i] holds the sum of channel value i over
 * rows y-R..y+R and slides down one row per output row. */
void filter_box(const uint8_t *src, uint8_t *dst, int width, int rows, int radius, uint32_t *colsum)
{
    const size_t rv = (size_t)width * 3;
    const int R = radius;
    const uint32_t n = (uint32_t)(2 * R + 1) * (uint32_t)(2 * R + 1);

    memset(colsum, 0, rv * sizeof *colsum);
    for (int k = -R; k <= R; k++) {
        const uint8_t *r = src + (ptrdiff_t)k * (ptrdiff_t)rv;
        for (size_t i = 0; i < rv; i++)
            colsum[i] += r[i];
    }

    for (int y = 0; y < rows; y++) {
        if (y > 0) {
            /* Slide the vertical window: add row y+R, drop row y-1-R (both
             * lie inside the halo-extended band). */
            const uint8_t *add = src + (ptrdiff_t)(y + R) * (ptrdiff_t)rv;
            const uint8_t *sub = src + (ptrdiff_t)(y - 1 - R) * (ptrdiff_t)rv;
            for (size_t i = 0; i < rv; i++)
                colsum[i] += (uint32_t)add[i] - sub[i];
        }
        uint8_t *out = dst + (size_t)y * rv;
        for (int c = 0; c < 3; c++) {
            uint32_t s = 0;
            for (int k = -R; k <= R; k++)
                s += colsum[3 * clampi(k, 0, width - 1) + c];
            for (int x = 0; x < width; x++) {
                out[3 * x + c] = (uint8_t)((s + n / 2) / n);
                s += colsum[3 * clampi(x + R + 1, 0, width - 1) + c];
                s -= colsum[3 * clampi(x - R, 0, width - 1) + c];
            }
        }
    }
}

/* out[i] = sum_k w[k] * row[clamp(x+k)] for every channel value i, with k
 * always accumulated in ascending order so the result is reproducible. The
 * clamped edge columns and the unclamped interior are separate loops so the
 * interior vectorizes. */
static void gauss_horizontal(const uint8_t *row, float *out, int width, int R, const float *w)
{
    const size_t rv = (size_t)width * 3;
    for (size_t i = 0; i < rv; i++)
        out[i] = 0.0f;
    for (int k = -R; k <= R; k++) {
        const float wk = w[k + R];
        int lo = clampi(-k, 0, width);              /* first x with x+k >= 0 */
        int hi = clampi(width - k, 0, width);       /* first x with x+k > width-1 */
        if (hi < lo)
            hi = lo;
        for (int x = 0; x < lo; x++)
            for (int c = 0; c < 3; c++)
                out[3 * x + c] += wk * row[c];
        for (size_t i = 3 * (size_t)lo; i < 3 * (size_t)hi; i++)
            out[i] += wk * row[(ptrdiff_t)i + 3 * k];
        for (int x = hi; x < width; x++)
            for (int c = 0; c < 3; c++)
                out[3 * x + c] += wk * row[3 * (size_t)(width - 1) + c];
    }
}

/* Separable Gaussian. The horizontal pass of each source row (including halo
 * rows) goes into a ring of 2R+1 float rows; ring slot (j + R) % (2R+1) holds
 * source row j. Each output row is the vertical weighted sum over the ring,
 * rounded once at the end. */
void filter_gaussian(const uint8_t *src, uint8_t *dst, int width, int rows, int radius,
                     const float *weights, float *ring, float *acc)
{
    const size_t rv = (size_t)width * 3;
    const int R = radius;
    const int taps = 2 * R + 1;
#define RING_ROW(j) (ring + (size_t)(((j) + R) % taps) * rv)
#define SRC_ROW(j) (src + (ptrdiff_t)(j) * (ptrdiff_t)rv)

    for (int j = -R; j < R; j++)
        gauss_horizontal(SRC_ROW(j), RING_ROW(j), width, R, weights);

    for (int y = 0; y < rows; y++) {
        /* Row y+R replaces y-R-1, which no output row needs any more. */
        gauss_horizontal(SRC_ROW(y + R), RING_ROW(y + R), width, R, weights);

        for (size_t i = 0; i < rv; i++)
            acc[i] = 0.0f;
        for (int k = -R; k <= R; k++) {
            const float wk = weights[k + R];
            const float *r = RING_ROW(y + k);
            for (size_t i = 0; i < rv; i++)
                acc[i] += wk * r[i];
        }
        uint8_t *out = dst + (size_t)y * rv;
        for (size_t i = 0; i < rv; i++) {
            float v = acc[i] + 0.5f;
            out[i] = v >= 255.0f ? 255 : (uint8_t)v;
        }
    }
#undef RING_ROW
#undef SRC_ROW
}

/* Sobel on luminance. Grayscale rows j-1, j, j+1 live in a 3-row ring
 * (slot (j + 1) % 3); the gradient magnitude is rounded and clamped to 255. */
void filter_sobel(const uint8_t *src, uint8_t *dst, int width, int rows, uint8_t *gray_ring)
{
    const size_t rv = (size_t)width * 3;
#define GRAY_ROW(j) (gray_ring + (size_t)(((j) + 1) % 3) * (size_t)width)

    for (int j = -1; j <= rows; j++) {
        const uint8_t *s = src + (ptrdiff_t)j * (ptrdiff_t)rv;
        uint8_t *g = GRAY_ROW(j);
        for (int x = 0; x < width; x++)
            g[x] = luma(s + 3 * x);
        if (j < 1)
            continue;

        /* Rows j-2, j-1, j are now in the ring: emit output row y = j-1. */
        int y = j - 1;
        const uint8_t *t = GRAY_ROW(y - 1), *m = GRAY_ROW(y), *b = GRAY_ROW(y + 1);
        uint8_t *out = dst + (size_t)y * rv;
        for (int x = 0; x < width; x++) {
            int l = x > 0 ? x - 1 : 0;
            int r = x < width - 1 ? x + 1 : width - 1;
            int gx = (t[r] + 2 * m[r] + b[r]) - (t[l] + 2 * m[l] + b[l]);
            int gy = (b[l] + 2 * b[x] + b[r]) - (t[l] + 2 * t[x] + t[r]);
            double mag = sqrt((double)(gx * gx + gy * gy));
            int v = (int)(mag + 0.5);
            uint8_t px = (uint8_t)(v > 255 ? 255 : v);
            out[3 * x + 0] = px;
            out[3 * x + 1] = px;
            out[3 * x + 2] = px;
        }
    }
#undef GRAY_ROW
}
