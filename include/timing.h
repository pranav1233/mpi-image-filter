#ifndef TIMING_H
#define TIMING_H

#include <stddef.h>

/* Per-run phase times in seconds. In the MPI build each field is reduced with
 * MPI_MAX across ranks before it is written, so a phase is as slow as its
 * slowest rank. t_compute and t_halo are sums over all iterations. */
typedef struct {
    double t_read;
    double t_scatter;
    double t_compute;
    double t_halo;
    double t_gather;
    double t_write;
    double t_total;
} Timings;

#define TIMINGS_COUNT 7   /* number of doubles in Timings, for MPI_Reduce */

/* Wall-clock seconds: MPI_Wtime in the MPI build (compiled with -DUSE_MPI),
 * clock_gettime(CLOCK_MONOTONIC) in the serial build. */
double timing_now(void);

/* Append one CSV row, writing the header first if the file is new or empty:
 * procs,width,height,filter,radius,iterations,t_read,...,t_total
 * Returns 0, or -1 if the file cannot be written. */
int timing_append_csv(const char *path, int procs, int width, int height, const char *filter,
                      int radius, int iterations, const Timings *t);

#endif
