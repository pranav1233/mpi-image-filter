/* clock_gettime is POSIX, not C11. */
#define _POSIX_C_SOURCE 199309L

#include "timing.h"

#include <stdio.h>

#ifdef USE_MPI
#include <mpi.h>

double timing_now(void) { return MPI_Wtime(); }
#else
#include <time.h>

double timing_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
#endif

int timing_append_csv(const char *path, int procs, int width, int height, const char *filter,
                      int radius, int iterations, const Timings *t)
{
    FILE *f = fopen(path, "a");
    if (!f)
        return -1;
    /* In append mode the position starts at the end, so 0 means empty. */
    fseek(f, 0, SEEK_END);
    if (ftell(f) == 0)
        fprintf(f, "procs,width,height,filter,radius,iterations,"
                   "t_read,t_scatter,t_compute,t_halo,t_gather,t_write,t_total\n");
    fprintf(f, "%d,%d,%d,%s,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
            procs, width, height, filter, radius, iterations,
            t->t_read, t->t_scatter, t->t_compute, t->t_halo, t->t_gather, t->t_write, t->t_total);
    return fclose(f) == 0 ? 0 : -1;
}
