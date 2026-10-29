#ifndef DECOMP_H
#define DECOMP_H

#include <stdint.h>

#include <mpi.h>

/* Abort every rank if an MPI call fails. MPI_COMM_WORLD is switched to
 * MPI_ERRORS_RETURN at startup so errors reach this check instead of
 * terminating inside the library. */
#define MPI_CHECK(call)                                              \
    do {                                                             \
        int mpi_rc_ = (call);                                        \
        if (mpi_rc_ != MPI_SUCCESS)                                  \
            mpi_fail(#call, mpi_rc_, __FILE__, __LINE__);            \
    } while (0)

void mpi_fail(const char *what, int rc, const char *file, int line);

/* Row-band decomposition of a width x height image over the ranks of comm. */
typedef struct {
    MPI_Comm comm;
    int rank, nprocs;
    int width, height;
    int rows;              /* rows owned by this rank */
    int start;             /* global index of this rank's first row */
    int *counts;           /* per-rank row counts (all ranks) */
    int *displs;           /* per-rank first row (all ranks) */
    MPI_Datatype row_t;    /* one image row: width * 3 bytes */
} Decomp;

/* Rows owned by `rank` and the global index of its first row, for an image of
 * `height` rows over `nprocs` ranks. */
int decomp_rows(int rank, int nprocs, int height);
int decomp_start(int rank, int nprocs, int height);

/* Collective. Builds the row datatype and count/displacement tables. */
void decomp_init(Decomp *d, MPI_Comm comm, int width, int height);
void decomp_free(Decomp *d);

/* Collective. Root's `full` image (height rows) -> each rank's `band` (rows). */
void decomp_scatter(const Decomp *d, const uint8_t *full, uint8_t *band);

/* Collective. Each rank's `band` -> root's `full` image. */
void decomp_gather(const Decomp *d, const uint8_t *band, uint8_t *full);

/* Collective. Fill the `halo` rows above and below `band` (which points at
 * the first owned row; see filters.h) from the neighboring ranks, and by
 * edge replication at the top and bottom of the image. Requires
 * d->rows >= halo on every rank. */
void decomp_exchange_halo(const Decomp *d, uint8_t *band, int halo);

#endif
