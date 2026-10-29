#include "decomp.h"

#include <stdio.h>
#include <stdlib.h>

#include "filters.h"

enum { TAG_TO_UPPER = 1, TAG_TO_LOWER = 2 };

void mpi_fail(const char *what, int rc, const char *file, int line)
{
    char msg[MPI_MAX_ERROR_STRING];
    int len = 0;
    MPI_Error_string(rc, msg, &len);
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(stderr, "rank %d: %s failed at %s:%d: %s\n", rank, what, file, line, msg);
    MPI_Abort(MPI_COMM_WORLD, 1);
}

/* Uneven split: every rank gets height / nprocs rows and the first
 * height % nprocs ranks get one extra. E.g. 10 rows over 4 ranks: 3,3,2,2. */
int decomp_rows(int rank, int nprocs, int height)
{
    return height / nprocs + (rank < height % nprocs ? 1 : 0);
}

/* Start of `rank`'s band: all earlier ranks hold height / nprocs rows, and
 * min(rank, height % nprocs) of them hold one extra. */
int decomp_start(int rank, int nprocs, int height)
{
    int extra = height % nprocs;
    return rank * (height / nprocs) + (rank < extra ? rank : extra);
}

void decomp_init(Decomp *d, MPI_Comm comm, int width, int height)
{
    d->comm = comm;
    MPI_CHECK(MPI_Comm_rank(comm, &d->rank));
    MPI_CHECK(MPI_Comm_size(comm, &d->nprocs));
    d->width = width;
    d->height = height;
    d->rows = decomp_rows(d->rank, d->nprocs, height);
    d->start = decomp_start(d->rank, d->nprocs, height);

    d->counts = malloc((size_t)d->nprocs * sizeof(int));
    d->displs = malloc((size_t)d->nprocs * sizeof(int));
    if (!d->counts || !d->displs) {
        fprintf(stderr, "rank %d: out of memory\n", d->rank);
        MPI_Abort(comm, 1);
    }
    for (int r = 0; r < d->nprocs; r++) {
        d->counts[r] = decomp_rows(r, d->nprocs, height);
        d->displs[r] = decomp_start(r, d->nprocs, height);
    }

    /* All counts and displacements are in rows of this type, so they stay
     * small ints even when the image itself exceeds 2 GB. */
    MPI_CHECK(MPI_Type_contiguous(width * 3, MPI_UNSIGNED_CHAR, &d->row_t));
    MPI_CHECK(MPI_Type_commit(&d->row_t));
}

void decomp_free(Decomp *d)
{
    MPI_CHECK(MPI_Type_free(&d->row_t));
    free(d->counts);
    free(d->displs);
    d->counts = d->displs = NULL;
}

void decomp_scatter(const Decomp *d, const uint8_t *full, uint8_t *band)
{
    MPI_CHECK(MPI_Scatterv(full, d->counts, d->displs, d->row_t,
                           band, d->rows, d->row_t, 0, d->comm));
}

void decomp_gather(const Decomp *d, const uint8_t *band, uint8_t *full)
{
    MPI_CHECK(MPI_Gatherv(band, d->rows, d->row_t,
                          full, d->counts, d->displs, d->row_t, 0, d->comm));
}

/* Buffer layout around `band` (R = halo, n = d->rows, rows of width*3 bytes):
 *
 *   band - R      top halo     <- last R rows of rank-1
 *   band          owned row 0  -> sent to rank-1 (its bottom halo)
 *   ...
 *   band + n-R    owned rows   -> sent to rank+1 (its top halo)
 *   band + n      bottom halo  <- first R rows of rank+1
 *
 * At the image edges the neighbor is MPI_PROC_NULL, which turns that half of
 * each Sendrecv into a no-op; those halos are then filled by replication. */
void decomp_exchange_halo(const Decomp *d, uint8_t *band, int halo)
{
    if (halo == 0)
        return;
    size_t rb = (size_t)d->width * 3;
    int upper = d->rank > 0 ? d->rank - 1 : MPI_PROC_NULL;
    int lower = d->rank < d->nprocs - 1 ? d->rank + 1 : MPI_PROC_NULL;
    uint8_t *top_halo = band - (size_t)halo * rb;
    uint8_t *bottom_halo = band + (size_t)d->rows * rb;
    uint8_t *last_owned = band + (size_t)(d->rows - halo) * rb;

    /* Everyone sends up and receives from below, then the reverse. Sendrecv
     * pairs the two halves so this cannot deadlock. */
    MPI_CHECK(MPI_Sendrecv(band, halo, d->row_t, upper, TAG_TO_UPPER,
                           bottom_halo, halo, d->row_t, lower, TAG_TO_UPPER,
                           d->comm, MPI_STATUS_IGNORE));
    MPI_CHECK(MPI_Sendrecv(last_owned, halo, d->row_t, lower, TAG_TO_LOWER,
                           top_halo, halo, d->row_t, upper, TAG_TO_LOWER,
                           d->comm, MPI_STATUS_IGNORE));

    filter_fill_edge_halo(band, d->width, d->rows, halo,
                          upper == MPI_PROC_NULL, lower == MPI_PROC_NULL);
}
