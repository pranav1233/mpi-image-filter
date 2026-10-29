/* MPI driver: rank 0 reads and validates, rows are scattered as bands,
 * every rank filters its band with the shared kernels, rank 0 gathers and
 * writes. */
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#include "bmp.h"
#include "cli.h"
#include "decomp.h"
#include "filters.h"

enum { RUN = 0, EXIT_OK = 1, EXIT_FAIL = 2 };

/* Everything non-root ranks need to know, broadcast from rank 0. */
typedef struct {
    int status;        /* RUN, EXIT_OK (e.g. --help) or EXIT_FAIL */
    int width, height;
    int filter, radius, iterations;
    double sigma;
} Params;

/* Rank 0 only: parse args, read the image and check that the run is
 * possible. Any problem is reported here and turned into a status, so all
 * ranks can exit together after the broadcast. */
static int setup_root(int argc, char **argv, int nprocs, Options *opt, Image *img)
{
    char err[512];
    int rc = cli_parse(argc, argv, opt, err, sizeof err);
    if (rc != 0) {
        fprintf(rc > 0 ? stdout : stderr, rc > 0 ? "%s" : "error: %s\n", err);
        return rc > 0 ? EXIT_OK : EXIT_FAIL;
    }
    int halo = filter_halo_rows(opt->filter, opt->radius, opt->sigma);
    if (bmp_read(opt->input, img, err, sizeof err) != 0) {
        fprintf(stderr, "error: %s\n", err);
        return EXIT_FAIL;
    }
    if (img->width > INT_MAX / 3) {
        fprintf(stderr, "error: image width %d too large for the row datatype\n", img->width);
        return EXIT_FAIL;
    }
    /* The smallest band has height / nprocs rows; a halo can only come from
     * the immediate neighbor, so every band must be at least `halo` rows. */
    if (halo > 0 && img->height / nprocs < halo) {
        fprintf(stderr,
                "error: %d processes split %d rows into bands of %d, but filter '%s' needs "
                "%d halo rows per band; use at most %d processes\n",
                nprocs, img->height, img->height / nprocs, cli_filter_name(opt->filter), halo,
                img->height / halo);
        return EXIT_FAIL;
    }
    return RUN;
}

static void broadcast_params(Params *p)
{
    int ints[6] = {p->status, p->width, p->height, p->filter, p->radius, p->iterations};
    MPI_CHECK(MPI_Bcast(ints, 6, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(&p->sigma, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD));
    p->status = ints[0];
    p->width = ints[1];
    p->height = ints[2];
    p->filter = ints[3];
    p->radius = ints[4];
    p->iterations = ints[5];
}

static void *xmalloc(size_t n, int rank)
{
    void *p = malloc(n ? n : 1);
    if (!p) {
        fprintf(stderr, "rank %d: out of memory allocating %zu bytes\n", rank, n);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return p;
}

/* Scatter, filter `iterations` times, gather. On rank 0 img->data receives
 * the result. */
static void run_filter(const Params *p, Image *img)
{
    Decomp d;
    decomp_init(&d, MPI_COMM_WORLD, p->width, p->height);

    FilterPlan plan;
    if (filter_plan_init(&plan, (FilterKind)p->filter, p->radius, p->sigma, p->width) != 0) {
        fprintf(stderr, "rank %d: out of memory\n", d.rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Each band buffer: `halo` rows, the owned rows, `halo` rows. */
    size_t rb = (size_t)p->width * 3;
    size_t band_bytes = ((size_t)d.rows + 2 * (size_t)plan.halo) * rb;
    size_t off = (size_t)plan.halo * rb;
    uint8_t *buf[2] = {xmalloc(band_bytes, d.rank), xmalloc(band_bytes, d.rank)};

    decomp_scatter(&d, d.rank == 0 ? img->data : NULL, buf[0] + off);

    /* Data stays distributed across iterations: no gather/rescatter. */
    int cur = 0;
    for (int it = 0; it < p->iterations; it++) {
        decomp_exchange_halo(&d, buf[cur] + off, plan.halo);
        filter_apply(&plan, buf[cur] + off, buf[1 - cur] + off, d.rows);
        cur = 1 - cur;
    }

    decomp_gather(&d, buf[cur] + off, d.rank == 0 ? img->data : NULL);

    free(buf[0]);
    free(buf[1]);
    filter_plan_free(&plan);
    decomp_free(&d);
}

int main(int argc, char **argv)
{
    MPI_CHECK(MPI_Init(&argc, &argv));
    MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    int rank, nprocs;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));

    Options opt;
    Image img = {0};
    Params p = {0};
    if (rank == 0) {
        p.status = setup_root(argc, argv, nprocs, &opt, &img);
        if (p.status == RUN) {
            p.width = img.width;
            p.height = img.height;
            p.filter = (int)opt.filter;
            p.radius = opt.radius;
            p.iterations = opt.iterations;
            p.sigma = opt.sigma;
        }
    }
    broadcast_params(&p);

    int exit_code = p.status == EXIT_FAIL ? EXIT_FAILURE : EXIT_SUCCESS;
    if (p.status == RUN) {
        run_filter(&p, &img);
        if (rank == 0) {
            char err[512];
            if (bmp_write(opt.output, &img, err, sizeof err) != 0) {
                fprintf(stderr, "error: %s\n", err);
                exit_code = EXIT_FAILURE;
            }
        }
    }

    if (rank == 0)
        image_free(&img);
    MPI_CHECK(MPI_Finalize());
    return exit_code;
}
