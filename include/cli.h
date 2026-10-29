#ifndef CLI_H
#define CLI_H

#include <stddef.h>

#include "filters.h"

typedef struct {
    const char *input;
    const char *output;
    const char *timing_csv;   /* NULL if not requested */
    FilterKind filter;
    int radius;               /* box radius (-r) */
    double sigma;             /* gaussian sigma (-s) */
    int iterations;           /* -n */
} Options;

/* Parse and validate argv. Returns 0 on success, 1 if help was requested
 * (usage already in err), -1 on error with a message in err. Shared by the
 * serial and MPI drivers so both accept identical flags. */
int cli_parse(int argc, char **argv, Options *opt, char *err, size_t errlen);

const char *cli_filter_name(FilterKind f);

#endif
