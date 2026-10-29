#include "cli.h"

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_RADIUS 1000
#define MAX_SIGMA 100.0

static const char *USAGE =
    "usage: %s -i in.bmp -o out.bmp -f <grayscale|box|gaussian|sobel>\n"
    "          [-r radius] [-s sigma] [-n iterations] [--timing-csv file.csv]\n";

static int parse_int(const char *s, int *out)
{
    char *end;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (errno || end == s || *end != '\0' || v < INT_MIN || v > INT_MAX)
        return -1;
    *out = (int)v;
    return 0;
}

static int parse_double(const char *s, double *out)
{
    char *end;
    errno = 0;
    double v = strtod(s, &end);
    if (errno || end == s || *end != '\0')
        return -1;
    *out = v;
    return 0;
}

static int parse_filter(const char *s, FilterKind *f)
{
    static const FilterKind kinds[] = {FILTER_GRAYSCALE, FILTER_BOX, FILTER_GAUSSIAN, FILTER_SOBEL};
    for (size_t i = 0; i < sizeof kinds / sizeof kinds[0]; i++) {
        if (strcmp(s, cli_filter_name(kinds[i])) == 0) {
            *f = kinds[i];
            return 0;
        }
    }
    return -1;
}

const char *cli_filter_name(FilterKind f)
{
    switch (f) {
    case FILTER_GRAYSCALE: return "grayscale";
    case FILTER_BOX:       return "box";
    case FILTER_GAUSSIAN:  return "gaussian";
    case FILTER_SOBEL:     return "sobel";
    }
    return "?";
}

int cli_parse(int argc, char **argv, Options *opt, char *err, size_t errlen)
{
    const char *prog = argc > 0 ? argv[0] : "filter";
    const char *filter = NULL;
    memset(opt, 0, sizeof *opt);
    opt->radius = 1;
    opt->sigma = 1.0;
    opt->iterations = 1;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (strcmp(a, "-h") == 0 || strcmp(a, "--help") == 0) {
            snprintf(err, errlen, USAGE, prog);
            return 1;
        }
        if (i + 1 >= argc) {
            snprintf(err, errlen, "missing value for %s", a);
            return -1;
        }
        const char *v = argv[++i];
        if (strcmp(a, "-i") == 0) {
            opt->input = v;
        } else if (strcmp(a, "-o") == 0) {
            opt->output = v;
        } else if (strcmp(a, "-f") == 0) {
            filter = v;
        } else if (strcmp(a, "--timing-csv") == 0) {
            opt->timing_csv = v;
        } else if (strcmp(a, "-r") == 0) {
            if (parse_int(v, &opt->radius) || opt->radius < 1 || opt->radius > MAX_RADIUS) {
                snprintf(err, errlen, "invalid radius '%s' (must be 1..%d)", v, MAX_RADIUS);
                return -1;
            }
        } else if (strcmp(a, "-s") == 0) {
            if (parse_double(v, &opt->sigma) || !(opt->sigma > 0.0 && opt->sigma <= MAX_SIGMA)) {
                snprintf(err, errlen, "invalid sigma '%s' (must be in (0, %g])", v, MAX_SIGMA);
                return -1;
            }
        } else if (strcmp(a, "-n") == 0) {
            if (parse_int(v, &opt->iterations) || opt->iterations < 1) {
                snprintf(err, errlen, "invalid iteration count '%s' (must be >= 1)", v);
                return -1;
            }
        } else {
            snprintf(err, errlen, "unknown option '%s'", a);
            return -1;
        }
    }

    if (!opt->input || !opt->output || !filter) {
        snprintf(err, errlen, "-i, -o and -f are required (see --help)");
        return -1;
    }
    if (parse_filter(filter, &opt->filter) != 0) {
        snprintf(err, errlen, "unknown filter '%s' (grayscale, box, gaussian, sobel)", filter);
        return -1;
    }
    return 0;
}
