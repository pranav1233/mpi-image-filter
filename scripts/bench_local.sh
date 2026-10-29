#!/usr/bin/env bash
# Local strong/weak scaling sweep. Writes one CSV row per run (all repeats;
# tools/plot_scaling.py takes the median) and then draws the plots.
#
# Everything is overridable from the environment, e.g. a quick smoke run:
#   PROCS="1 2" REPEATS=1 STRONG_SIZE=2000x1500 WEAK_WIDTH=2000 WEAK_ROWS=500 scripts/bench_local.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PROCS=${PROCS:-"1 2 4 8"}
REPEATS=${REPEATS:-3}
STRONG_SIZE=${STRONG_SIZE:-16000x12000}   # ~576 MB
WEAK_WIDTH=${WEAK_WIDTH:-16000}
WEAK_ROWS=${WEAK_ROWS:-2000}               # rows per process
SIGMA=${SIGMA:-3}
ITERS=${ITERS:-10}
MPIRUN=${MPIRUN:-mpirun}
MPIRUN_FLAGS=${MPIRUN_FLAGS:-}
OUT=${OUT:-results/local}
IMG=${IMG:-build/bench}                    # large inputs/outputs, git-ignored
PY=${PYTHON:-python3}

[ -x bin/filter_mpi ] || { echo "build first: make" >&2; exit 2; }
mkdir -p "$OUT" "$IMG"
rm -f "$OUT/strong.csv" "$OUT/weak.csv"

image() {   # image W H  -> path, generated once
    local f="$IMG/bench_$1x$2.bmp"
    [ -f "$f" ] || { echo "   generating $1x$2" >&2; $PY tools/gen_image.py "$1" "$2" "$f" --seed 7; }
    echo "$f"
}

run() {     # run P CSV INPUT FILTER-FLAGS...
    local p=$1 csv=$2 in=$3; shift 3
    # shellcheck disable=SC2086
    $MPIRUN $MPIRUN_FLAGS -np "$p" bin/filter_mpi -i "$in" -o "$IMG/out.bmp" "$@" \
        --timing-csv "$csv" </dev/null
}

echo "== strong scaling: $STRONG_SIZE, P in {$PROCS}, $REPEATS repeats"
strong=$(image "${STRONG_SIZE%x*}" "${STRONG_SIZE#*x}")
# Repeats are the outer loop so slow drift (thermals, background load) is
# spread across all P instead of landing on one of them.
for r in $(seq "$REPEATS"); do
    for p in $PROCS; do
        echo "   rep $r  P=$p"
        run "$p" "$OUT/strong.csv" "$strong" -f gaussian -s "$SIGMA" -n "$ITERS"
        run "$p" "$OUT/strong.csv" "$strong" -f grayscale
    done
done

echo "== weak scaling: $WEAK_ROWS rows/process x $WEAK_WIDTH wide, P in {$PROCS}"
for r in $(seq "$REPEATS"); do
    for p in $PROCS; do
        echo "   rep $r  P=$p"
        run "$p" "$OUT/weak.csv" "$(image "$WEAK_WIDTH" $((WEAK_ROWS * p)))" -f gaussian -s "$SIGMA" -n "$ITERS"
    done
done

rm -f "$IMG/out.bmp"
$PY tools/plot_scaling.py --strong "$OUT/strong.csv" --weak "$OUT/weak.csv" --out "$OUT" \
    --label "local, $(sysctl -n machdep.cpu.brand_string 2>/dev/null || hostname)"
