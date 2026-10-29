#!/usr/bin/env bash
# Correctness suite. Exits nonzero if any check fails.
#   BIN_DIR=bin/debug scripts/run_tests.sh   # run against the ASan build
set -u
cd "$(dirname "$0")/.."

BIN_DIR=${BIN_DIR:-bin}
PY=${PYTHON:-python3}
MPIRUN=${MPIRUN:-mpirun}     # e.g. MPIRUN="mpirun --oversubscribe" on small nodes
PROCS=${PROCS:-"1 2 3 4 7"}
TIME_LIMIT=${TIME_LIMIT:-120} # seconds; a run that exceeds this counts as a hang
FIX=tests/images            # small fixtures (git-tracked)
WORK=build/test-work        # outputs and the large fixture (git-ignored)
mkdir -p "$FIX" "$WORK"
# Start from a clean slate so a stale output can never satisfy a comparison.
find "$WORK" -name '*.bmp' ! -name 'img_4000x3000.bmp' -delete

for b in filter_serial filter_mpi bmp_roundtrip; do
    [ -x "$BIN_DIR/$b" ] || { echo "missing $BIN_DIR/$b; run 'make' (or 'make debug') first" >&2; exit 2; }
done

SIZES="1x1 3x7 101x37 1023x769"
# Filter configurations: "name|halo rows|flags". Radius 4 is wider than the
# 1x1 and 3x7 images, which exercises clamping on both axes at once.
CONFIGS="grayscale|0|-f grayscale
box_r1|1|-f box -r 1
box_r4|4|-f box -r 4
gauss_s1|3|-f gaussian -s 1
gauss_s2.5|8|-f gaussian -s 2.5
sobel|1|-f sobel"
# Gaussian accumulates in float32 in C but float64 in the reference.
tolerance() { case $1 in gauss*) echo 1 ;; *) echo 0 ;; esac; }

pass=0
fail=0
ok()  { pass=$((pass + 1)); }
bad() { fail=$((fail + 1)); echo "FAIL: $*"; }

# check DESCRIPTION CMD...  -- passes if CMD exits 0
check() {
    local desc=$1 out; shift
    if out=$("$@" 2>&1); then ok; else bad "$desc"; echo "$out" | sed 's/^/    /'; fi
}

# expect_error DESCRIPTION PATTERN CMD...  -- passes if CMD fails and stderr matches PATTERN
expect_error() {
    local desc=$1 pattern=$2; shift 2
    local out
    if out=$("$@" 2>&1); then
        bad "$desc (exited 0)"
    elif ! grep -qiE "$pattern" <<<"$out"; then
        bad "$desc (message did not match /$pattern/)"; echo "$out" | sed 's/^/    /'
    else
        ok
    fi
}

section() { echo "== $*"; }

# limited CMD...  -- run CMD, killing it after TIME_LIMIT seconds (portable:
# macOS has no `timeout`). The alarm survives exec, so it lands on mpirun.
limited() { perl -e 'alarm shift; exec @ARGV or die "exec: $!"' "$TIME_LIMIT" "$@"; }
# stdin from /dev/null: mpirun forwards stdin to rank 0 and would otherwise
# swallow the rest of a `while read` loop's input.
mpi() { local np=$1; shift; limited $MPIRUN -np "$np" "$BIN_DIR/filter_mpi" "$@" </dev/null; }

# ---------------------------------------------------------------- fixtures
section "generating fixtures"
# fixture FILE CMD...  -- create FILE with CMD unless it already exists
fixture() { local f=$1; shift; [ -f "$f" ] || check "generate $f" "$@"; }
for s in $SIZES; do
    w=${s%x*}; h=${s#*x}
    fixture "$FIX/img_$s.bmp" $PY tools/gen_image.py "$w" "$h" "$FIX/img_$s.bmp" --seed "$w"
done
fixture "$FIX/topdown_101x37.bmp" $PY tools/gen_image.py 101 37 "$FIX/topdown_101x37.bmp" --seed 101 --top-down
fixture "$FIX/rgba32_8x8.bmp"     $PY tools/gen_image.py 8 8 "$FIX/rgba32_8x8.bmp" --bpp 32
fixture "$FIX/not_a_bmp.bmp"      sh -c "printf 'this is not a bitmap image, just text\n%.0s' 1 2 3 > '$FIX/not_a_bmp.bmp'"
# Valid headers, pixel data cut short.
fixture "$FIX/truncated_101x37.bmp" sh -c "head -c 2000 '$FIX/img_101x37.bmp' > '$FIX/truncated_101x37.bmp'"
# Valid 24-bit file with biCompression patched to 3 (BI_BITFIELDS).
fixture "$FIX/compressed_3x7.bmp" $PY -c "
d = bytearray(open('$FIX/img_3x7.bmp', 'rb').read()); d[30] = 3
open('$FIX/compressed_3x7.bmp', 'wb').write(d)"
fixture "$WORK/img_4000x3000.bmp" $PY tools/gen_image.py 4000 3000 "$WORK/img_4000x3000.bmp" --seed 4000
ALL_IMAGES="$(for s in $SIZES; do echo "$FIX/img_$s.bmp"; done) $WORK/img_4000x3000.bmp"

# ---------------------------------------------------------------- BMP I/O
section "BMP round-trip (read then write unchanged)"
for img in $ALL_IMAGES; do
    out="$WORK/rt_$(basename "$img")"
    check "round-trip run $img" "$BIN_DIR/bmp_roundtrip" "$img" "$out"
    check "round-trip identical $img" cmp -s "$img" "$out"
done

# A BMP written by Pillow (different resolution fields) must also survive.
check "Pillow writes a BMP" \
    $PY -c "from PIL import Image; Image.open('$FIX/img_101x37.bmp').save('$WORK/pillow_101x37.bmp')"
check "round-trip run (Pillow-written file)" "$BIN_DIR/bmp_roundtrip" "$WORK/pillow_101x37.bmp" "$WORK/rt_pillow.bmp"
check "round-trip identical (Pillow-written file)" cmp -s "$WORK/pillow_101x37.bmp" "$WORK/rt_pillow.bmp"

# Top-down input: same pixels as the bottom-up file with the same seed.
check "top-down read" "$BIN_DIR/bmp_roundtrip" "$FIX/topdown_101x37.bmp" "$WORK/rt_topdown.bmp"
check "top-down pixels match bottom-up" $PY tools/compare.py --pixels "$WORK/rt_topdown.bmp" "$FIX/img_101x37.bmp"
check "top-down output rewritten bottom-up, identical to original" cmp -s "$WORK/rt_topdown.bmp" "$FIX/img_101x37.bmp"

# ---------------------------------------------------------------- serial vs reference
section "serial filters vs NumPy reference"
# serial_vs_reference IMAGE NAME ITERATIONS FLAGS...
serial_vs_reference() {
    local img=$1 name=$2 n=$3; shift 3
    local tag="${name}_n${n}_$(basename "$img" .bmp)"
    local s="$WORK/serial_$tag.bmp" r="$WORK/ref_$tag.bmp"
    check "serial $tag" "$BIN_DIR/filter_serial" -i "$img" -o "$s" -n "$n" "$@"
    check "reference $tag" $PY tools/reference.py -i "$img" -o "$r" -n "$n" "$@"
    check "serial == reference ($tag)" $PY tools/compare.py --pixels -t "$(tolerance "$name")" "$s" "$r"
    check "Pillow opens serial output ($tag)" $PY tools/compare.py --validate "$s"
}

while IFS='|' read -r name _halo flags; do
    for img in $ALL_IMAGES; do
        # shellcheck disable=SC2086  # flags are intentionally word-split
        serial_vs_reference "$img" "$name" 1 $flags
    done
    # Iterations feed each output back in as input (small images only).
    for s in $SIZES; do
        # shellcheck disable=SC2086
        serial_vs_reference "$FIX/img_$s.bmp" "$name" 3 $flags
    done
done <<<"$CONFIGS"

# ---------------------------------------------------------------- MPI vs serial
section "MPI vs serial (byte-identical), P in {$PROCS}"
MPI_CONFIGS=$CONFIGS

# image_height IMAGE  -- parsed from the img_WxH.bmp naming convention
image_height() { local b=${1##*x}; echo "${b%.bmp}"; }

# mpi_vs_serial IMAGE NAME HALO ITERATIONS FLAGS...  -- compares against the
# serial output produced above for the same tag. When P would leave a band
# with fewer than HALO rows, the run must instead be refused cleanly.
mpi_vs_serial() {
    local img=$1 name=$2 halo=$3 n=$4; shift 4
    local tag="${name}_n${n}_$(basename "$img" .bmp)"
    local h; h=$(image_height "$img")
    for p in $PROCS; do
        local m="$WORK/mpi_p${p}_$tag.bmp"
        if [ "$halo" -gt 0 ] && [ $((h / p)) -lt "$halo" ]; then
            expect_error "mpi P=$p $tag refused (bands < $halo rows)" "use at most" \
                mpi "$p" -i "$img" -o "$m" -n "$n" "$@"
            continue
        fi
        check "mpi P=$p $tag" mpi "$p" -i "$img" -o "$m" -n "$n" "$@"
        check "mpi P=$p == serial ($tag)" $PY tools/compare.py "$m" "$WORK/serial_$tag.bmp"
    done
}

while IFS='|' read -r name halo flags; do
    for img in $ALL_IMAGES; do
        # shellcheck disable=SC2086
        mpi_vs_serial "$img" "$name" "$halo" 1 $flags
    done
    for s in $SIZES; do
        # shellcheck disable=SC2086
        mpi_vs_serial "$FIX/img_$s.bmp" "$name" "$halo" 3 $flags
    done
done <<<"$MPI_CONFIGS"

section "MPI halo edge cases"
# 37 rows over 7 ranks -> bands 6,6,5,5,5,5,5: radius 5 is the tightest legal
# case (smallest band == halo), radius 6 must be refused.
check "serial baseline" "$BIN_DIR/filter_serial" -i "$FIX/img_101x37.bmp" -o "$WORK/serial_box_r5_n2_img_101x37.bmp" -f box -r 5 -n 2
PROCS=7 mpi_vs_serial "$FIX/img_101x37.bmp" box_r5 5 2 -f box -r 5
expect_error "mpi P=7 box r6 on 37 rows refused" "use at most 6 processes" \
    mpi 7 -i "$FIX/img_101x37.bmp" -o "$WORK/x.bmp" -f box -r 6
# Many iterations: halos must be refreshed every pass.
check "serial baseline" "$BIN_DIR/filter_serial" -i "$FIX/img_1023x769.bmp" -o "$WORK/serial_box_r2_n10_img_1023x769.bmp" -f box -r 2 -n 10
PROCS="4 7" mpi_vs_serial "$FIX/img_1023x769.bmp" box_r2 2 10 -f box -r 2
check "serial baseline" "$BIN_DIR/filter_serial" -i "$FIX/img_1023x769.bmp" -o "$WORK/serial_gauss_s2_n10_img_1023x769.bmp" -f gaussian -s 2 -n 10
PROCS="4 7" mpi_vs_serial "$FIX/img_1023x769.bmp" gauss_s2 6 10 -f gaussian -s 2
check "serial baseline" "$BIN_DIR/filter_serial" -i "$FIX/img_1023x769.bmp" -o "$WORK/serial_sobel_n10_img_1023x769.bmp" -f sobel -n 10
PROCS="4 7" mpi_vs_serial "$FIX/img_1023x769.bmp" sobel 1 10 -f sobel
# Gaussian halo is ceil(3*sigma): sigma 2.5 -> 8 rows, so 37 rows allow P <= 4.
expect_error "mpi P=7 gaussian s2.5 on 37 rows refused" "use at most 4 processes" \
    mpi 7 -i "$FIX/img_101x37.bmp" -o "$WORK/x.bmp" -f gaussian -s 2.5
# Top-down input through MPI: same pixels as the bottom-up fixture, so the
# output must equal the serial result for that fixture.
for p in 3 7; do
    check "mpi P=$p top-down input" mpi "$p" -i "$FIX/topdown_101x37.bmp" -o "$WORK/mpi_p${p}_topdown_box.bmp" -f box -r 1
    check "mpi P=$p top-down == serial bottom-up" \
        $PY tools/compare.py "$WORK/mpi_p${p}_topdown_box.bmp" "$WORK/serial_box_r1_n1_img_101x37.bmp"
done

section "Pillow opens every MPI output"
mpi_outputs=$(find "$WORK" -name 'mpi_p*.bmp' | sort)
echo "   ($(wc -l <<<"$mpi_outputs" | tr -d ' ') files)"
# shellcheck disable=SC2086
check "Pillow validates all MPI outputs" $PY tools/compare.py --validate $mpi_outputs

# ---------------------------------------------------------------- error paths
section "error paths (MPI, P=4: must fail cleanly, no hang)"
expect_error "mpi non-BMP input"  "not a BMP"     mpi 4 -i "$FIX/not_a_bmp.bmp" -o "$WORK/x.bmp" -f grayscale
expect_error "mpi 32-bit BMP"     "bit depth 32"  mpi 4 -i "$FIX/rgba32_8x8.bmp" -o "$WORK/x.bmp" -f grayscale
expect_error "mpi missing file"   "no such file"  mpi 4 -i "$WORK/does_not_exist.bmp" -o "$WORK/x.bmp" -f grayscale
expect_error "mpi unknown filter" "unknown filter" mpi 4 -i "$FIX/img_3x7.bmp" -o "$WORK/x.bmp" -f sharpen
expect_error "mpi missing -o"     "required"      mpi 4 -i "$FIX/img_3x7.bmp" -f grayscale
expect_error "mpi unwritable output" "no such file" mpi 4 -i "$FIX/img_3x7.bmp" -o "$WORK/no/such/dir/x.bmp" -f grayscale
expect_error "mpi truncated BMP"  "truncated"     mpi 4 -i "$FIX/truncated_101x37.bmp" -o "$WORK/x.bmp" -f box
expect_error "mpi compressed BMP" "compression"   mpi 4 -i "$FIX/compressed_3x7.bmp" -o "$WORK/x.bmp" -f sobel
expect_error "mpi bad sigma"      "invalid sigma" mpi 4 -i "$FIX/img_3x7.bmp" -o "$WORK/x.bmp" -f gaussian -s 0

section "error paths (serial)"
expect_error "non-BMP input"    "not a BMP"             "$BIN_DIR/filter_serial" -i "$FIX/not_a_bmp.bmp" -o "$WORK/x.bmp" -f grayscale
expect_error "32-bit BMP"       "bit depth 32"          "$BIN_DIR/filter_serial" -i "$FIX/rgba32_8x8.bmp" -o "$WORK/x.bmp" -f grayscale
expect_error "truncated BMP"    "truncated"             "$BIN_DIR/filter_serial" -i "$FIX/truncated_101x37.bmp" -o "$WORK/x.bmp" -f grayscale
expect_error "compressed BMP"   "compression"           "$BIN_DIR/filter_serial" -i "$FIX/compressed_3x7.bmp" -o "$WORK/x.bmp" -f grayscale
expect_error "missing file"     "no such file"          "$BIN_DIR/filter_serial" -i "$WORK/does_not_exist.bmp" -o "$WORK/x.bmp" -f grayscale
expect_error "unknown filter"   "unknown filter"        "$BIN_DIR/filter_serial" -i "$FIX/img_3x7.bmp" -o "$WORK/x.bmp" -f sharpen
expect_error "bad radius"       "invalid radius"        "$BIN_DIR/filter_serial" -i "$FIX/img_3x7.bmp" -o "$WORK/x.bmp" -f box -r 0
expect_error "bad sigma"        "invalid sigma"         "$BIN_DIR/filter_serial" -i "$FIX/img_3x7.bmp" -o "$WORK/x.bmp" -f gaussian -s -2
expect_error "bad iterations"   "invalid iteration"     "$BIN_DIR/filter_serial" -i "$FIX/img_3x7.bmp" -o "$WORK/x.bmp" -f box -n 0
expect_error "missing -o"       "required"              "$BIN_DIR/filter_serial" -i "$FIX/img_3x7.bmp" -f grayscale

echo
echo "passed: $pass  failed: $fail"
[ "$fail" -eq 0 ]
