#!/usr/bin/env python3
"""Compare BMP files, or validate one with Pillow.

    compare.py A.bmp B.bmp                 byte-identical check (default)
    compare.py A.bmp B.bmp --pixels [-t N] decoded pixels within tolerance N
    compare.py --validate OUT.bmp ...      Pillow can open it as a 24-bit RGB BMP

Exit status: 0 = match/valid, 1 = mismatch/invalid, 2 = usage or I/O error.
"""
import argparse
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # benchmark images are legitimately huge


def load_rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def compare_bytes(a, b):
    with open(a, "rb") as fa, open(b, "rb") as fb:
        da, db = fa.read(), fb.read()
    if da == db:
        print(f"OK identical ({len(da)} bytes)")
        return 0
    if len(da) != len(db):
        print(f"DIFF size {len(da)} vs {len(db)} bytes")
    else:
        diff = np.flatnonzero(np.frombuffer(da, np.uint8) != np.frombuffer(db, np.uint8))
        print(f"DIFF {diff.size} bytes differ, first at offset {diff[0]}")
    return 1


def compare_pixels(a, b, tol):
    pa, pb = load_rgb(a), load_rgb(b)
    if pa.shape != pb.shape:
        print(f"DIFF shape {pa.shape} vs {pb.shape}")
        return 1
    d = np.abs(pa.astype(np.int16) - pb.astype(np.int16))
    worst = int(d.max()) if d.size else 0
    if worst <= tol:
        print(f"OK pixels match (max diff {worst}, tolerance {tol})")
        return 0
    y, x, c = np.unravel_index(int(d.argmax()), d.shape)
    bad = int((d > tol).sum())
    print(f"DIFF {bad} channel values exceed tolerance {tol}; worst {worst} at (x={x}, y={y}, ch={'RGB'[c]})")
    return 1


def validate(path):
    try:
        with Image.open(path) as im:
            im.load()
            if im.format != "BMP" or im.mode != "RGB":
                print(f"INVALID {path}: format={im.format} mode={im.mode}")
                return 1
            print(f"OK {path}: {im.size[0]}x{im.size[1]} {im.mode} BMP")
            return 0
    except Exception as e:  # Pillow raises many exception types for bad files
        print(f"INVALID {path}: {e}")
        return 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+")
    p.add_argument("--pixels", action="store_true")
    p.add_argument("-t", "--tolerance", type=int, default=0)
    p.add_argument("--validate", action="store_true")
    a = p.parse_args()
    try:
        if a.validate:
            return max(validate(f) for f in a.files)
        if len(a.files) != 2:
            p.error("need exactly two files to compare")
        if a.pixels:
            return compare_pixels(*a.files, a.tolerance)
        return compare_bytes(*a.files)
    except OSError as e:
        print(f"ERROR {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
