#!/usr/bin/env python3
"""Independent NumPy reference for the filters, used only by the test suite.

It decodes with Pillow (not our C reader) and uses whole-array NumPy math
(edge padding, summed-area tables, float64) rather than the C kernels' row
rings, so it catches orientation, channel-order, padding and boundary bugs
that a serial-vs-MPI comparison alone cannot. Gaussian results may differ
from C by 1 due to float32 vs float64 accumulation.

    reference.py -i in.bmp -o out.bmp -f FILTER [-r radius] [-s sigma] [-n iterations]
"""
import argparse
import math
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def luma(rgb):
    r, g, b = (rgb[..., i].astype(np.int64) for i in range(3))
    return (299 * r + 587 * g + 114 * b + 500) // 1000


def gray3(y):
    return np.repeat(np.asarray(y, np.uint8)[..., None], 3, axis=2)


def grayscale(rgb, a):
    return gray3(luma(rgb))


def box(rgb, a):
    R = a.r
    h, w, _ = rgb.shape
    p = np.pad(rgb.astype(np.int64), ((R, R), (R, R), (0, 0)), mode="edge")
    # Summed-area table with a zero row/column in front.
    s = np.zeros((p.shape[0] + 1, p.shape[1] + 1, 3), np.int64)
    s[1:, 1:] = p.cumsum(0).cumsum(1)
    k = 2 * R + 1
    total = s[k:k + h, k:k + w] - s[:h, k:k + w] - s[k:k + h, :w] + s[:h, :w]
    n = k * k
    return ((total + n // 2) // n).astype(np.uint8)


def gaussian(rgb, a):
    R = math.ceil(3 * a.s)
    h, w, _ = rgb.shape
    taps = np.exp(-np.arange(-R, R + 1, dtype=np.float64) ** 2 / (2 * a.s * a.s))
    taps /= taps.sum()
    p = np.pad(rgb.astype(np.float64), ((R, R), (R, R), (0, 0)), mode="edge")
    horiz = sum(t * p[:, i:i + w] for i, t in enumerate(taps))
    vert = sum(t * horiz[i:i + h] for i, t in enumerate(taps))
    return np.clip(np.floor(vert + 0.5), 0, 255).astype(np.uint8)


def sobel(rgb, a):
    h, w, _ = rgb.shape
    g = np.pad(luma(rgb), 1, mode="edge")
    win = lambda dy, dx: g[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
    gx = (win(-1, 1) + 2 * win(0, 1) + win(1, 1)) - (win(-1, -1) + 2 * win(0, -1) + win(1, -1))
    gy = (win(1, -1) + 2 * win(1, 0) + win(1, 1)) - (win(-1, -1) + 2 * win(-1, 0) + win(-1, 1))
    mag = np.floor(np.sqrt((gx * gx + gy * gy).astype(np.float64)) + 0.5)
    return gray3(np.minimum(mag, 255))


FILTERS = {"grayscale": grayscale, "box": box, "gaussian": gaussian, "sobel": sobel}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", required=True)
    p.add_argument("-o", required=True)
    p.add_argument("-f", required=True, choices=sorted(FILTERS))
    p.add_argument("-r", type=int, default=1)
    p.add_argument("-s", type=float, default=1.0)
    p.add_argument("-n", type=int, default=1)
    a = p.parse_args()
    with Image.open(a.i) as im:
        rgb = np.asarray(im.convert("RGB"))
    for _ in range(a.n):
        rgb = FILTERS[a.f](rgb, a)
    Image.fromarray(rgb, "RGB").save(a.o, format="BMP")
    return 0


if __name__ == "__main__":
    sys.exit(main())
