#!/usr/bin/env python3
"""Generate a deterministic synthetic BMP of arbitrary size.

The file is written by hand (not through Pillow) so we control every header
field and can emit the variants the tests need: bottom-up or top-down 24-bit,
and 32-bit (which the filter must reject).

    gen_image.py WIDTH HEIGHT OUT.bmp [--seed N] [--bpp 24|32] [--top-down]
"""
import argparse
import struct
import sys

import numpy as np

CHUNK_ROWS = 512


def make_pixels(width, height, seed):
    """RGB uint8 array (height, width, 3): gradients, rings and noise, so every
    filter has edges, smooth regions and texture to work on."""
    rng = np.random.default_rng(seed)
    img = np.empty((height, width, 3), dtype=np.uint8)
    x = np.arange(width, dtype=np.float32)
    cx, cy = width * 0.4, height * 0.6
    period = max(8.0, min(width, height) / 6.0)
    for y0 in range(0, height, CHUNK_ROWS):
        y1 = min(height, y0 + CHUNK_ROWS)
        y = np.arange(y0, y1, dtype=np.float32)[:, None]
        r = np.sqrt((x[None, :] - cx) ** 2 + (y - cy) ** 2)
        rings = (np.sin(r * (2 * np.pi / period)) > 0).astype(np.float32)
        noise = rng.integers(0, 48, size=(y1 - y0, width, 3), dtype=np.int16)
        red = 255.0 * x[None, :] / max(1, width - 1) * 0.7 + rings * 60
        green = 255.0 * y / max(1, height - 1) * 0.7 + rings * 40
        blue = 128.0 + 80.0 * np.sin(x[None, :] / 23.0 + y / 17.0)
        chunk = np.stack(np.broadcast_arrays(red, green, blue), axis=-1)
        img[y0:y1] = np.clip(chunk + noise, 0, 255).astype(np.uint8)
    return img


def write_bmp(path, rgb, bpp=24, top_down=False):
    height, width, _ = rgb.shape
    bytes_pp = bpp // 8
    stride = ((width * bytes_pp + 3) // 4) * 4
    image_size = stride * height
    offset = 14 + 40
    file_header = struct.pack("<2sIHHI", b"BM", offset + image_size, 0, 0, offset)
    info_header = struct.pack(
        "<IiiHHIIiiII",
        40, width, -height if top_down else height, 1, bpp, 0,
        image_size, 2835, 2835, 0, 0,
    )
    with open(path, "wb") as f:
        f.write(file_header)
        f.write(info_header)
        order = range(height) if top_down else range(height - 1, -1, -1)
        row = np.zeros(stride, dtype=np.uint8)
        for y in order:
            bgr = rgb[y, :, ::-1]
            if bpp == 32:
                bgr = np.concatenate([bgr, np.full((width, 1), 255, np.uint8)], axis=1)
            row[: width * bytes_pp] = bgr.reshape(-1)
            f.write(row.tobytes())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("width", type=int)
    p.add_argument("height", type=int)
    p.add_argument("out")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bpp", type=int, choices=(24, 32), default=24)
    p.add_argument("--top-down", action="store_true", help="negative biHeight (rows stored top to bottom)")
    a = p.parse_args()
    if a.width < 1 or a.height < 1:
        sys.exit("width and height must be >= 1")
    write_bmp(a.out, make_pixels(a.width, a.height, a.seed), a.bpp, a.top_down)


if __name__ == "__main__":
    main()
