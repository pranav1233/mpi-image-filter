#!/usr/bin/env python3
"""Speedup, efficiency and time-breakdown plots from the timing CSVs.

    plot_scaling.py --strong strong.csv [--weak weak.csv] --out DIR [--label TEXT]

Repeated runs of the same configuration are reduced to the run with the
median t_total (one real run, so its phase times still add up). Writes
speedup.png, efficiency.png, breakdown.png and summary.md (the same numbers
as tables) into DIR.
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Validated categorical palette (light surface), assigned in fixed order.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["o", "s", "D", "^", "v"]
# Color follows the entity: each filter always gets the same slot.
FILTER_SLOT = {"gaussian": 0, "grayscale": 1, "box": 2, "sobel": 3}
PHASES = [  # (label, columns) bottom to top in the stacked bars
    ("compute", ["t_compute"]),
    ("halo exchange", ["t_halo"]),
    ("scatter + gather", ["t_scatter", "t_gather"]),
    ("file I/O (rank 0)", ["t_read", "t_write"]),
]
TIME_COLS = ["t_read", "t_scatter", "t_compute", "t_halo", "t_gather", "t_write", "t_total"]


def load(path):
    """{config key: {procs: median row}} where key = (filter, radius, iterations)."""
    runs = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for c in TIME_COLS:
                row[c] = float(row[c])
            for c in ("procs", "width", "height", "radius", "iterations"):
                row[c] = int(row[c])
            runs[(row["filter"], row["radius"], row["iterations"], row["procs"])].append(row)
    out = defaultdict(dict)
    for (flt, rad, its, p), rows in runs.items():
        rows.sort(key=lambda r: r["t_total"])
        out[(flt, rad, its)][p] = rows[len(rows) // 2]
    return out


def describe(key):
    flt, rad, its = key
    parts = [flt]
    if flt != "grayscale":
        parts.append(f"r={rad}")
    if its > 1:
        parts.append(f"{its} iter")
    return ", ".join(parts) if len(parts) == 1 else f"{parts[0]} ({', '.join(parts[1:])})"


def style(ax, title, xlabel, ylabel, procs):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=12)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0)
    if procs is not None:
        ax.set_xscale("log", base=2)
        ax.set_xticks(procs)
        ax.set_xticklabels([str(p) for p in procs])
        ax.minorticks_off()


def new_fig(w=7.5, h=5.4):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def plot_lines(ax, series, ideal):
    """series: list of (label, slot, linestyle, procs, values)."""
    all_p = sorted({p for s in series for p in s[3]})
    if ideal is not None:
        ax.plot(all_p, [ideal(p) for p in all_p], color=INK_2, linewidth=1, linestyle=(0, (4, 3)),
                label="ideal", zorder=1)
    ends = []
    for label, slot, ls, procs, vals in series:
        ax.plot(procs, vals, color=SLOTS[slot], linewidth=2, linestyle=ls, marker=MARKERS[slot],
                markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=3)
        ends.append([procs[-1], vals[-1], vals[-1]])   # x, value, label y
    # Nudge end labels apart (in data units) so equal values stay readable.
    gap = 0.045 * max(max(e[1] for e in ends), 1.0)
    ends.sort(key=lambda e: e[1])
    for lo, hi in zip(ends, ends[1:]):
        if hi[0] == lo[0] and hi[2] - lo[2] < gap:
            hi[2] = lo[2] + gap
    for x, v, y in ends:
        ax.annotate(f"{v:.2f}", (x, y), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    # Below the plot area, where it can never sit on a line.
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left",
              bbox_to_anchor=(0, -0.16), ncol=2)
    ax.set_xlim(all_p[0] / 1.15, all_p[-1] * 1.45)
    return all_p


def strong_series(strong):
    out = []
    for key, by_p in sorted(strong.items(), key=lambda kv: FILTER_SLOT.get(kv[0][0], 9)):
        if 1 not in by_p:
            print(f"warning: no P=1 run for {describe(key)}; skipped", file=sys.stderr)
            continue
        procs = sorted(by_p)
        t1 = by_p[1]["t_total"]
        speed = [t1 / by_p[p]["t_total"] for p in procs]
        out.append((key, procs, speed, [s / p for s, p in zip(speed, procs)]))
    return out


def fmt_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strong", required=True)
    ap.add_argument("--weak")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="", help="machine/run description for plot subtitles")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sub = f"  ·  {a.label}" if a.label else ""
    md = [f"# Scaling results{sub}", ""]

    strong = load(a.strong)
    ss = strong_series(strong)
    size = next(iter(next(iter(strong.values())).values()))
    size_txt = f"{size['width']}x{size['height']}"

    # ---- speedup
    fig, ax = new_fig()
    procs = plot_lines(ax, [(describe(k), FILTER_SLOT.get(k[0], 4), "-", p, s) for k, p, s, _ in ss],
                       ideal=lambda p: p)
    style(ax, f"Strong scaling speedup, {size_txt} image{sub}", "processes (P)", "speedup  T(1) / T(P)", procs)
    ax.set_ylim(0, max(procs) * 1.08)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "speedup.png"), facecolor=SURFACE)
    plt.close(fig)

    # ---- efficiency (strong + weak)
    series = [(f"{describe(k)}, strong", FILTER_SLOT.get(k[0], 4), "-", p, e) for k, p, _, e in ss]
    weak = load(a.weak) if a.weak else {}
    weak_rows = []
    for key, by_p in weak.items():
        if 1 not in by_p:
            continue
        wp = sorted(by_p)
        eff = [by_p[1]["t_total"] / by_p[p]["t_total"] for p in wp]
        series.append((f"{describe(key)}, weak", FILTER_SLOT.get(key[0], 4), (0, (2, 1.5)), wp, eff))
        weak_rows += [[str(p), f"{by_p[p]['width']}x{by_p[p]['height']}", f"{by_p[p]['t_total']:.3f}",
                       f"{by_p[p]['t_compute']:.3f}", f"{by_p[p]['t_halo']:.4f}", f"{e:.2f}"]
                      for p, e in zip(wp, eff)]
    fig, ax = new_fig()
    procs = plot_lines(ax, series, ideal=lambda p: 1.0)
    style(ax, f"Parallel efficiency{sub}", "processes (P)",
          "efficiency  (strong: S/P,  weak: T(1)/T(P))", procs)
    ax.set_ylim(0, 1.15)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "efficiency.png"), facecolor=SURFACE)
    plt.close(fig)

    # ---- breakdown: one panel per strong config (own y-scale: different magnitudes)
    keys = [k for k, *_ in ss]
    fig, axes = plt.subplots(1, len(keys), figsize=(5.2 * len(keys), 4.8), dpi=150, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, key in zip(axes[0], keys):
        by_p = strong[key]
        ps = sorted(by_p)
        xs = range(len(ps))
        bottom = [0.0] * len(ps)
        for i, (label, cols) in enumerate(PHASES):
            vals = [sum(by_p[p][c] for c in cols) for p in ps]
            ax.bar(xs, vals, bottom=bottom, width=0.62, color=SLOTS[i], label=label,
                   edgecolor=SURFACE, linewidth=1.5)
            bottom = [b + v for b, v in zip(bottom, vals)]
        for x, p in zip(xs, ps):
            ax.annotate(f"{by_p[p]['t_total']:.3g}s", (x, bottom[x]), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=8, color=INK_2)
        style(ax, describe(key), "processes (P)", "seconds (slowest rank per phase)", None)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([str(p) for p in ps])
        ax.set_ylim(0, max(bottom) * 1.12)
    handles, labels = axes[0][0].get_legend_handles_labels()
    # Legend on its own row under the title, in stacking order (top segment first).
    fig.suptitle(f"Where the time goes, {size_txt} image{sub}", x=0.01, y=0.98, ha="left",
                 fontsize=12, color=INK)
    fig.legend(handles[::-1], labels[::-1], frameon=False, fontsize=9, labelcolor=INK,
               loc="upper left", bbox_to_anchor=(0.005, 0.925), ncol=len(PHASES))
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    fig.savefig(os.path.join(a.out, "breakdown.png"), facecolor=SURFACE)
    plt.close(fig)

    # ---- tables (the same data as text)
    for key, ps, speed, eff in ss:
        by_p = strong[key]
        md += [f"## Strong scaling: {describe(key)}, {size_txt}", ""]
        md.append(fmt_table(
            ["P", "total s", "read", "scatter", "compute", "halo", "gather", "write", "speedup", "efficiency"],
            [[str(p)] + [f"{by_p[p][c]:.3f}" for c in
                         ("t_total", "t_read", "t_scatter", "t_compute", "t_halo", "t_gather", "t_write")]
             + [f"{s:.2f}", f"{e:.2f}"] for p, s, e in zip(ps, speed, eff)]))
        md.append("")
    if weak_rows:
        md += ["## Weak scaling", ""]
        md.append(fmt_table(["P", "image", "total s", "compute", "halo", "efficiency"], weak_rows))
        md.append("")
    with open(os.path.join(a.out, "summary.md"), "w") as f:
        f.write("\n".join(md))
    print("\n".join(md))
    print(f"\nwrote speedup.png, efficiency.png, breakdown.png, summary.md to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
