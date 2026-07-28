#!/usr/bin/env python3
"""Plot the distribution of ADC inputs for two checkpoints on one axis.

Input files are produced by ``--adc_zhist_path`` and hold per-tile histograms of
z = y_int / Delta, the accumulated MVM partial sum measured in ADC steps. Delta
is fixed by the hardware configuration rather than by the data, so histograms
from different checkpoints share one lattice and are directly comparable.

Two failure modes are visible on the same axis: mass inside |z| < 1 is floored
to zero by the converter, and mass beyond the clamp bounds is saturated.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Histogram file with a legend label, repeat for each checkpoint",
    )
    parser.add_argument("--output", type=Path, default=Path("adc_input_hist.pdf"))
    parser.add_argument(
        "--filter",
        default=None,
        help="Regex over tile names, e.g. 'layers\\.11\\..*down_proj' for one projection",
    )
    parser.add_argument(
        "--xlim",
        type=float,
        default=200.0,
        help="Plot range in ADC steps (default 200, just past the 8-bit clamp)",
    )
    parser.add_argument("--title", default=None)
    return parser.parse_args()


def load_histogram(path: Path, pattern: str | None) -> dict:
    payload = np.load(path, allow_pickle=False)
    names = [str(name) for name in payload["names"]]

    selected = list(range(len(names)))
    if pattern is not None:
        regex = re.compile(pattern)
        selected = [i for i, name in enumerate(names) if regex.search(name)]
        if not selected:
            raise SystemExit(f"{path}: no tile matches {pattern!r}")

    counts = payload["counts"][selected].sum(axis=0)
    z_range = float(payload["z_range"])
    n_bins = int(payload["n_bins"])
    edges = np.linspace(-z_range, z_range, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Values outside the histogram range were counted separately, so they still
    # belong in the denominator and in the saturated fraction.
    outside = float(payload["under"][selected].sum() + payload["over"][selected].sum())
    total = counts.sum() + outside

    na = float(payload["na"][selected].min())
    pa = float(payload["pa"][selected].max())
    dead = float(counts[np.abs(centers) < 1.0].sum())
    clipped = float(counts[(centers < na) | (centers > pa)].sum()) + outside

    metadata = {}
    if "metadata" in payload:
        try:
            metadata = json.loads(str(payload["metadata"]))
        except json.JSONDecodeError:
            metadata = {}

    return {
        "centers": centers,
        "density": counts / max(total, 1.0),
        "dead_frac": dead / max(total, 1.0),
        "clip_frac": clipped / max(total, 1.0),
        "na": na,
        "pa": pa,
        "n_tiles": len(selected),
        "metadata": metadata,
    }


def main() -> None:
    args = parse_args()

    series = []
    for spec in args.input:
        if "=" not in spec:
            raise SystemExit(f"Expected LABEL=PATH, got {spec!r}")
        label, _, raw_path = spec.partition("=")
        series.append((label, load_histogram(Path(raw_path), args.filter)))

    figure, axis = plt.subplots(figsize=(6.4, 3.6))
    colors = ["#E07A5F", "#2E86AB", "#5B8C5A", "#8E7CC3"]

    reference = series[0][1]
    axis.axvspan(-1.0, 1.0, color="0.85", zorder=0)
    axis.annotate(
        "floored to 0", xy=(0.0, 1.0), xycoords=("data", "axes fraction"),
        xytext=(0, -10), textcoords="offset points",
        ha="center", va="top", fontsize=8, color="0.35",
    )
    for bound in (reference["na"], reference["pa"]):
        axis.axvline(bound, color="0.4", linestyle="--", linewidth=0.9, zorder=1)
    axis.annotate(
        "ADC clamp", xy=(reference["pa"], 1.0), xycoords=("data", "axes fraction"),
        xytext=(-4, -10), textcoords="offset points",
        ha="right", va="top", fontsize=8, color="0.35",
    )

    for index, (label, data) in enumerate(series):
        axis.plot(
            data["centers"], data["density"],
            color=colors[index % len(colors)], linewidth=1.4,
            label=(
                f"{label}  (dead {100 * data['dead_frac']:.1f}%, "
                f"clipped {100 * data['clip_frac']:.1f}%)"
            ),
            zorder=2 + index,
        )
        print(
            f"{label:<28} tiles={data['n_tiles']:<5} "
            f"dead={data['dead_frac']:.4f} clipped={data['clip_frac']:.4f}"
        )

    axis.set_yscale("log")
    axis.set_xlim(-args.xlim, args.xlim)
    axis.set_xlabel(r"ADC input $z = y_{\mathrm{int}} / \Delta$  (ADC steps)")
    axis.set_ylabel("fraction of accumulations")
    if args.title:
        axis.set_title(args.title, fontsize=10)
    axis.legend(fontsize=8, frameon=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
