#!/usr/bin/env python3
"""Compare what reaches the ADC across checkpoints.

Input files are produced by ``--adc_zhist_path`` and hold one histogram per tile
over ADC codes floor(z), where z = y_int / Delta is the accumulated MVM partial
sum measured in ADC steps. Delta is fixed by the hardware configuration rather
than by the data, so histograms from different checkpoints share one lattice and
are comparable.

Two failure modes live on that axis: codes -1 and 0 are what the converter reads
when the accumulation never reaches a full step, and mass beyond the clamp
bounds saturates. Because every tile is stored separately, the same files
support both an aggregate view and a per-layer breakdown, which is where a
localized collapse becomes visible.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LAYER_PATTERN = re.compile(r"layers\.(\d+)\.")
COLORS = ["#E07A5F", "#2E86AB", "#5B8C5A", "#8E7CC3"]


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
        "--mode",
        choices=["hist", "per-layer"],
        default="hist",
        help="Single aggregate histogram, or per-layer summary curves",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="Regex over tile names, e.g. 'down_proj' to restrict to one projection",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="With --mode hist, restrict to a single decoder layer",
    )
    parser.add_argument(
        "--xlim",
        type=float,
        default=200.0,
        help="Histogram range in ADC steps (default 200, just past the 8-bit clamp)",
    )
    parser.add_argument(
        "--yscale",
        choices=["log", "linear"],
        default="log",
        help="Log resolves the tails, linear emphasizes the peak at zero",
    )
    parser.add_argument("--title", default=None)
    return parser.parse_args()


def load_tiles(path: Path, pattern: str | None, layer: int | None) -> dict:
    payload = np.load(path, allow_pickle=False)
    names = [str(name) for name in payload["names"]]

    keep = list(range(len(names)))
    if pattern is not None:
        regex = re.compile(pattern)
        keep = [i for i in keep if regex.search(names[i])]
    if layer is not None:
        marker = f"layers.{layer}."
        keep = [i for i in keep if marker in names[i]]
    if not keep:
        raise SystemExit(f"{path}: no tile matches the requested selection")

    code_lo = int(payload["code_lo"])
    n_bins = payload["counts"].shape[1]
    # Bin i holds ADC code code_lo + i, which covers z in [code, code + 1).
    codes = code_lo + np.arange(n_bins)

    metadata = {}
    if "metadata" in payload:
        try:
            metadata = json.loads(str(payload["metadata"]))
        except json.JSONDecodeError:
            metadata = {}

    return {
        "names":     [names[i] for i in keep],
        "counts":    payload["counts"][keep],
        "outside":   payload["under"][keep] + payload["over"][keep],
        "abs_z_sum": payload["abs_z_sum"][keep],
        "na":        float(payload["na"][keep].min()),
        "pa":        float(payload["pa"][keep].max()),
        "codes":     codes,
        "metadata":  metadata,
    }


def summarize(counts: np.ndarray, outside: float, abs_z_sum: float, tiles: dict) -> dict:
    """Reduce a stack of tile histograms to the quantities the figures show."""
    codes = tiles["codes"]
    summed = counts.sum(axis=0) if counts.ndim > 1 else counts
    # Codes outside the stored range still belong in the denominator and are
    # saturated by definition.
    total = max(float(summed.sum()) + float(outside), 1.0)

    # The converter reads zero whenever the accumulation stays below one step,
    # which is exactly codes -1 and 0.
    dead = float(summed[(codes == -1) | (codes == 0)].sum())
    clipped = float(
        summed[(codes < tiles["na"]) | (codes > tiles["pa"])].sum()
    ) + float(outside)

    return {
        "density":    summed / total,
        "dead_frac":  dead / total,
        "clip_frac":  clipped / total,
        "mean_abs_z": float(abs_z_sum) / total,
    }


def group_by_layer(tiles: dict) -> dict:
    groups: dict[int, list[int]] = {}
    for index, name in enumerate(tiles["names"]):
        match = LAYER_PATTERN.search(name)
        if match is not None:
            groups.setdefault(int(match.group(1)), []).append(index)
    if not groups:
        raise SystemExit("No tile name carries a layer index")
    return groups


def format_percent(fraction: float) -> str:
    if 0.0 < fraction < 1e-4:
        return "<0.01%"
    return f"{100 * fraction:.2f}%"


def plot_hist(series: list, args: argparse.Namespace) -> None:
    figure, axis = plt.subplots(figsize=(6.4, 3.6))
    reference = series[0][1]

    axis.axvspan(-1.0, 1.0, color="0.85", zorder=0)
    axis.annotate(
        "floored to 0", xy=(0.0, 1.0), xycoords=("data", "axes fraction"),
        xytext=(0, -10), textcoords="offset points",
        ha="center", va="top", fontsize=8, color="0.35",
    )
    # The clamp sits at +-128 steps and falls outside a zoomed-in view.
    bounds = [
        bound for bound in (reference["tiles"]["na"], reference["tiles"]["pa"])
        if abs(bound) <= args.xlim
    ]
    for bound in bounds:
        axis.axvline(bound, color="0.4", linestyle="--", linewidth=0.9, zorder=1)
    if bounds:
        axis.annotate(
            "ADC clamp", xy=(max(bounds), 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(-4, -10), textcoords="offset points",
            ha="right", va="top", fontsize=8, color="0.35",
        )

    for index, (label, data) in enumerate(series):
        stats = data["all"]
        axis.plot(
            data["tiles"]["codes"] + 0.5, stats["density"],
            color=COLORS[index % len(COLORS)], linewidth=1.4,
            drawstyle="steps-mid",
            label=(
                f"{label}  (floored {format_percent(stats['dead_frac'])}, "
                f"mean $|z|$ {stats['mean_abs_z']:.1f})"
            ),
            zorder=2 + index,
        )

    axis.set_yscale(args.yscale)
    axis.set_xlim(-args.xlim, args.xlim)
    axis.set_xlabel(r"ADC output code $\lfloor y_{\mathrm{int}} / \Delta \rfloor$")
    axis.set_ylabel("fraction of accumulations")
    axis.legend(fontsize=8, frameon=False)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    if args.title:
        axis.set_title(args.title, fontsize=10)
    figure.tight_layout()
    save(figure, args.output)


def plot_per_layer(series: list, args: argparse.Namespace) -> None:
    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(6.4, 5.0), sharex=True,
    )

    for index, (label, data) in enumerate(series):
        layers = sorted(data["per_layer"])
        color = COLORS[index % len(COLORS)]
        top.plot(
            layers, [100 * data["per_layer"][l]["dead_frac"] for l in layers],
            color=color, linewidth=1.4, marker="o", markersize=3, label=label,
        )
        bottom.plot(
            layers, [data["per_layer"][l]["mean_abs_z"] for l in layers],
            color=color, linewidth=1.4, marker="o", markersize=3, label=label,
        )

    top.set_ylabel("floored to zero (%)")
    top.legend(fontsize=8, frameon=False)
    bottom.set_ylabel(r"mean $|z|$  (ADC steps)")
    bottom.set_xlabel("decoder layer")
    for axis in (top, bottom):
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    if args.title:
        top.set_title(args.title, fontsize=10)
    figure.tight_layout()
    save(figure, args.output)


def save(figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    print(f"wrote {output}")


def main() -> None:
    args = parse_args()

    series = []
    for spec in args.input:
        if "=" not in spec:
            raise SystemExit(f"Expected LABEL=PATH, got {spec!r}")
        label, _, raw_path = spec.partition("=")
        tiles = load_tiles(Path(raw_path), args.filter, args.layer)
        entry = {
            "tiles": tiles,
            "all": summarize(
                tiles["counts"], tiles["outside"].sum(),
                tiles["abs_z_sum"].sum(), tiles,
            ),
        }
        if args.mode == "per-layer":
            entry["per_layer"] = {
                layer: summarize(
                    tiles["counts"][rows], tiles["outside"][rows].sum(),
                    tiles["abs_z_sum"][rows].sum(), tiles,
                )
                for layer, rows in group_by_layer(tiles).items()
            }
        series.append((label, entry))

    for label, data in series:
        stats = data["all"]
        print(
            f"{label:<28} tiles={len(data['tiles']['names']):<5} "
            f"dead={stats['dead_frac']:.4f} clipped={stats['clip_frac']:.3e} "
            f"mean|z|={stats['mean_abs_z']:.2f}"
        )

    if args.mode == "per-layer":
        layers = sorted(series[0][1]["per_layer"])
        header = "layer " + "  ".join(f"{label[:16]:>16}" for label, _ in series)
        print(f"\ndead fraction and mean |z| per layer\n{header}")
        for layer in layers:
            cells = []
            for _, data in series:
                stats = data["per_layer"][layer]
                cells.append(f"{100 * stats['dead_frac']:6.2f}% {stats['mean_abs_z']:8.2f}")
            print(f"{layer:>5} " + "  ".join(cells))
        plot_per_layer(series, args)
    else:
        plot_hist(series, args)


if __name__ == "__main__":
    main()
