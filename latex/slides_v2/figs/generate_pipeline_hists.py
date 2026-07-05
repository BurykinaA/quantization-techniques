"""Generate six mini-distributions for the analog MVM pipeline slide.

All quantised plots use a *real* INT4 staircase (bar per discrete level),
explicit clipping ranges (red dashed lines + pink shade outside), and the
ADC plot divides the analog output by the hardware step

    delta = 2 * M * q_x * q_w / (2**b_a * k),   with  k = 4.

Saves PDFs into the same directory:
  dist_w.pdf      -- weight distribution (Gaussian)
  dist_w_hat.pdf  -- weight + INT4 staircase
  dist_x.pdf      -- activation (heavy-tailed positive)
  dist_x_hat.pdf  -- activation + INT4 staircase
  dist_y.pdf      -- tile output (heavy-tailed near 0)
  dist_y_hat.pdf  -- ADC output, bars at multiples of delta, dead zone
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent

BLUE = "#2E5EAA"
BLUE_FILL = "#6E9CD4"
RED = "#C0392B"
ORANGE = "#E67E22"
GRAY = "#B0B0B0"
DARK = "#1F1F1F"
PINK_SHADE = "#F2C7C2"
DEAD_SHADE = "#FCE4B8"

FIGSIZE = (3.4, 2.1)
DPI = 220

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.edgecolor": GRAY,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.bottom": False,
    "ytick.left": False,
    "xtick.labelbottom": False,
    "ytick.labelleft": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# ── Hardware parameters (INT4, k=4) ──────────────────────────────────────────
M = 256
B_A = 8
K = 4
Q_X = 7
Q_W = 7
DELTA = 2 * M * Q_X * Q_W / (2 ** B_A * K)   # 24.5
CLIP_A = (2 ** (B_A - 1)) * DELTA            # 128 * 24.5  = 3136

# Dequantisation scales:  hat{w} = w / s_w  and  hat{x} = x / s_x
# so y_deq = s_x * s_w * hat{y_int}   (back to FP scale of  w*x)
S_X = 1.0 / Q_X
S_W = 1.0 / Q_W
S_OUT = S_X * S_W                            # ≈ 0.0204
CLIP_FP = CLIP_A * S_OUT                     # ≈ 64
DELTA_FP = DELTA * S_OUT                     # ≈ 0.5


def _style(ax, title=None):
    # All histograms on slides 3 and 4 are shown WITHOUT titles or labels.
    # The pipeline diagram on the slide provides the context.
    ax.margins(x=0, y=0)


def _mark_clip(ax, lo, hi, shade=True):
    """Red dashed verticals at clip boundaries + pink shade in saturated region."""
    ax.axvline(lo, color=RED, linestyle=(0, (4, 2)), linewidth=1.0)
    ax.axvline(hi, color=RED, linestyle=(0, (4, 2)), linewidth=1.0)
    if shade:
        xmin, xmax = ax.get_xlim()
        if xmin < lo:
            ax.axvspan(xmin, lo, color=PINK_SHADE, alpha=0.55, zorder=-2)
        if xmax > hi:
            ax.axvspan(hi, xmax, color=PINK_SHADE, alpha=0.55, zorder=-2)


def _fill_curve(ax, xs, ys, alpha=0.55):
    ax.fill_between(xs, 0, ys, color=BLUE_FILL, alpha=alpha, linewidth=0)
    ax.plot(xs, ys, color=BLUE, linewidth=1.3)


# ═════════════════════════════════════════════════════════════════════════════
# Raw weight (FP, fine-grained histogram, clip at ±1)
# ═════════════════════════════════════════════════════════════════════════════
def plot_w_raw():
    rng = np.random.default_rng(11)
    samples = rng.normal(0, 0.42, 200_000)

    xlim = 1.6
    n_bins = 110
    edges = np.linspace(-xlim, xlim, n_bins + 1)
    counts, _ = np.histogram(samples, bins=edges)
    counts = counts.astype(float)
    counts /= counts.max()
    centres = (edges[:-1] + edges[1:]) / 2
    bw = edges[1] - edges[0]

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(0, 1.15)
    ax.bar(centres, counts, width=bw * 0.95,
           color=BLUE_FILL, edgecolor=BLUE, linewidth=0.25,
           alpha=0.95, align="center")
    _mark_clip(ax, -1.0, 1.0)
    _style(ax)
    fig.savefig(OUT / "dist_w.pdf")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# Quantised weight – INT4 symmetric (15 levels)
# ═════════════════════════════════════════════════════════════════════════════
def plot_w_hat():
    n_levels = 15
    step = 2.0 / (n_levels - 1)
    levels = np.linspace(-1, 1, n_levels)

    rng = np.random.default_rng(42)
    samples = rng.normal(0, 0.42, 60_000)
    clipped = np.clip(samples, -1.0, 1.0)
    idx = np.round((clipped + 1) / step).astype(int)
    idx = np.clip(idx, 0, n_levels - 1)
    counts = np.bincount(idx, minlength=n_levels).astype(float)
    counts /= counts.sum() * step

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.set_xlim(-1.6, 1.6)
    ymax = counts.max() * 1.15
    ax.set_ylim(0, ymax)

    ax.bar(levels, counts, width=step * 0.92, color=BLUE_FILL,
           edgecolor=BLUE, linewidth=0.8, alpha=0.95, align="center")
    _mark_clip(ax, -1.0, 1.0)

    _style(ax)
    fig.savefig(OUT / "dist_w_hat.pdf")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# Raw activation (FP, fine-grained histogram, clip at [0, 1])
# ═════════════════════════════════════════════════════════════════════════════
def plot_x_raw():
    rng = np.random.default_rng(13)
    bulk = rng.exponential(scale=0.23, size=190_000)
    tail = rng.normal(1.15, 0.07, 10_000)
    samples = np.concatenate([bulk, tail])
    samples = samples[(samples >= 0) & (samples <= 1.6)]

    xlim_lo, xlim_hi = -0.08, 1.6
    n_bins = 110
    edges = np.linspace(0.0, xlim_hi, n_bins + 1)
    counts, _ = np.histogram(samples, bins=edges)
    counts = counts.astype(float)
    counts /= counts.max()
    centres = (edges[:-1] + edges[1:]) / 2
    bw = edges[1] - edges[0]

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.set_xlim(xlim_lo, xlim_hi)
    ax.set_ylim(0, 1.15)
    ax.bar(centres, counts, width=bw * 0.95,
           color=BLUE_FILL, edgecolor=BLUE, linewidth=0.25,
           alpha=0.95, align="center")
    _mark_clip(ax, 0.0, 1.0)
    _style(ax)
    fig.savefig(OUT / "dist_x.pdf")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# Quantised activation – INT4 unsigned (16 levels)
# ═════════════════════════════════════════════════════════════════════════════
def plot_x_hat():
    n_levels = 16
    step = 1.0 / (n_levels - 1)
    levels = np.linspace(0, 1, n_levels)

    rng = np.random.default_rng(43)
    bulk = rng.exponential(scale=0.23, size=58_000)
    tail = rng.normal(1.15, 0.07, 2_000)
    samples = np.concatenate([bulk, tail])
    samples = samples[samples >= 0]

    clipped = np.clip(samples, 0.0, 1.0)
    idx = np.round(clipped / step).astype(int)
    idx = np.clip(idx, 0, n_levels - 1)
    counts = np.bincount(idx, minlength=n_levels).astype(float)
    counts /= counts.sum() * step

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.set_xlim(-0.08, 1.6)
    ymax = counts.max() * 1.15
    ax.set_ylim(0, ymax)

    ax.bar(levels, counts, width=step * 0.92, color=BLUE_FILL,
           edgecolor=BLUE, linewidth=0.8, alpha=0.95, align="center")
    _mark_clip(ax, 0.0, 1.0)

    _style(ax)
    fig.savefig(OUT / "dist_x_hat.pdf")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# Tile output y (analog), pre-ADC
# y = sum_i  hat_w_i * hat_x_i   -> integer, bound  |y| <= M * q_x * q_w
# Here (INT4, M=256):  |y| <= 12544
# ═════════════════════════════════════════════════════════════════════════════
Y_MAX = M * Q_X * Q_W   # 12544 for INT4


def _sample_y(seed):
    """Realistic 'bad case': bulk is much narrower than delta/2,
    so most samples land in the ADC dead zone.
    Heavy Cauchy tail still reaches the clipping range to show outliers."""
    rng = np.random.default_rng(seed)
    n = 400_000
    bulk = rng.normal(0.0, 7.5, int(0.96 * n))
    tail = rng.standard_cauchy(int(0.04 * n)) * 70.0
    samples = np.concatenate([bulk, tail])
    samples = samples[np.abs(samples) < 2 * Y_MAX]
    return np.rint(samples).astype(int)


def plot_y():
    samples = _sample_y(101)

    xlim = 5000
    bw = 25
    edges = np.arange(-xlim, xlim + bw, bw)
    heights, _ = np.histogram(samples, bins=edges)
    heights = heights.astype(float)
    heights[heights == 0] = 0.5

    centres = (edges[:-1] + edges[1:]) / 2

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.set_xlim(-xlim, xlim)
    ax.set_yscale("log")
    ax.set_ylim(0.7, heights.max() * 2.2)

    ax.axvspan(-xlim, -CLIP_A, color=PINK_SHADE, alpha=0.55, zorder=-2)
    ax.axvspan(+CLIP_A, +xlim, color=PINK_SHADE, alpha=0.55, zorder=-2)

    ax.bar(centres, heights, width=bw,
           color=BLUE_FILL, edgecolor=None, linewidth=0,
           alpha=0.9, align="center")

    ax.axvline(-CLIP_A, color=RED, linestyle=(0, (4, 2)), linewidth=1.0)
    ax.axvline(+CLIP_A, color=RED, linestyle=(0, (4, 2)), linewidth=1.0)

    _style(ax)
    fig.savefig(OUT / "dist_y.pdf")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# ADC output  ŷ  =  δ · clip( floor(y/δ), -128, 127 )   with k=4
# Full post-ADC range:  |ŷ| ≤ 128·δ = 3136  (≈ 4× narrower than y)
# ═════════════════════════════════════════════════════════════════════════════
def plot_y_hat():
    samples = _sample_y(44).astype(float)

    q_idx = np.floor((samples + DELTA / 2) / DELTA).astype(int)
    q_idx = np.clip(q_idx, -(2 ** (B_A - 1)), 2 ** (B_A - 1) - 1)

    n_vis = 12
    idx_range = np.arange(-n_vis, n_vis + 1)
    counts = np.array([(q_idx == ix).sum() for ix in idx_range], dtype=float)
    centres = idx_range * DELTA

    peak_pct = counts[n_vis] / counts.sum() * 100
    counts_norm = counts / counts.max() * 10.0   # peak = 10

    xlim = (n_vis + 0.7) * DELTA

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.set_xlim(-xlim, xlim)

    ymax = 1.5    # show only the bottom ~15% so small bars are visible
    ax.set_ylim(0, ymax)

    ax.axvspan(-DELTA / 2, DELTA / 2,
               color=DEAD_SHADE, alpha=0.85, zorder=-1)

    ax.bar(centres, counts_norm, width=DELTA * 0.75,
           color=BLUE_FILL, edgecolor=BLUE, linewidth=0.4,
           alpha=0.95, align="center", clip_on=True)

    _style(ax)
    fig.savefig(OUT / "dist_y_hat.pdf")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# Dequantised output  y_deq = s_x * s_w * hat{y_int}
# Same discrete structure as hat{y}, just rescaled to FP units of  w*x.
# Step in FP: delta_fp = s_x * s_w * delta ≈ 0.5
# Clip in FP: ±64
# ═════════════════════════════════════════════════════════════════════════════
def plot_y_deq():
    samples = _sample_y(44).astype(float)

    q_idx = np.floor((samples + DELTA / 2) / DELTA).astype(int)
    q_idx = np.clip(q_idx, -(2 ** (B_A - 1)), 2 ** (B_A - 1) - 1)

    n_vis = 12
    idx_range = np.arange(-n_vis, n_vis + 1)
    counts = np.array([(q_idx == ix).sum() for ix in idx_range], dtype=float)

    centres_fp = idx_range * DELTA_FP
    peak_pct = counts[n_vis] / counts.sum() * 100
    counts_norm = counts / counts.max() * 10.0

    xlim = (n_vis + 0.7) * DELTA_FP

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.set_xlim(-xlim, xlim)
    ymax = 1.5
    ax.set_ylim(0, ymax)

    ax.axvspan(-DELTA_FP / 2, DELTA_FP / 2,
               color=DEAD_SHADE, alpha=0.85, zorder=-1)

    ax.bar(centres_fp, counts_norm, width=DELTA_FP * 0.75,
           color=BLUE_FILL, edgecolor=BLUE, linewidth=0.4,
           alpha=0.95, align="center", clip_on=True)

    _style(ax)
    fig.savefig(OUT / "dist_y_deq.pdf")
    plt.close(fig)


def main():
    plot_w_raw()
    plot_w_hat()
    plot_x_raw()
    plot_x_hat()
    plot_y()
    plot_y_hat()
    print(f"Wrote 6 PDFs to {OUT}   "
          f"(delta = {DELTA},  clip = ±{CLIP_A})")


if __name__ == "__main__":
    main()
