"""
Synthetic illustration of FlatQuant activation rotation.

Shows what happens to activation codes before and after the orthogonal transform:
  Before: one outlier feature at code 127, the rest cluster near 0
  After:  uniform spread across codes 50–110 (no dominant feature)

Run from the slides/ directory:
  python scripts/plot_flatquant_intuition.py
Output: figs/fig_flatquant_intuition.pdf
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent.parent / "figs"
OUT.mkdir(exist_ok=True)

# ── Shared style (matches plot_all.py) ───────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  9,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

C_BYPASS = "#2E86AB"
C_ADC    = "#E07A5F"

# ── Synthetic data ────────────────────────────────────────────────────────────
np.random.seed(42)
N = 256   # tile size (mvm_limit=256 features per tile)

# Before rotation: one outlier dominates, rest are small
activations_before = np.abs(np.random.randn(N)) * 0.3 + 0.5   # small values ~0.5
activations_before[0] = 12.7                                    # outlier: 25× larger

# Per-token quantization scale = max(|x|) / 127
scale_before = activations_before.max() / 127.0
codes_before = np.clip(np.round(activations_before / scale_before), 0, 127).astype(int)
# → feature 0 gets code 127; most others get codes 0–5

# After rotation: orthogonal transform redistributes energy evenly
activations_after = np.random.uniform(4.5, 7.5, N)             # flat distribution
scale_after = activations_after.max() / 127.0
codes_after = np.clip(np.round(activations_after / scale_after), 0, 127).astype(int)
# → codes spread uniformly ~76–127; no dead features

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5), sharey=False)

bins = np.arange(0, 130, 4)

ax1.hist(codes_before, bins=bins, color=C_ADC, alpha=0.85, edgecolor="white", linewidth=0.4)
ax1.set_title("Before FlatQuant\n(outlier in feature 0)")
ax1.set_xlabel("INT8 activation code")
ax1.set_ylabel("Number of features")
ax1.set_xlim(-2, 130)
ax1.axvline(127, color="#888", linestyle="--", linewidth=0.8, label="max code")
# annotate dead zone
ax1.axvspan(0, 15, alpha=0.12, color=C_ADC, label="near-zero codes")
ax1.text(2, ax1.get_ylim()[1] * 0.7, "most features\nnear 0", fontsize=8,
         color=C_ADC, va="top")
ax1.text(115, ax1.get_ylim()[1] * 0.85, "127", fontsize=8, color="#888", ha="center")

ax2.hist(codes_after, bins=bins, color=C_BYPASS, alpha=0.85, edgecolor="white", linewidth=0.4)
ax2.set_title("After FlatQuant\n(uniform distribution)")
ax2.set_xlabel("INT8 activation code")
ax2.set_xlim(-2, 130)
ax2.text(40, ax2.get_ylim()[1] * 0.7, "codes spread\nuniformly\n→ no dead zone",
         fontsize=8, color=C_BYPASS, va="top")

fig.suptitle("FlatQuant: orthogonal rotation flattens activation distribution", fontsize=11)
plt.tight_layout()

out_path = OUT / "fig_flatquant_intuition.pdf"
plt.savefig(out_path, bbox_inches="tight")
print(f"Saved: {out_path}")
