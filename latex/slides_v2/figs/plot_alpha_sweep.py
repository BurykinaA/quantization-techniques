"""Generate PPL_ADC vs alpha sweep plot for the alpha-mixed objective slide.

Data: Branch llama-flatquant-adc-int4-v2, 1024 samples, Llama-3.2-1B.
Loss = (1-alpha)*MSE(layer(fp_inp), ref) + alpha*MSE(layer(quant_inp), ref)

INT4 only: a clean alpha sweep was never run for INT8, so showing an
"INT8 curve" alongside would require fabricating intermediate points
(an earlier version of this script placed the bin-center result at
alpha=0.5 — that was misleading and has been removed).

Saves: plot_alpha_sweep.pdf
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

OUT = Path(__file__).parent

# ── Presentation color palette ──────────────────────────────────────────────
ORANGE = "#E07A5F"   # myorange   — INT4
GRAY   = "#AAAAAA"   # mygray
GREEN  = "#3D9970"   # mygreen

# ── Data — branch llama-flatquant-adc-int4-v2, 1024 samples ──────────────────
# INT4: full α sweep (propalpha_* rows), no add_diag
alpha_int4 = np.array([0.0,   0.25,  0.50,  0.75,  1.0 ])
ppl_int4   = np.array([56.83, 32.35, 31.22, 35.92, 40.20])

# ── Figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(4.0, 2.8))

ax.plot(alpha_int4, ppl_int4, color=ORANGE, marker="s", markersize=6,
        linewidth=1.8, label=r"INT4 $\mathrm{PPL}_{\mathrm{ADC}}$", zorder=3)

# Highlight α=0.5 column
ax.axvline(0.5, color=GREEN, linewidth=1.0, linestyle=":", alpha=0.8, zorder=1)
ax.axvspan(0.38, 0.62, color=GREEN, alpha=0.07, zorder=0)

# Annotate minimum at α=0.5
ax.annotate("31.22", xy=(0.5, 31.22), xytext=(0.64, 29.5),
            fontsize=7.5, color=ORANGE,
            arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.8))

# ── Axes ─────────────────────────────────────────────────────────────────────
ax.set_xlabel(r"$\alpha$", fontsize=10)
ax.set_ylabel(r"$\mathrm{PPL}_{\mathrm{ADC}}$", fontsize=10)
ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1.0"], fontsize=8)
ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
ax.tick_params(axis="y", labelsize=8)
ax.set_xlim(-0.08, 1.08)
ax.set_ylim(25, 65)

ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", linestyle=":", linewidth=0.6, color=GRAY, alpha=0.6)

ax.legend(fontsize=8, frameon=False, loc="upper right",
          handlelength=1.6, handletextpad=0.5)

fig.tight_layout(pad=0.4)
fig.savefig(OUT / "plot_alpha_sweep.pdf", bbox_inches="tight")
print(f"Saved {OUT / 'plot_alpha_sweep.pdf'}")
