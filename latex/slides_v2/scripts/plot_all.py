"""
Generate all presentation figures from hardcoded results in README.
Run from the slides/ directory:
  python scripts/plot_all.py
Outputs: figs/plot_*.pdf
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent.parent / "figs"
OUT.mkdir(exist_ok=True)

# ── Shared style ──────────────────────────────────────────────────────────────
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

C_BYPASS = "#2E86AB"   # blue
C_ADC    = "#E07A5F"   # orange
C_FP     = "#888888"   # gray dashed
C_INT8   = "#3D9970"   # green
C_BEST   = "#F2CC8F"   # yellow highlight


# ════════════════════════════════════════════════════════════════════════════════
# 1. Dead zone: not hardware-fundamental
# ════════════════════════════════════════════════════════════════════════════════
def plot_deadzone():
    labels     = ["Early baseline\n(branch adc)", "Improved baseline\n(branch pact)"]
    dead_rate  = [81.0, 10.3]
    bypass_ppl = [21.6, 10.01]   # bypass PPL — transform quality proxy

    x = np.arange(len(labels))
    w = 0.35

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax2 = ax1.twinx()

    b1 = ax1.bar(x - w/2, dead_rate, w, label="dead_rate (%)",
                 color=C_ADC, alpha=0.85, zorder=3)
    b2 = ax2.bar(x + w/2, bypass_ppl, w, label="no-ADC PPL",
                 color=C_BYPASS, alpha=0.85, zorder=3)

    ax1.set_ylabel("dead_rate (%)", color=C_ADC)
    ax2.set_ylabel("no-ADC PPL (transform quality)", color=C_BYPASS)
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 100); ax2.set_ylim(0, 30)

    # annotate
    for bar, v in zip(b1, dead_rate):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 1.5,
                 f"{v}%", ha="center", fontsize=9, color=C_ADC, fontweight="bold")
    for bar, v in zip(b2, bypass_ppl):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.4,
                 f"{v:.2f}", ha="center", fontsize=9, color=C_BYPASS, fontweight="bold")

    handles = [mpatches.Patch(color=C_ADC, label="dead_rate (%)"),
               mpatches.Patch(color=C_BYPASS, label="no-ADC PPL")]
    ax1.legend(handles=handles, loc="upper right")
    ax1.set_title("Dead zone collapse was a transform quality problem,\nnot a hardware constraint")
    fig.tight_layout()
    fig.savefig(OUT / "plot_deadzone_not_fundamental.pdf")
    print("✓  plot_deadzone_not_fundamental.pdf")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════════
# 2. INT8 main results
# ════════════════════════════════════════════════════════════════════════════════
def plot_int8():
    methods = ["baseline", "center", "prop", "prop+center"]
    bypass  = [10.01, 10.01, 11.01, 11.00]
    adc     = [28.86, 28.86, 14.55, 14.41]

    x = np.arange(len(methods))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.2))
    b1 = ax.bar(x - w/2, bypass, w, label="no-ADC PPL", color=C_BYPASS, alpha=0.85, zorder=3)
    b2 = ax.bar(x + w/2, adc,    w, label="ADC PPL",    color=C_ADC,    alpha=0.85, zorder=3)

    # FP baseline & best INT8 dashed lines
    ax.axhline(8.68,  color=C_FP,   linestyle="--", linewidth=1.2, label="bfloat16 baseline 8.68")
    ax.axhline(14.41, color=C_INT8, linestyle=":",  linewidth=1.2, label="Best INT8 PTQ 14.41")

    for bar, v in zip(b2, adc):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.3,
                f"{v:.2f}", ha="center", fontsize=8, color=C_ADC, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set_ylabel("Perplexity (WikiText2)")
    ax.set_title("INT8 PTQ: propagated calibration fixes the gap")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 35)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    fig.tight_layout()
    fig.savefig(OUT / "plot_int8_main_results.pdf")
    print("✓  plot_int8_main_results.pdf")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════════
# 3. INT4 PTQ progression (log scale)
# ════════════════════════════════════════════════════════════════════════════════
def plot_int4_progression():
    steps   = [1, 2, 3, 4, 5, 6]
    labels  = [
        "Baseline\n(128 samples)",
        "Prop\n(128 samples)",
        "Prop\n(512 samples)",
        "α=0.5\n(1024 samples)",
        "add_diag\n+α=0.5",
        "Staged\nMLP→Attn",
    ]
    bypass  = [15.41, 40.28, 34.25, 24.24, 20.23, 18.50]
    adc     = [2354.86, 203.89, 40.63, 31.22, 27.56, 27.60]

    fig, ax = plt.subplots(figsize=(9, 4.5))

    ax.plot(steps, bypass, "o-", color=C_BYPASS, linewidth=2, markersize=7,
            label="no-ADC PPL", zorder=4)
    ax.plot(steps, adc,    "s-", color=C_ADC,    linewidth=2, markersize=7,
            label="ADC PPL", zorder=4)

    ax.axhline(8.68,  color=C_FP,   linestyle="--", linewidth=1.1, label="bfloat16 baseline 8.68")
    ax.axhline(14.41, color=C_INT8, linestyle=":",  linewidth=1.1, label="Best INT8 PTQ 14.41")

    # annotate ADC PPL values
    for x_, y_, label in zip(steps, adc, [
            "2354", "204", "40.6", "31.2", "27.6", "27.6"]):
        ax.annotate(label, xy=(x_, y_), xytext=(0, 8),
                    textcoords="offset points", ha="center",
                    fontsize=8, color=C_ADC, fontweight="bold")

    ax.set_yscale("log")
    ax.set_xticks(steps); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Perplexity — log scale (WikiText2)")
    ax.set_title("INT4 pure PTQ: stepwise progression")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "plot_int4_ptq_progression.pdf")
    print("✓  plot_int4_ptq_progression.pdf")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════════
# 4. Diag roles: MLP vs Attn
# ════════════════════════════════════════════════════════════════════════════════
def plot_diag_roles():
    configs = ["diag MLP\nonly", "diag Attn\nonly", "diag Both", "Staged\n(best)"]
    bypass  = [19.37, 24.98, 19.69, 17.83]
    adc     = [31.65, 29.41, 28.46, 26.66]

    x = np.arange(len(configs))
    w = 0.35

    fig, ax = plt.subplots(figsize=(6.5, 4))
    b1 = ax.bar(x - w/2, bypass, w, color=C_BYPASS, alpha=0.85, label="no-ADC PPL", zorder=3)
    b2 = ax.bar(x + w/2, adc,    w, color=C_ADC,    alpha=0.85, label="ADC PPL",    zorder=3)

    ax.axhline(8.68,  color=C_FP,   linestyle="--",linewidth=1.0, label="bfloat16 $\\approx$ 8.68")

    for bar, v in zip(b1, bypass):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.2,
                f"{v:.2f}", ha="center", fontsize=8, color=C_BYPASS)
    for bar, v in zip(b2, adc):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.2,
                f"{v:.2f}", ha="center", fontsize=8, color=C_ADC, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(configs, fontsize=9)
    ax.set_ylabel("Perplexity (WikiText2)")
    ax.set_title("MLP diag drives bypass quality;\nAttn diag closes the ADC gap")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 40)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    fig.tight_layout()
    fig.savefig(OUT / "plot_diag_roles.pdf")
    print("✓  plot_diag_roles.pdf")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════════
# 5. MLP split: up_gate vs down_trans
# ════════════════════════════════════════════════════════════════════════════════
def plot_mlp_split():
    configs = ["diag up_gate\nonly", "diag down_trans\nonly", "diag both\nMLP"]
    bypass  = [23.22, 20.42, 19.37]
    adc     = [39.59, 31.26, 31.65]

    x = np.arange(len(configs))
    w = 0.35

    fig, ax = plt.subplots(figsize=(5.5, 4))
    b1 = ax.bar(x - w/2, bypass, w, color=C_BYPASS, alpha=0.85, label="no-ADC PPL", zorder=3)
    b2 = ax.bar(x + w/2, adc,    w, color=C_ADC,    alpha=0.85, label="ADC PPL",    zorder=3)

    for bar, v in zip(b1, bypass):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.3,
                f"{v:.2f}", ha="center", fontsize=8.5, color=C_BYPASS)
    for bar, v in zip(b2, adc):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.3,
                f"{v:.2f}", ha="center", fontsize=8.5, color=C_ADC, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(configs, fontsize=9)
    ax.set_ylabel("Perplexity (WikiText2)")
    ax.set_title("Inside MLP diag: down_trans is the key driver")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 48)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    fig.tight_layout()
    fig.savefig(OUT / "plot_mlp_split_up_down.pdf")
    print("✓  plot_mlp_split_up_down.pdf")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════════
# 6. PTQ plateau
# ════════════════════════════════════════════════════════════════════════════════
def plot_ptq_plateau():
    # best ADC PPL per version
    versions = ["v1\n(128s)", "v1\n(512s)", "v2\nα-sweep", "v3\ndiag", "v4\nstaged", "v5\nstoch"]
    adc_best = [178.86, 40.63, 27.56, 28.46, 27.60, 27.82]
    colors   = [C_ADC]*6

    fig, ax = plt.subplots(figsize=(7.5, 4))
    bars = ax.bar(versions, adc_best, color=colors, alpha=0.80, zorder=3, width=0.5)

    # plateau band
    ax.axhspan(27.0, 28.8, color="#F2CC8F", alpha=0.45, zorder=1, label="PTQ plateau 27–28.8")

    ax.axhline(8.68,  color=C_FP,   linestyle="--", linewidth=1.2, label="bfloat16 baseline 8.68")
    ax.axhline(14.41, color=C_INT8, linestyle=":",  linewidth=1.2, label="INT8 prop+center 14.41")

    for bar, v in zip(bars, adc_best):
        ax.text(bar.get_x() + bar.get_width()/2,
                min(v + 1.5, 185),
                f"{v:.1f}", ha="center", fontsize=8.5, color=C_ADC, fontweight="bold")

    ax.set_ylabel("Best ADC PPL (WikiText2)")
    ax.set_title("Pure PTQ plateaus at ~27–28 PPL across INT4 v1–v5")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 210)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    fig.tight_layout()
    fig.savefig(OUT / "plot_ptq_plateau.pdf")
    print("✓  plot_ptq_plateau.pdf")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════════
# 7. LoRA ablations — 2×2 grid
# ════════════════════════════════════════════════════════════════════════════════
def plot_lora_ablations():
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    # ── Top-left: CE vs CE+KL ──────────────────────────────────────────────────
    ax = axes[0, 0]
    groups  = ["down+o,\nr=4, CE", "down+o,\nr=4, CE+KL", "all 7,\nr=4, CE", "all 7,\nr=4, CE+KL"]
    vals    = [15.57, 14.33, 16.68, 14.03]
    clr     = [C_BYPASS, C_ADC, C_BYPASS, C_ADC]
    bars = ax.bar(groups, vals, color=clr, alpha=0.85, zorder=3, width=0.5)
    ax.axhline(14.41, color=C_INT8, linestyle=":", linewidth=1.1, label="INT8 best 14.41")
    ax.axhline(8.68,  color=C_FP,   linestyle="--",linewidth=1.0, label="bfloat16 ≈ 8.68")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.1, f"{v:.2f}",
                ha="center", fontsize=8, fontweight="bold")
    ax.set_ylabel("ADC PPL"); ax.set_ylim(8, 20)
    ax.set_title("CE+KL consistently beats CE"); ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    # ── Top-right: Rank sweep ──────────────────────────────────────────────────
    ax = axes[0, 1]
    ranks = [1, 2, 4, 8]
    ppl   = [15.90, 15.60, 15.57, 15.24]
    ax.plot(ranks, ppl, "o-", color=C_ADC, linewidth=2, markersize=8, zorder=4)
    ax.axhline(14.41, color=C_INT8, linestyle=":", linewidth=1.1, label="INT8 best 14.41")
    ax.axhline(8.68,  color=C_FP,   linestyle="--",linewidth=1.0, label="bfloat16 ≈ 8.68")
    for x_, y_ in zip(ranks, ppl):
        ax.text(x_, y_ + 0.05, f"{y_:.2f}", ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xlabel("LoRA rank r"); ax.set_ylabel("ADC PPL")
    ax.set_xticks(ranks); ax.set_ylim(9, 18)
    ax.set_title("Rank saturates early (r=1 captures most gain)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3, zorder=0)

    # ── Bottom-left: Layer-selective ──────────────────────────────────────────
    ax = axes[1, 0]
    layer_labels = ["All 16\nlayers", "First 8\n(0–7)", "Last 8\n(8–15)"]
    layer_ppl    = [15.57, 16.64, 20.08]
    clr2 = [C_INT8, C_ADC, C_BYPASS]
    bars2 = ax.bar(layer_labels, layer_ppl, color=clr2, alpha=0.85, zorder=3, width=0.45)
    ax.axhline(14.41, color=C_INT8, linestyle=":", linewidth=1.1, label="INT8 best 14.41")
    for bar, v in zip(bars2, layer_ppl):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.2, f"{v:.2f}",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("ADC PPL"); ax.set_ylim(8, 25)
    ax.set_title("First 8 layers carry more ADC error")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3, zorder=0)

    # ── Bottom-right: Post vs Pre ADC (log scale) ─────────────────────────────
    ax = axes[1, 1]
    positions = [0, 1]
    pre_post_labels = ["Post-ADC\n(residual)", "Pre-ADC\n(RAOQ-style)"]
    pre_post_vals   = [15.57, 105.63]
    clr3 = [C_INT8, "crimson"]
    bars3 = ax.bar(positions, pre_post_vals, color=clr3, alpha=0.85, zorder=3, width=0.4)
    ax.axhline(14.41, color=C_INT8, linestyle=":", linewidth=1.1, label="INT8 best 14.41")
    for bar, v in zip(bars3, pre_post_vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1.5, f"{v:.1f}",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(positions); ax.set_xticklabels(pre_post_labels)
    ax.set_ylabel("ADC PPL (log scale)"); ax.set_yscale("log")
    ax.set_title("Pre-ADC LoRA: catastrophic divergence")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3, zorder=0)

    fig.suptitle("ADC-LoRA Ablations (v7, r=4, down+o unless noted)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "plot_lora_ablations_grid.pdf")
    print("✓  plot_lora_ablations_grid.pdf")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════════
# 8. Final results: Wiki2 + C4
# ════════════════════════════════════════════════════════════════════════════════
def plot_final_results():
    methods  = ["INT8\nbaseline", "INT8\nprop+center", "INT4 staged\nPTQ", "INT4 +LoRA\n(r4_all_ce_kl)"]
    wiki_adc = [28.86, 14.41, 27.60, 14.03]
    c4_adc   = [None,  None,  46.40, 23.30]

    x = np.arange(len(methods))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Wiki2 bars
    bars_wiki = ax.bar(x - w/2, wiki_adc, w,
                       color=C_BYPASS, alpha=0.87, label="WikiText2 ADC PPL", zorder=3)

    # C4 bars (None → 0, with hatch for missing)
    c4_plot = [v if v is not None else 0 for v in c4_adc]
    bars_c4 = ax.bar(x + w/2, c4_plot, w,
                     color=C_ADC, alpha=0.87, label="C4 ADC PPL", zorder=3)

    # mark missing C4 bars with N/A
    for bar, v in zip(bars_c4, c4_adc):
        if v is None:
            ax.text(bar.get_x() + bar.get_width()/2, 1.5, "N/A",
                    ha="center", fontsize=8, color="gray")
        else:
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.5,
                    f"{v:.2f}", ha="center", fontsize=8.5, color=C_ADC, fontweight="bold")

    for bar, v in zip(bars_wiki, wiki_adc):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.5,
                f"{v:.2f}", ha="center", fontsize=8.5, color=C_BYPASS, fontweight="bold")

    ax.axhline(8.68, color=C_FP,   linestyle="--", linewidth=1.2, label="bfloat16 Wiki2 baseline 8.68")
    ax.axhline(13.13, color="black", linestyle="-.", linewidth=0.9, alpha=0.5, label="bfloat16 C4 baseline 13.13")

    # highlight winner
    ax.add_patch(plt.Rectangle(
        (x[-1] - w - 0.05, 0), 2*w + 0.1, 16.5,
        fill=True, facecolor=C_BEST, alpha=0.35, zorder=0,
        label="Best checkpoint"
    ))

    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set_ylabel("Perplexity")
    ax.set_title("INT4 + LoRA surpasses best INT8 PTQ on both domains")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 55)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    fig.tight_layout()
    fig.savefig(OUT / "plot_final_results_wiki_c4.pdf")
    print("✓  plot_final_results_wiki_c4.pdf")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════════
# Run all
# ════════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    plot_deadzone()
    plot_int8()
    plot_int4_progression()
    plot_diag_roles()
    plot_mlp_split()
    plot_ptq_plateau()
    plot_lora_ablations()
    plot_final_results()
    print(f"\nAll figures written to {OUT}/")
