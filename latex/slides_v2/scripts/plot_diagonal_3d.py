"""
plot_diagonal_3d.py — 3D illustration of FlatQuant: Kronecker alone vs Kronecker + diag.

Runs a short FlatQuant calibration on Llama-3.2-1B twice:
    1) Kronecker rotation only (add_diag=False)
    2) Kronecker rotation + diagonal scaling (add_diag=True)

Captures the rotated activation entering one MLP block in both setups and
produces a 3-panel 3D figure:

    Activation original | After Kronecker only | After Kronecker + diag

This shows what FlatQuant's diagonal does *on top of* the Kronecker rotation.

Run:
    cd ADC/llama && python ../../latex/slides_v2/scripts/plot_diagonal_3d.py \
        --layer 5 --proj up_proj --epochs 5 --nsamples 32 \
        --out ../../latex/slides_v2/figs/plot_diag_3d.pdf

Note: must be run from a directory where `core/flat_quant.py` is importable,
i.e. inside ADC/llama, OR you can adjust sys.path.append below.
"""

import argparse
import copy
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Make ADC/llama/core importable regardless of cwd.
HERE = Path(__file__).resolve()
ADC_LLAMA = HERE.parents[3] / "ADC" / "llama"
sys.path.insert(0, str(ADC_LLAMA))

from core.flat_quant import (  # noqa: E402
    apply_flatquant_to_model,
    calibrate_flat_quant,
)


# ── Calibration data loader ─────────────────────────────────────────────────

class WikiTextLoader:
    """Yields (input_ids,) tuples — minimal interface that calibrate_flat_quant expects."""

    def __init__(self, tok, n_samples, seq_len=2048):
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n\n".join(s for s in ds["text"] if len(s.strip()) > 100)
        ids = tok(text, return_tensors="pt").input_ids[0]
        self.batches = []
        for i in range(n_samples):
            start = i * seq_len
            end = start + seq_len
            if end > ids.numel():
                break
            self.batches.append((ids[start:end].unsqueeze(0),))

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


# ── Activation capture ──────────────────────────────────────────────────────

def _trans_attr_for(proj_name):
    """Map a projection name to the KroneckerTransform that produces its input."""
    if proj_name in ("up_proj", "gate_proj"):
        return "up_gate_trans"
    if proj_name == "down_proj":
        return "down_trans"
    raise ValueError(proj_name)


def capture_raw_activation(model, tok, layer_idx, proj_name, text, device, n_tokens):
    """Capture INPUT to layers[layer_idx].mlp.<proj_name> on the original
    (pre-FlatQuant) model. proj_name is a plain nn.Linear here, so a regular
    forward hook fires."""
    target = getattr(model.model.layers[layer_idx].mlp, proj_name)

    captured = {}

    def hook(module, inp, out):
        captured["input"] = inp[0].detach().float().cpu()

    handle = target.register_forward_hook(hook)
    try:
        ids = tok(text, return_tensors="pt", truncation=True,
                  max_length=n_tokens).input_ids.to(device)
        with torch.no_grad():
            model(ids)
    finally:
        handle.remove()

    return captured["input"][0].numpy()


def capture_post_kron_activation(model, tok, layer_idx, proj_name, text, device, n_tokens):
    """Capture the OUTPUT of the KroneckerTransform that feeds <proj_name>.

    In FlatQuantLlamaMLP._trans_forward the projections are called via
    ``train_forward(...)`` which bypasses nn.Module.__call__ — so a hook on
    up_proj / down_proj never fires.  Hook the trans module instead: its
    output is precisely the post-Kronecker (and post-diag, if enabled)
    activation we want to plot."""
    trans_attr = _trans_attr_for(proj_name)
    target = getattr(model.model.layers[layer_idx].mlp, trans_attr)

    captured = {}

    def hook(module, inp, out):
        # The same KroneckerTransform is invoked both on activations
        # (3D: batch×seq×hidden) and weights (2D, via train_forward with
        # inv_t=True). Keep only the activation call.
        if "output" in captured:
            return
        if out.dim() != 3:
            return
        captured["output"] = out.detach().float().cpu()

    handle = target.register_forward_hook(hook)
    try:
        ids = tok(text, return_tensors="pt", truncation=True,
                  max_length=n_tokens).input_ids.to(device)
        with torch.no_grad():
            model(ids)
    finally:
        handle.remove()

    if "output" not in captured:
        raise RuntimeError(
            f"Hook on {trans_attr} never saw a 3D activation — did the model "
            f"actually run _trans_forward? Check that calibration completed and "
            f"_ori_mode is False."
        )
    return captured["output"][0].numpy()


# ── Plot ────────────────────────────────────────────────────────────────────

def downsample_channels(act, n_keep):
    """Keep n_keep/2 largest-amax channels + n_keep/2 evenly-spaced ones."""
    in_dim = act.shape[1]
    amax = np.abs(act).max(axis=0)
    n_top = n_keep // 2
    keep_top = np.argsort(amax)[-n_top:]
    grid = np.linspace(0, in_dim - 1, n_keep - n_top).astype(int)
    return np.sort(np.unique(np.concatenate([keep_top, grid])))


def bar3d(ax, data, title, color_inlier="#4a78c0", color_outlier="#c0392b",
          outlier_thresh=None, zmax=None):
    n_rows, n_cols = data.shape
    xs, ys = np.meshgrid(np.arange(n_cols), np.arange(n_rows))
    xs, ys = xs.flatten(), ys.flatten()
    zs = np.zeros_like(xs, dtype=float)
    dx = np.ones_like(xs) * 0.9
    dy = np.ones_like(ys) * 0.9
    dz = np.abs(data).flatten()
    if outlier_thresh is None:
        outlier_thresh = np.inf
    colors = np.where(dz > outlier_thresh, color_outlier, color_inlier)
    ax.bar3d(xs, ys, zs, dx, dy, dz, color=colors, alpha=0.9, shade=True,
             edgecolor="none")
    if zmax is not None:
        ax.set_zlim(0, zmax)
    ax.set_xlabel("channel", labelpad=8)
    ax.set_ylabel("token", labelpad=8)
    ax.set_zlabel("|value|", labelpad=4)
    ax.set_title(title, pad=10)
    ax.view_init(elev=22, azim=-60)
    ax.grid(False)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_alpha(0.15)


# ── Main ────────────────────────────────────────────────────────────────────

def calibrate_with_config(model, dataloader, device, *, add_diag, epochs):
    """Apply FlatQuant wrappers and run calibration with the given add_diag flag.

    Calibration loss includes the full ADC pipeline (tiled INT MVM + floor
    quantisation at fixed delta), matching the paper_comparison FQ_INT4
    setup. The Kronecker (+diag) transforms thus learn to flatten activations
    *for the analog ADC*, not just for plain INT4 fake-quant.
    """
    adc_config = {
        "bx": 4, "bw": 4, "ba": 8, "k": 16,
        "mvm_limit": 256,
        "signed_activations": True,   # symmetric quantisation: codes in [-7, 7]
    }
    apply_flatquant_to_model(
        model,
        w_bits=4, a_bits=4,
        add_diag=add_diag,
        lwc=False, lac=False,
        adc_config=adc_config,
    )
    calibrate_flat_quant(
        model, dataloader, device=device,
        nsamples=len(dataloader), cali_bsz=2,
        epochs=epochs, flat_lr=5e-3,
        diag_alpha=0.5,
        add_diag=add_diag, lwc=False, lac=False,
        propagate_quant_inputs=False,   # plain "trained Kronecker(+diag)" — no propagation
    )
    # calibrate_flat_quant offloads layers to CPU during training to save VRAM;
    # move the whole model back so subsequent forward passes don't fail with
    # device mismatch on embed_tokens.
    model.to(device)
    model.eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--layer", type=int, default=5)
    p.add_argument("--proj", default="up_proj",
                   choices=["gate_proj", "up_proj", "down_proj"])
    p.add_argument("--epochs", type=int, default=5,
                   help="Short calibration: 3-10 epochs is enough for an illustration")
    p.add_argument("--nsamples", type=int, default=16,
                   help="Calibration samples (16 fits on 8GB GPU; up to 64 on 24GB)")
    p.add_argument("--n_tokens", type=int, default=48,
                   help="Tokens shown on the y-axis of each panel")
    p.add_argument("--n_channels", type=int, default=160)
    p.add_argument("--out", default="../figs/plot_diag_3d.pdf")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    print(f"Loading {args.model} on {args.device} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32

    # ── 1. Original model: capture pre-FlatQuant activation ────────────────
    print("[1/3] Capturing original (pre-FlatQuant) activation ...")
    model_orig = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype,
    ).to(args.device).eval()

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    plot_text = "\n\n".join(s for s in ds["text"][:200] if len(s.strip()) > 80)[:3000]

    act_orig = capture_raw_activation(
        model_orig, tok, args.layer, args.proj, plot_text, args.device, args.n_tokens,
    )
    print(f"  shape: {act_orig.shape}")
    del model_orig
    torch.cuda.empty_cache() if args.device.startswith("cuda") else None

    # ── 2. Calibrate Kronecker only (no diag) ──────────────────────────────
    print(f"[2/3] Calibrating Kronecker-only ({args.epochs} ep, {args.nsamples} samples) ...")
    model_kron = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype,
    ).to(args.device).eval()
    dl = WikiTextLoader(tok, n_samples=args.nsamples)
    calibrate_with_config(model_kron, dl, args.device, add_diag=False, epochs=args.epochs)
    act_kron = capture_post_kron_activation(
        model_kron, tok, args.layer, args.proj, plot_text, args.device, args.n_tokens,
    )
    del model_kron
    torch.cuda.empty_cache() if args.device.startswith("cuda") else None

    # ── 3. Calibrate Kronecker + diag ──────────────────────────────────────
    print(f"[3/3] Calibrating Kronecker + diag ({args.epochs} ep, {args.nsamples} samples) ...")
    model_kd = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype,
    ).to(args.device).eval()
    dl = WikiTextLoader(tok, n_samples=args.nsamples)
    calibrate_with_config(model_kd, dl, args.device, add_diag=True, epochs=args.epochs)
    act_kd = capture_post_kron_activation(
        model_kd, tok, args.layer, args.proj, plot_text, args.device, args.n_tokens,
    )
    del model_kd
    torch.cuda.empty_cache() if args.device.startswith("cuda") else None

    # ── Subsample for plotting (use original-activation outliers as the reference channels) ──
    keep = downsample_channels(act_orig, args.n_channels)
    a0 = act_orig[:, keep]
    a1 = act_kron[:, keep]
    a2 = act_kd[:, keep]

    # Common z-axis across panels for fair comparison
    zmax = max(np.abs(a0).max(), np.abs(a1).max(), np.abs(a2).max()) * 1.05
    outlier_thresh = np.quantile(np.abs(a0), 0.99)

    fig = plt.figure(figsize=(15, 4.6))
    axes = [fig.add_subplot(1, 3, i + 1, projection="3d") for i in range(3)]

    bar3d(axes[0], a0, "Original activation\n(strong outliers)",
          outlier_thresh=outlier_thresh, zmax=zmax)
    bar3d(axes[1], a1, "After Kronecker rotation only\n(outliers redistributed)",
          outlier_thresh=outlier_thresh, zmax=zmax)
    bar3d(axes[2], a2, "After Kronecker + diagonal\n(flat across channels)",
          outlier_thresh=outlier_thresh, zmax=zmax)

    fig.suptitle(
        f"FlatQuant diagonal: what it adds on top of the Kronecker rotation  |  "
        f"Llama-3.2-1B layer {args.layer}.mlp.{args.proj} input  |  "
        f"{args.epochs} ep × {args.nsamples} samples",
        y=1.02, fontsize=12,
    )
    plt.tight_layout()

    out = Path(args.out)
    if not out.is_absolute():
        out = Path.cwd() / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=180)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
