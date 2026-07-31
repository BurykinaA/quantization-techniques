#!/usr/bin/env python3
"""
ADC-aware INT4 quantization of timm Vision Transformers on ImageNet.

Ports the LLaMA ADC pipeline to vision transformers, reusing the core modules
in ADC/llama/core (adc_layers.py, adc_lora.py, flat_quant.py) unchanged, plus
the ViT-specific wrappers in this directory.

Pipeline (quantized configs):
  1. Load pretrained timm ViT
  2. FlatQuant calibration  → learn Kronecker transforms that flatten
                              activations before INT4 quantization  (staged)
  3. ADC conversion         → replace nn.Linear with TiledLinearADC
  4. ADC calibration        → set per-channel quant scales (percentile)
  5. (optional) post-ADC LoRA → learn residual correction (CE+KL to FP teacher)
  6. Evaluate ImageNet top-1/top-5, measure latency

Four configurations:
  fp            full precision
  int4_no_adc   INT4 FlatQuant, ADC floor bypassed
  ptq           INT4 FlatQuant + ADC hardware model (default k=4)
  ptq_lora      ptq + post-ADC LoRA (rank 4)

Run (via run.sh, which sets the env):
  ./run.sh --model vit_tiny_patch16_224 --configs fp int4_no_adc --smoke --val_portion 0.05
  ./run.sh --model vit_tiny_patch16_224 --configs ptq ptq_lora --k 4
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import timm
import torch
import torch.nn as nn

# Make the ADC.llama.core package (absolute imports, like the llama runs use)
# and the sibling vit_* modules importable regardless of the launch cwd.
#   _HERE      = ADC/llama/vit        → sibling `vit_*` modules
#   _REPO_ROOT = quantization_techniques → `ADC.llama.core.*` namespace packages
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _HERE)

from ADC.llama.core.adc_layers import TiledLinearADC    # noqa: E402
from ADC.llama.core.adc_lora import apply_adc_lora      # noqa: E402
from ADC.llama.core.flat_quant import FlatQuantLinear   # noqa: E402

from vit_configs import build_config, ViTFPConfig, ViTInt4NoADCConfig, \
    ViTPTQConfig, ViTPTQLoRAConfig                       # noqa: E402
from vit_data import ViTImageNetLoaderGenerator, resolve_imagenet_root  # noqa: E402
from vit_eval import validate, measure_latency          # noqa: E402
from vit_flat_quant import (                             # noqa: E402
    apply_flatquant_to_vit, reparameterize_vit, calibrate_flat_quant_vit,
)
from vit_lora import calibrate_adc_lora_vit              # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_vit_model(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading {cfg.model_name} (pretrained) ...")
    model = timm.create_model(cfg.model_name, pretrained=True).to(device).eval()
    n = sum(p.numel() for p in model.parameters())
    logger.info(f"  {n:,} parameters, device={device}")
    return model, device


# ─────────────────────────────────────────────────────────────────────────────
# ADC conversion + calibration  (reused pattern from the LLaMA ADC pipeline)
# ─────────────────────────────────────────────────────────────────────────────

EXCLUDE = ("head", "_orig_attn")  # classifier head + the retained orig attn module


def _replace_linear_with_adc(model, cfg):
    """Replace every nn.Linear (except the classifier head) with TiledLinearADC."""
    def _replace(module, prefix=""):
        for name, child in module.named_children():
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and not any(e in full for e in EXCLUDE):
                adc = TiledLinearADC(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=(child.bias is not None),
                    bx=cfg.bx, bw=cfg.bw, ba=cfg.ba, k=cfg.k,
                    ashift=False, signed_activations=True,
                    mvm_limit=cfg.mvm_limit,
                    use_kurtosis_loss=False,
                )
                adc.load_weights(child)
                setattr(module, name, adc)
            else:
                _replace(child, full)

    _replace(model)
    n_adc = sum(1 for _, m in model.named_modules() if isinstance(m, TiledLinearADC))
    delta0 = next(m.tiles[0].delta for _, m in model.named_modules()
                  if isinstance(m, TiledLinearADC))
    logger.info(f"  Replaced {n_adc} linear layers with TiledLinearADC "
                f"(k={cfg.k}, example δ≈{delta0:.3f})")
    return model


def _set_bypass_all(model, bypass):
    """Toggle FULL bypass (plain F.linear, NO quantization) on every ADC layer.

    Used only for ADC calibration, where we want clean FP activations flowing
    into each tile so the percentile scales reflect the true activation range.
    """
    for _, m in model.named_modules():
        if isinstance(m, TiledLinearADC):
            m.set_bypass_adc(bypass)
            m.set_bypass_all(bypass)


def _set_bypass_adc_floor(model, bypass):
    """Toggle ONLY the ADC floor (floor(y/δ)) on every ADC layer.

    With bypass=True the model still applies INT4 weight+activation
    quantization but skips the ADC accumulator quantization — this is exactly
    what the ``int4_no_adc`` config measures.  ``bypass_all`` is forced False so
    the quantization path is not short-circuited.
    """
    for _, m in model.named_modules():
        if isinstance(m, TiledLinearADC):
            m.set_bypass_all(False)
            m.set_bypass_adc(bypass)


class _ADCCalibrator:
    """Sets per-channel quant scales from activation/weight percentile stats.

    Runs in bypass mode (clean FP forward) to collect stats, then writes scales.
    (Same approach as the LLaMA ADC pipeline's _ADCCalibrator.)
    """

    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.stats = {}

    def _hook(self, name):
        def hook(module, inp, _out):
            s = self.stats.setdefault(name, {"act": [], "module": module})
            s["act"].append(inp[0].detach().float().abs().max().item())
        return hook

    @torch.no_grad()
    def run(self, loader, device):
        hooks = []
        for name, m in self.model.named_modules():
            if isinstance(m, TiledLinearADC):
                for i, tile in enumerate(m.tiles):
                    hooks.append(tile.register_forward_hook(self._hook(f"{name}.tiles.{i}")))
        self.model.eval()
        for bi, (imgs, _) in enumerate(loader):
            if bi >= self.cfg.num_calibration_batches:
                break
            self.model(imgs.to(device))
        for h in hooks:
            h.remove()
        logger.info(f"  Collected ADC stats for {len(self.stats)} tiles")

    def apply(self):
        q_x = 2 ** (self.cfg.bx - 1) - 1
        q_w = 2 ** (self.cfg.bw - 1) - 1
        updated = 0
        for name, m in self.model.named_modules():
            if not isinstance(m, TiledLinearADC):
                continue
            for i, tile in enumerate(m.tiles):
                key = f"{name}.tiles.{i}"
                if key not in self.stats:
                    continue
                act = np.array(self.stats[key]["act"])
                if self.cfg.calibration_method == "percentile":
                    act_scale = np.percentile(act, 99.9) / q_x
                else:
                    act_scale = act.max() / q_x
                act_scale = max(float(act_scale), 1e-8)
                if hasattr(tile, "activation_quantizer"):
                    tile.activation_quantizer.scale.data.fill_(act_scale)
                aq = tile.weight_quantizer
                w = tile.weight.detach().float()
                per_ch = w.abs().max(dim=1)[0].clamp(min=1e-6) / q_w
                aq.scale.data = per_ch.to(aq.scale.device)
                updated += 1
        logger.info(f"  Applied calibration scales to {updated} tiles")


# ─────────────────────────────────────────────────────────────────────────────
# FlatQuant + ADC  (cache-aware)
# ─────────────────────────────────────────────────────────────────────────────

def _fq_cache_key(cfg):
    import hashlib
    # int4_no_adc and ptq now train an identical FlatQuant+ADC model (they differ
    # only by the eval-time ADC-floor bypass), so use_adc is no longer part of the
    # key — both configs share one cached training run.  v5 invalidates the old
    # v4 int4_no_adc cache, which was trained with the now-removed mismatch.
    s = (f"{cfg.model_name}_{cfg.fq_epochs}_{cfg.fq_stage_b_epochs}_{cfg.fq_nsamples}"
         f"_{cfg.fq_lr}_{cfg.bx}_{cfg.bw}_{cfg.ba}_{cfg.k}_{cfg.mvm_limit}"
         f"_v5")
    return hashlib.md5(s.encode()).hexdigest()[:10]

def apply_flatquant(model, loader, cfg, device, cache_dir=None):
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"fq_adc_vit_{_fq_cache_key(cfg)}.pt")
        if os.path.exists(cache_path):
            logger.info(f"Loading cached FlatQuant+ADC ViT: {cache_path}")
            return torch.load(cache_path, weights_only=False, map_location=device).to(device)
        logger.info(f"FlatQuant cache miss — will save to {cache_path}")

    # Always train FlatQuant through the ADC path (_train_forward_adc), whose
    # activation quantizer (per-token amax) matches QATLinearADC at eval.
    # int4_no_adc then bypasses ONLY the ADC floor at eval (see
    # _set_bypass_adc_floor in run_config), so the sole train/eval difference is
    # the floor itself — no activation-quantizer mismatch, mirroring the LLaMA
    # FlatQuant pipeline (train with adc_config, toggle bypass_adc at eval).
    fq_adc_config = dict(bx=cfg.bx, bw=cfg.bw, ba=cfg.ba, k=cfg.k,
                         mvm_limit=cfg.mvm_limit, signed_activations=True)

    logger.info("─── Apply FlatQuant wrappers ───")
    model = apply_flatquant_to_vit(
        model, w_bits=cfg.fq_w_bits, a_bits=cfg.fq_a_bits,
        add_diag=cfg.fq_add_diag, lwc=cfg.fq_lwc, lac=cfg.fq_lac,
        adc_config=fq_adc_config)

    logger.info(f"─── FlatQuant Stage A ({cfg.fq_epochs} ep, lr={cfg.fq_lr}, "
                f"nsamples={cfg.fq_nsamples}) ───")
    model = calibrate_flat_quant_vit(
        model, loader, device, nsamples=cfg.fq_nsamples, cali_bsz=cfg.fq_cali_bsz,
        epochs=cfg.fq_epochs, flat_lr=cfg.fq_lr,
        add_diag=cfg.fq_add_diag, lwc=cfg.fq_lwc, lac=cfg.fq_lac,
        propagate_quant_inputs=False, diag_attn=False, diag_mlp=True)

    if cfg.fq_stage_b_epochs > 0:
        logger.info(f"─── FlatQuant Stage B ({cfg.fq_stage_b_epochs} ep, "
                    f"lr={cfg.fq_lr * 0.1}, prop α={cfg.fq_stage_b_prop_alpha}) ───")
        model = calibrate_flat_quant_vit(
            model, loader, device, nsamples=cfg.fq_nsamples, cali_bsz=cfg.fq_cali_bsz,
            epochs=cfg.fq_stage_b_epochs, flat_lr=cfg.fq_lr * 0.1,
            add_diag=cfg.fq_add_diag, lwc=cfg.fq_lwc, lac=cfg.fq_lac,
            propagate_quant_inputs=True, propagate_quant_alpha=cfg.fq_stage_b_prop_alpha,
            diag_attn=cfg.fq_stage_b_diag_attn, diag_mlp=True)

    logger.info("─── Reparameterize transforms into weights ───")
    model = model.to(device)
    model = reparameterize_vit(model)

    logger.info("─── Convert to TiledLinearADC ───")
    model = _replace_linear_with_adc(model, cfg).to(device)

    logger.info("─── ADC calibration (percentile scales) ───")
    _set_bypass_all(model, True)   # clean FP forward to collect activation stats
    calibrator = _ADCCalibrator(model, cfg)
    calibrator.run(loader, device)
    _set_bypass_all(model, False)
    calibrator.apply()

    for _, m in model.named_modules():
        if hasattr(m, "set_quantizer_mode"):
            m.set_quantizer_mode("fixed")

    if cache_path:
        logger.info(f"Saving FlatQuant+ADC ViT to cache: {cache_path}")
        torch.save(model, cache_path)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# LoRA
# ─────────────────────────────────────────────────────────────────────────────

def apply_lora(model, loader_gen, cfg, device):
    logger.info(f"─── Post-ADC LoRA (rank={cfg.lora_rank}, α={cfg.lora_alpha}, "
                f"targets={list(cfg.lora_target_modules)}) ───")
    model = apply_adc_lora(
        model, target_modules=list(cfg.lora_target_modules),
        rank=cfg.lora_rank, lora_alpha=cfg.lora_alpha,
        mode=cfg.lora_mode, layer_indices=None)
    # Dedicated LoRA calibration loader batched at cfg.lora_cali_bsz.  LoRA runs a
    # full forward+backward through the ADC graph, so it needs a much smaller
    # batch than FlatQuant's fq_cali_bsz (reusing that loader is what caused OOM).
    lora_loader = loader_gen.calib_loader(
        num=cfg.lora_nsamples, batch_size=cfg.lora_cali_bsz)
    model = calibrate_adc_lora_vit(
        model, lora_loader, device,
        nsamples=cfg.lora_nsamples,
        epochs=cfg.lora_epochs, lora_lr=cfg.lora_lr,
        lora_loss=cfg.lora_loss,
        teacher_model_name=cfg.model_name if cfg.lora_loss == "ce_kl" else None,
        kl_weight=cfg.lora_kl_weight, kl_temperature=cfg.lora_kl_temperature)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(model, val_loader, cfg, device):
    _, top1, top5 = validate(val_loader, model, device=device, desc=cfg.name)
    lat = measure_latency(model, device)
    logger.info(f"  Latency mean={lat['latency_mean_ms']:.1f}ms  "
                f"throughput={lat['throughput_img_s']:.0f} img/s")
    return {"top1": top1, "top5": top5,
            "latency_mean_ms": lat["latency_mean_ms"],
            "throughput_img_s": lat["throughput_img_s"]}


def run_config(cfg, cache_dir=None):
    logger.info("\n" + "=" * 70 + f"\nCONFIG: {cfg.name}  (model={cfg.model_name})\n" + "=" * 70)
    t0 = time.time()
    torch.manual_seed(cfg.seed)

    model, device = load_vit_model(cfg)
    loader_gen = ViTImageNetLoaderGenerator(
        cfg.data_dir, model, val_batch_size=cfg.val_batch_size,
        calib_batch_size=getattr(cfg, "fq_cali_bsz", 32), num_workers=cfg.num_workers)
    val_loader = loader_gen.val_loader(portion=cfg.val_portion)

    if isinstance(cfg, ViTFPConfig):
        results = run_evaluation(model, val_loader, cfg, device)
    else:
        calib_loader = loader_gen.calib_loader(num=cfg.fq_nsamples)
        model = apply_flatquant(model, calib_loader, cfg, device, cache_dir=cache_dir)

        if isinstance(cfg, ViTInt4NoADCConfig):
            logger.info("Evaluating with INT4 quantization, ADC floor bypassed")
            # Bypass ONLY the ADC floor — keep INT4 weight+activation quant.
            # (bypass_all would short-circuit to plain F.linear on the
            # LWC-clipped weights, which is not an INT4 measurement.)
            _set_bypass_adc_floor(model, True)
            results = run_evaluation(model, val_loader, cfg, device)
            _set_bypass_adc_floor(model, False)
        elif isinstance(cfg, ViTPTQLoRAConfig):
            model = apply_lora(model, loader_gen, cfg, device)
            results = run_evaluation(model, val_loader, cfg, device)
        else:  # ViTPTQConfig
            results = run_evaluation(model, val_loader, cfg, device)

    results["config"] = cfg.name
    results["model"] = cfg.model_name
    results["k"] = cfg.k
    logger.info(f"Config '{cfg.name}' finished in {(time.time() - t0) / 60:.1f} min "
                f"→ top1={results['top1']:.2f} top5={results['top5']:.2f}")

    # Free GPU between configs
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def print_table(all_results):
    hdr = f"{'Config':<20} {'Model':<22} {'k':>3} {'Top-1':>7} {'Top-5':>7} {'ms':>6}"
    print("\n" + "=" * len(hdr)); print(hdr); print("-" * len(hdr))
    for r in all_results.values():
        if "error" in r:
            print(f"{r.get('config', '?'):<20} ERROR: {r['error']}")
            continue
        print(f"{r['config']:<20} {r['model']:<22} {r['k']:>3} "
              f"{r['top1']:>7.2f} {r['top5']:>7.2f} {r['latency_mean_ms']:>6.1f}")
    print("=" * len(hdr) + "\n")


def main():
    p = argparse.ArgumentParser(description="ADC-aware INT4 ViT quantization")
    p.add_argument("--configs", nargs="+",
                   default=["fp", "int4_no_adc", "ptq", "ptq_lora"],
                   choices=["fp", "int4_no_adc", "ptq", "ptq_lora"])
    p.add_argument("--model", default="vit_tiny_patch16_224",
                   help="timm model, e.g. vit_tiny_patch16_224 / vit_base_patch16_224")
    p.add_argument("--k", type=int, default=4, help="ADC parallelism (thesis default 16)")
    p.add_argument("--data_dir", default=None, help="ImageNet root (else $IMAGENET_ROOT)")
    p.add_argument("--val_portion", type=float, default=1.0,
                   help="fraction of val set (strided subset) for a fast check")
    p.add_argument("--val_batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--smoke", action="store_true",
                   help="shrink calibration (few epochs / samples) for a fast run")
    p.add_argument("--fq_epochs", type=int, default=None,
                   help="override FlatQuant Stage A epochs (full-scale data, fewer epochs)")
    p.add_argument("--fq_stage_b_epochs", type=int, default=None,
                   help="override FlatQuant Stage B epochs (0 to skip Stage B)")
    p.add_argument("--output_dir", default="./outputs_vit")
    p.add_argument("--fq_cache_dir", default="./outputs_vit/fq_cache")
    p.add_argument("--no_fq_cache", action="store_true")
    args = p.parse_args()

    resolve_imagenet_root(args.data_dir)  # fail fast if ImageNet is missing
    cache_dir = None if args.no_fq_cache else args.fq_cache_dir

    all_results = {}
    for name in args.configs:
        cfg = build_config(
            name, model_name=args.model, k=args.k, data_dir=args.data_dir,
            val_portion=args.val_portion, val_batch_size=args.val_batch_size,
            num_workers=args.num_workers, output_dir=args.output_dir)
        if args.smoke and hasattr(cfg, "apply_smoke"):
            cfg.apply_smoke()
        # Explicit epoch overrides win over --smoke, letting you run full-scale
        # calibration data (fq_nsamples etc.) with a reduced epoch count.
        if args.fq_epochs is not None and hasattr(cfg, "fq_epochs"):
            cfg.fq_epochs = args.fq_epochs
        if args.fq_stage_b_epochs is not None and hasattr(cfg, "fq_stage_b_epochs"):
            cfg.fq_stage_b_epochs = args.fq_stage_b_epochs
        try:
            all_results[name] = run_config(cfg, cache_dir=cache_dir)
        except Exception as e:
            logger.error(f"Config '{name}' failed: {e}", exc_info=True)
            all_results[name] = {"config": name, "error": str(e)}

    print_table(all_results)
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "results_vit.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {out}")


if __name__ == "__main__":
    main()
