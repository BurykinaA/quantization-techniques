#!/usr/bin/env python3
"""
ADC-aware INT4 quantization of Llama — clean reference implementation.

Pipeline (for INT4 configs):
  1. Load FP16 Llama model
  2. FlatQuant calibration  → learns invertible Kronecker transforms that flatten
                              the activation distribution before INT4 quantization
  3. ADC conversion         → replace nn.Linear with TiledLinearADC (hardware model)
  4. ADC calibration        → set per-channel quantization scales (percentile method)
  5. (optional) Post-ADC LoRA → learn residual correction y = ADC(x) + B(A(x))
  6. Evaluate PPL on WikiText2 + C4, measure latency

Run all four configs and print a comparison table:
  python pipeline.py

Expected results (Llama-3.2-1B):
  fp           PPL wiki ≈ 10.5
  int4_no_adc  PPL wiki ≈ 19   (pure INT4 quantization cost, no ADC)
  best_ptq     PPL wiki ≈ 27.6 (INT4 + ADC hardware overhead)
  best_lora    PPL wiki ≈ 14.0 (INT4 + ADC + LoRA correction, beats INT8 PTQ!)
"""

import json
import logging
import os
import re
import sys
import time
from collections import Counter

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# Add the parent directory so `core` imports resolve whether the script is
# run from ADC/best/ or from the repo root.
sys.path.insert(0, os.path.dirname(__file__))

from configs import (
    BaseConfig, FPConfig, INT4NoADCConfig, BestPTQConfig, BestLoRAConfig,
    BestPTQKConfig, BestPTQKRecalConfig,
    ALL_CONFIGS, _SharedFlatQuantConfig,
)
from eval import compute_perplexity, load_eval_encodings, measure_latency
from core.adc_layers import TiledLinearADC, QATLinearADC
from core.adc_lora import apply_adc_lora, calibrate_adc_lora
from core.flat_quant import (
    FlatQuantLinear,
    apply_flatquant_to_model,
    calibrate_flat_quant,
    reparameterize_model as fq_reparameterize_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Step 0: Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(cfg: BaseConfig):
    """Load LLaMA from HuggingFace in fp16 (or the dtype set in cfg)."""
    logger.info(f"Loading {cfg.model_name} ...")
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[cfg.torch_dtype]

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    # Resolve device (handles single-GPU, multi-GPU, and CPU)
    if torch.cuda.is_available():
        if hasattr(model, "hf_device_map") and model.hf_device_map:
            device = torch.device(next(iter(model.hf_device_map.values())))
        else:
            device = next(model.parameters()).device
    else:
        device = torch.device("cpu")

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  {n_params:,} parameters, dtype={torch_dtype}, device={device}")
    return model, tokenizer, device


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Calibration data loader
# ─────────────────────────────────────────────────────────────────────────────

def build_calibration_loader(tokenizer, cfg: _SharedFlatQuantConfig) -> DataLoader:
    """
    Build a DataLoader over WikiText2 training samples for FlatQuant and
    ADC calibration. Each sample is truncated to cfg.calibration_max_length tokens.
    """
    logger.info(f"Loading calibration data from {cfg.calibration_dataset} ...")

    is_streaming = (cfg.calibration_dataset == "c4")
    if cfg.calibration_dataset == "wikitext2":
        raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    else:
        raw = load_dataset("allenai/c4", "en", split="train", streaming=True)

    # Tokenize and filter short texts
    MIN_LEN = 50
    if is_streaming:
        samples = []
        for example in raw:
            if len(samples) >= 2000:
                break
            if len(example["text"].strip()) > MIN_LEN:
                samples.append(example)
        from datasets import Dataset
        raw = Dataset.from_list(samples)
    else:
        raw = raw.filter(lambda x: len(x["text"].strip()) > MIN_LEN)
        if len(raw) > 2000:
            raw = raw.select(range(2000))

    def tokenize(examples):
        texts = [t for t in examples["text"] if t.strip() and len(t.strip()) > MIN_LEN]
        if not texts:
            return {"input_ids": [], "attention_mask": []}
        return tokenizer(
            texts,
            truncation=True,
            max_length=cfg.calibration_max_length,
            padding="max_length",
            return_tensors=None,
        )

    tokenized = raw.map(tokenize, batched=True, remove_columns=raw.column_names,
                        desc="Tokenizing calibration data")
    tokenized = tokenized.filter(lambda x: len(x["input_ids"]) > 0)

    def collate(features):
        return {
            "input_ids":      torch.tensor([f["input_ids"]      for f in features]),
            "attention_mask": torch.tensor([f["attention_mask"] for f in features]),
        }

    return DataLoader(tokenized, batch_size=cfg.calibration_batch_size,
                      shuffle=False, collate_fn=collate)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: FlatQuant calibration + ADC conversion + ADC calibration
# ─────────────────────────────────────────────────────────────────────────────

def _get_k_for_layer(cfg: _SharedFlatQuantConfig, full_name: str) -> int:
    """Return the k value for this layer: check k_per_layer first, fall back to cfg.k."""
    kpl = getattr(cfg, 'k_per_layer', {})
    if not kpl:
        return cfg.k
    if full_name in kpl:
        return kpl[full_name]
    for pattern, k in kpl.items():
        if pattern in full_name:
            return k
    return cfg.k


def _replace_linear_with_adc(model: nn.Module, cfg: _SharedFlatQuantConfig) -> nn.Module:
    """
    Replace all nn.Linear layers (except embed_tokens and lm_head) with
    TiledLinearADC.  This implements the hardware model:

        z_int = x_int · W_int^T          (integer dot product, tiled to mvm_limit columns)
        z     = floor(z_int / δ)          (ADC floor quantization)
        δ     = 2 · tile_in · q_x · q_w / (2^{b_a} · k)

    After this call the model is in the "ADC hardware simulation" mode.
    """
    EXCLUDE = {"embed_tokens", "lm_head", "_orig_attn"}

    def _replace(module: nn.Module, prefix: str = ""):
        for name, child in module.named_children():
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and not any(p in full for p in EXCLUDE):
                adc = TiledLinearADC(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=(child.bias is not None),
                    bx=cfg.bx, bw=cfg.bw, ba=cfg.ba, k=_get_k_for_layer(cfg, full),
                    ashift=False,
                    signed_activations=True,
                    mvm_limit=cfg.mvm_limit,
                    unipolar_adc=getattr(cfg, "unipolar_adc", False),
                )
                adc.load_weights(child)
                setattr(module, name, adc)
            else:
                _replace(child, full)

    _replace(model)
    n_adc = sum(1 for _, m in model.named_modules() if isinstance(m, TiledLinearADC))
    logger.info(f"  Replaced {n_adc} linear layers with TiledLinearADC")
    return model


class _ADCCalibrator:
    """
    Calibrates quantization scales for all TiledLinearADC layers.

    Runs the model in bypass mode (FP16 forward, no ADC floor) while recording
    activation and weight statistics, then sets per-channel scales so that the
    quantization covers the observed range.

    method="percentile" uses the 99.9th percentile of observed |activations|,
    which clips the top 0.1% of outliers and significantly reduces the dead-zone
    (compared to absmax which is dominated by a single large activation).
    """

    def __init__(self, model: nn.Module, cfg: _SharedFlatQuantConfig):
        self.model = model
        self.cfg = cfg
        self.stats: dict = {}

    def _make_hook(self, name: str):
        stats = self.stats
        bx, bw = self.cfg.bx, self.cfg.bw

        def hook(module, inp, _out):
            if name not in stats:
                stats[name] = {"act_absmax": [], "w_absmax": [], "module": module}
            x = inp[0].detach().float()
            stats[name]["act_absmax"].append(x.abs().max().item())
            stats[name]["w_absmax"].append(module.weight.detach().float().abs().max().item())
        return hook

    def run(self, loader: DataLoader) -> None:
        """Collect activation and weight statistics over the calibration loader."""
        hooks = []
        for name, m in self.model.named_modules():
            if isinstance(m, TiledLinearADC):
                for i, tile in enumerate(m.tiles):
                    hooks.append(tile.register_forward_hook(self._make_hook(f"{name}.tiles.{i}")))

        logger.info(f"  Registered {len(hooks)} calibration hooks ...")
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= self.cfg.num_calibration_batches:
                    break
                batch = {k: v.to(next(self.model.parameters()).device)
                         for k, v in batch.items() if isinstance(v, torch.Tensor)}
                self.model(**batch)

        for h in hooks:
            h.remove()
        logger.info(f"  Collected stats for {len(self.stats)} layer tiles")

    def apply(self) -> None:
        """Set quantization scales from collected statistics."""
        import numpy as np

        q_x = 2 ** (self.cfg.bx - 1) - 1  # e.g. 7 for INT4, 127 for INT8
        q_w = 2 ** (self.cfg.bw - 1) - 1

        updated = 0
        for name, m in self.model.named_modules():
            if not isinstance(m, TiledLinearADC):
                continue
            for i, tile in enumerate(m.tiles):
                tile_name = f"{name}.tiles.{i}"
                if tile_name not in self.stats:
                    continue
                s = self.stats[tile_name]
                act_arr = np.array(s["act_absmax"])
                w_arr   = np.array(s["w_absmax"])

                if self.cfg.calibration_method == "percentile":
                    act_scale = np.percentile(act_arr, 99.9) / q_x
                    w_scale   = np.percentile(w_arr,   99.9) / q_w
                else:  # minmax
                    act_scale = act_arr.max() / q_x
                    w_scale   = w_arr.max()   / q_w

                act_scale = max(act_scale, 1e-8)
                w_scale   = max(w_scale,   1e-8)

                if hasattr(tile, "activation_quantizer"):
                    tile.activation_quantizer.scale.data.fill_(act_scale)
                if hasattr(tile, "weight_quantizer"):
                    aq = tile.weight_quantizer
                    if aq.per_channel:
                        w = tile.weight.detach().float()
                        per_ch = w.abs().max(dim=1)[0].clamp(min=1e-6) / q_w
                        # Assign .data directly — avoids broadcast error when
                        # scale was initialised as shape [1] but per_ch is [C]
                        aq.scale.data = per_ch.to(aq.scale.device)
                    else:
                        aq.scale.data.fill_(w_scale)
                updated += 1

        logger.info(f"  Applied calibration scales to {updated} layer tiles")


def _fq_cache_key(cfg: _SharedFlatQuantConfig) -> str:
    """Short hash of all FlatQuant+ADC config params that affect the trained model."""
    import hashlib
    kpl = getattr(cfg, 'k_per_layer', {})
    kpl_str = "_".join(f"{k}:{v}" for k, v in sorted(kpl.items())) if kpl else "global"
    key_str = (f"{cfg.fq_epochs}_{cfg.fq_stage_b_epochs}_{cfg.fq_nsamples}"
               f"_{cfg.fq_lr}_{cfg.fq_w_bits}_{cfg.fq_a_bits}"
               f"_{cfg.bx}_{cfg.bw}_{cfg.ba}_{cfg.k}_{cfg.mvm_limit}"
               f"_{cfg.fq_add_diag}_{cfg.fq_lwc}_{cfg.fq_lac}"
               f"_{cfg.fq_stage_b_prop_alpha}_{cfg.fq_stage_b_diag_attn}"
               f"_unipolar{getattr(cfg, 'unipolar_adc', False)}"
               f"_kpl{kpl_str}")
    return hashlib.md5(key_str.encode()).hexdigest()[:10]


def apply_flatquant(model: nn.Module, loader: DataLoader, cfg: _SharedFlatQuantConfig,
                    device: torch.device, cache_dir: str | None = None) -> nn.Module:
    """
    Full FlatQuant pipeline:
      Stage A — 30 epochs, MLP diagonal scaling, no propagation
      Stage B — 10 epochs (lr × 0.1), adds attention diagonal, propagation α=0.5
      Reparameterize — bake learned transforms into weight matrices
      ADC conversion — replace nn.Linear with TiledLinearADC
      ADC calibration — set per-channel quantization scales

    If cache_dir is set, saves the fully-calibrated model after the first run and
    reloads it on subsequent runs — skipping the ~1 hour FlatQuant training.
    """
    # ── Cache check ──────────────────────────────────────────────────────────
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"fq_adc_{_fq_cache_key(cfg)}.pt")
        if os.path.exists(cache_path):
            logger.info(f"Loading cached FlatQuant+ADC model: {cache_path}")
            model = torch.load(cache_path, weights_only=False, map_location=device)
            model = model.to(device)
            return model
        logger.info(f"FlatQuant cache not found — will save to: {cache_path}")
    logger.info("─── FlatQuant Stage A ───────────────────────────────────────")
    logger.info(f"  {cfg.fq_epochs} epochs, lr={cfg.fq_lr}, nsamples={cfg.fq_nsamples}")
    logger.info("  Diagonal scaling: MLP blocks only")

    fq_adc_config = dict(bx=cfg.bx, bw=cfg.bw, ba=cfg.ba, k=cfg.k,
                         mvm_limit=cfg.mvm_limit, signed_activations=True)

    # Apply FlatQuant wrappers (FlatQuantLinear around each nn.Linear)
    model = apply_flatquant_to_model(
        model,
        w_bits=cfg.fq_w_bits, a_bits=cfg.fq_a_bits,
        add_diag=cfg.fq_add_diag, lwc=cfg.fq_lwc, lac=cfg.fq_lac,
        adc_config=fq_adc_config,
    )

    # Stage A: train Kronecker transforms (MLP diagonal only, no propagation)
    model = calibrate_flat_quant(
        model, dataloader=loader, device=device,
        nsamples=cfg.fq_nsamples, cali_bsz=cfg.fq_cali_bsz,
        epochs=cfg.fq_epochs, flat_lr=cfg.fq_lr,
        add_diag=cfg.fq_add_diag, lwc=cfg.fq_lwc, lac=cfg.fq_lac,
        propagate_quant_inputs=False,  # Stage A: no propagation
        diag_attn=False,    # Stage A: MLP diagonal only
        diag_mlp=True,
        diag_mlp_up=True, diag_mlp_down=True,
    )

    logger.info("─── FlatQuant Stage B ───────────────────────────────────────")
    logger.info(f"  {cfg.fq_stage_b_epochs} epochs, lr={cfg.fq_lr * 0.1}")
    logger.info(f"  Propagation α={cfg.fq_stage_b_prop_alpha}, adds attention diagonal")

    # Stage B: add attention diagonal + propagated calibration (α=0.5)
    model = calibrate_flat_quant(
        model, dataloader=loader, device=device,
        nsamples=cfg.fq_nsamples, cali_bsz=cfg.fq_cali_bsz,
        epochs=cfg.fq_stage_b_epochs, flat_lr=cfg.fq_lr * 0.1,
        add_diag=cfg.fq_add_diag, lwc=cfg.fq_lwc, lac=cfg.fq_lac,
        propagate_quant_inputs=True,
        propagate_quant_alpha=cfg.fq_stage_b_prop_alpha,
        diag_attn=cfg.fq_stage_b_diag_attn,
        diag_mlp=True,
        diag_mlp_up=True, diag_mlp_down=True,
    )

    logger.info("─── Reparameterize FlatQuant transforms into weights ─────────")
    model = model.to(device)
    model = fq_reparameterize_model(model)

    logger.info("─── Convert to TiledLinearADC ────────────────────────────────")
    model = _replace_linear_with_adc(model, cfg)
    model = model.to(device)

    logger.info("─── ADC calibration (percentile scales) ─────────────────────")
    # Run in bypass mode during calibration (FP16 forward, no ADC floor)
    _set_bypass_adc(model, bypass=True)
    calibrator = _ADCCalibrator(model, cfg)
    calibrator.run(loader)
    _set_bypass_adc(model, bypass=False)
    calibrator.apply()

    # Freeze quantizer scales
    for _, m in model.named_modules():
        if hasattr(m, "set_quantizer_mode"):
            m.set_quantizer_mode("fixed")

    # ── Save cache ───────────────────────────────────────────────────────────
    if cache_path:
        logger.info(f"Saving FlatQuant+ADC model to cache: {cache_path}")
        torch.save(model, cache_path)

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Per-layer k search
# ─────────────────────────────────────────────────────────────────────────────

def search_k_per_layer(
    model: nn.Module,
    cfg: _SharedFlatQuantConfig,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Find the optimal k per TiledLinearADC layer using dead-rate statistics.

    Algorithm:
      1. Enable stats capture on every QATLinearADC tile.
      2. Run one forward pass with bypass_adc=True to collect raw y_int samples.
      3. For each layer, aggregate all tile samples and sweep k candidates.
         Choose the largest k where dead_rate = mean(|y_int| < δ) ≤ target.
      4. Return {module_path: k} dict (paths match model.named_modules() output).
    """
    candidates = sorted(getattr(cfg, 'k_search_candidates', (4, 8, 16, 32, 64)))
    target     = getattr(cfg, 'k_search_target_dead_rate', 0.05)

    # Enable capture on all QATLinearADC tiles
    all_tiles = {name: m for name, m in model.named_modules()
                 if isinstance(m, QATLinearADC)}
    for t in all_tiles.values():
        t._capturing = True
        t._y_int_samples = []

    # One forward pass in bypass mode (get y_int before ADC quantisation)
    saved_bypass = {}
    for name, m in model.named_modules():
        if isinstance(m, TiledLinearADC):
            saved_bypass[name] = getattr(m, 'bypass_adc', False)
            m.set_bypass_adc(True)

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()
                     if isinstance(v, torch.Tensor)}
            model(**batch)

    # Restore bypass state
    for name, m in model.named_modules():
        if isinstance(m, TiledLinearADC) and name in saved_bypass:
            m.set_bypass_adc(saved_bypass[name])

    # Aggregate per TiledLinearADC layer (strip ".tiles.N" suffix)
    from collections import defaultdict
    layer_samples: dict = defaultdict(list)
    layer_tile_ref: dict = {}
    for tile_name, tile in all_tiles.items():
        if not tile._y_int_samples:
            continue
        layer_name = re.sub(r'\.tiles\.\d+$', '', tile_name)
        layer_samples[layer_name].append(torch.cat(tile._y_int_samples))
        layer_tile_ref[layer_name] = tile
        tile._capturing = False
        tile._y_int_samples = []

    # For each layer pick largest k with dead_rate ≤ target
    k_per_layer: dict = {}
    for layer_name, sample_list in layer_samples.items():
        abs_y = torch.cat(sample_list)
        tile  = layer_tile_ref[layer_name]
        M = 2.0 * tile.in_features * tile._activation_level_magnitude * tile._weight_level_max

        best_k = candidates[0]
        for k in candidates:
            delta = M / float((2 ** tile.ba) * k)
            dead_rate = (abs_y < delta).float().mean().item()
            if dead_rate <= target:
                best_k = k   # take largest valid k (lowest hardware cost)

        k_per_layer[layer_name] = best_k

    dist = Counter(k_per_layer.values())
    logger.info(f"  k_per_layer search done — distribution: {dict(sorted(dist.items()))}")
    return k_per_layer


def _apply_flatquant_with_k(
    model: nn.Module,
    loader: DataLoader,
    cfg: _SharedFlatQuantConfig,
    device: torch.device,
    cache_dir: str | None,
) -> nn.Module:
    """Re-run apply_flatquant with cfg.k_per_layer injected into each FlatQuantLinear.

    After apply_flatquant_to_model creates the wrappers, each FlatQuantLinear's
    _adc_config['k'] is updated to the per-layer value before calibration starts,
    so FlatQuant trains with the correct delta for each projection.
    """
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"fq_adc_{_fq_cache_key(cfg)}.pt")
        if os.path.exists(cache_path):
            logger.info(f"Loading cached FlatQuant+ADC model (recal): {cache_path}")
            m = torch.load(cache_path, weights_only=False, map_location=device)
            return m.to(device)
        logger.info(f"FlatQuant recal cache not found — will save to: {cache_path}")

    fq_adc_config = dict(bx=cfg.bx, bw=cfg.bw, ba=cfg.ba, k=cfg.k,
                         mvm_limit=cfg.mvm_limit, signed_activations=True)

    model = apply_flatquant_to_model(
        model,
        w_bits=cfg.fq_w_bits, a_bits=cfg.fq_a_bits,
        add_diag=cfg.fq_add_diag, lwc=cfg.fq_lwc, lac=cfg.fq_lac,
        adc_config=fq_adc_config,
    )

    # Inject per-layer k into each FlatQuantLinear's adc_config copy.
    # Path mapping: FlatQuantLinear at "a.b.q_proj" → TiledLinearADC will be at
    # "a.b.q_proj.linear", which is the key in k_per_layer.
    n_updated = 0
    for name, m in model.named_modules():
        if isinstance(m, FlatQuantLinear) and m._adc_config is not None:
            adc_path = name + ".linear"
            k_val = _get_k_for_layer(cfg, adc_path)
            if k_val != m._adc_config.get('k'):
                m._adc_config['k'] = k_val
                n_updated += 1
    logger.info(f"  Injected per-layer k into {n_updated} FlatQuantLinear layers")

    logger.info("─── FlatQuant Stage A (recal with per-layer k) ──────────────")
    model = calibrate_flat_quant(
        model, dataloader=loader, device=device,
        nsamples=cfg.fq_nsamples, cali_bsz=cfg.fq_cali_bsz,
        epochs=cfg.fq_epochs, flat_lr=cfg.fq_lr,
        add_diag=cfg.fq_add_diag, lwc=cfg.fq_lwc, lac=cfg.fq_lac,
        propagate_quant_inputs=False, diag_attn=False,
        diag_mlp=True, diag_mlp_up=True, diag_mlp_down=True,
    )
    logger.info("─── FlatQuant Stage B (recal with per-layer k) ──────────────")
    model = calibrate_flat_quant(
        model, dataloader=loader, device=device,
        nsamples=cfg.fq_nsamples, cali_bsz=cfg.fq_cali_bsz,
        epochs=cfg.fq_stage_b_epochs, flat_lr=cfg.fq_lr * 0.1,
        add_diag=cfg.fq_add_diag, lwc=cfg.fq_lwc, lac=cfg.fq_lac,
        propagate_quant_inputs=True,
        propagate_quant_alpha=cfg.fq_stage_b_prop_alpha,
        diag_attn=cfg.fq_stage_b_diag_attn,
        diag_mlp=True, diag_mlp_up=True, diag_mlp_down=True,
    )

    model = model.to(device)
    model = fq_reparameterize_model(model)
    model = _replace_linear_with_adc(model, cfg)   # uses k_per_layer via _get_k_for_layer
    model = model.to(device)

    _set_bypass_adc(model, bypass=True)
    calibrator = _ADCCalibrator(model, cfg)
    calibrator.run(loader)
    _set_bypass_adc(model, bypass=False)
    calibrator.apply()

    for _, m in model.named_modules():
        if hasattr(m, "set_quantizer_mode"):
            m.set_quantizer_mode("fixed")

    if cache_path:
        logger.info(f"Saving recal FlatQuant+ADC model to cache: {cache_path}")
        torch.save(model, cache_path)

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Helper: toggle ADC floor quantization
# ─────────────────────────────────────────────────────────────────────────────

def _set_bypass_adc(model: nn.Module, bypass: bool) -> None:
    """
    Enable or disable the ADC floor quantization (z = floor(z_int / δ)).
    When bypass=True the model runs as a standard INT4 model without the ADC
    hardware overhead — useful for isolating the pure quantization error.
    """
    for _, m in model.named_modules():
        if isinstance(m, TiledLinearADC):
            if hasattr(m, "set_bypass_adc"):
                m.set_bypass_adc(bypass)
            if hasattr(m, "set_bypass_all"):
                m.set_bypass_all(bypass)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 (optional): Post-ADC LoRA correction
# ─────────────────────────────────────────────────────────────────────────────

def apply_lora(model: nn.Module, loader: DataLoader, cfg: BestLoRAConfig,
               device: torch.device) -> nn.Module:
    """
    Wrap target layers with ResidualLoRATiledLinearADC and train.

    Architecture:
        y = TiledLinearADC(x) + (lora_alpha / lora_rank) · B(A(x.float()))

    LoRA matrices A and B are FP32 and added AFTER the ADC floor quantization.
    This makes training stable: gradients never flow through the floor() op.

    Loss: L = L_CE + λ · KL(student ‖ teacher_fp)
    where teacher_fp is a frozen copy of the original FP16 model.
    """
    logger.info(f"─── Post-ADC LoRA (rank={cfg.lora_rank}, α={cfg.lora_alpha}) ────")
    logger.info(f"  Targets: {list(cfg.lora_target_modules)}")
    logger.info(f"  Loss: {cfg.lora_loss}, KL weight={cfg.lora_kl_weight}, T={cfg.lora_kl_temperature}")

    model = apply_adc_lora(
        model,
        target_modules=list(cfg.lora_target_modules),
        rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        mode=cfg.lora_mode,
        layer_indices=None,  # all layers
    )
    model = calibrate_adc_lora(
        model,
        dataloader=loader,
        device=device,
        nsamples=cfg.fq_nsamples,
        cali_bsz=cfg.fq_cali_bsz,
        epochs=cfg.lora_epochs,
        lora_lr=cfg.lora_lr,
        lora_loss=cfg.lora_loss,
        teacher_name_or_path=cfg.model_name if cfg.lora_loss == "ce_kl" else None,
        kl_weight=cfg.lora_kl_weight,
        kl_temperature=cfg.lora_kl_temperature,
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(model: nn.Module, tokenizer, cfg: BaseConfig,
                   device: torch.device) -> dict:
    """Evaluate PPL on WikiText2 + C4 and measure latency."""
    results = {}
    model.eval()

    for ds in cfg.eval_datasets:
        split = "test" if ds == "wikitext2" else "validation"
        encodings = load_eval_encodings(
            ds, split, tokenizer,
            max_samples=cfg.max_eval_samples,
        )
        if encodings is None:
            logger.warning(f"  Skipping {ds} (could not load)")
            continue
        m = compute_perplexity(
            model, encodings, device,
            max_length=cfg.max_length,
            stride=cfg.stride,
            desc=f"{cfg.name} / {ds}",
        )
        results[ds] = m["perplexity"]
        logger.info(f"  {ds.upper():10s}  PPL = {m['perplexity']:.2f}")

    lat = measure_latency(model, device, seq_len=512)
    results["latency_mean_ms"]  = lat["latency_mean_ms"]
    results["latency_p95_ms"]   = lat["latency_p95_ms"]
    results["throughput_tok_s"] = lat["throughput_tok_s"]
    logger.info(
        f"  Latency  mean={lat['latency_mean_ms']:.1f}ms  "
        f"p95={lat['latency_p95_ms']:.1f}ms  "
        f"throughput={lat['throughput_tok_s']:.0f} tok/s"
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator: run one configuration end-to-end
# ─────────────────────────────────────────────────────────────────────────────

def run_config(cfg: BaseConfig, cache_dir: str | None = None) -> dict:
    """Run a single configuration and return evaluation results."""
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"CONFIG: {cfg.name}")
    logger.info("=" * 70)
    t0 = time.time()
    set_seed(cfg.seed)

    model, tokenizer, device = load_model(cfg)

    if isinstance(cfg, FPConfig):
        # Full precision: skip all quantization steps
        logger.info("Full precision — no quantization")
        results = run_evaluation(model, tokenizer, cfg, device)

    else:
        # All quantized configs share the same FlatQuant training
        loader = build_calibration_loader(tokenizer, cfg)
        model = apply_flatquant(model, loader, cfg, device, cache_dir=cache_dir)

        if isinstance(cfg, INT4NoADCConfig):
            # Evaluate WITHOUT ADC floor quantization → isolates INT4 quantization
            logger.info("Evaluating in bypass mode (INT4 quantization, NO ADC floor)")
            _set_bypass_adc(model, bypass=True)
            results = run_evaluation(model, tokenizer, cfg, device)
            _set_bypass_adc(model, bypass=False)

        elif isinstance(cfg, BestLoRAConfig):
            # Apply and train post-ADC LoRA, then evaluate with full ADC
            model = apply_lora(model, loader, cfg, device)
            results = run_evaluation(model, tokenizer, cfg, device)

        elif isinstance(cfg, BestPTQKConfig):
            # Stage 1: FlatQuant already done (global k).  Discover per-layer k.
            cfg.k_per_layer = search_k_per_layer(model, cfg, loader, device)

            if getattr(cfg, 'fq_recal_with_k', False):
                # Variant B: reload FP model, retrain FlatQuant with per-layer k active
                logger.info("─── Reloading FP model for FlatQuant recalibration ──────────")
                model_fp2, _, _ = load_model(cfg)
                model = _apply_flatquant_with_k(model_fp2, loader, cfg, device, cache_dir)
            else:
                # Variant A: apply found k in-place (fast, no retrain)
                n_updated = 0
                for name, m in model.named_modules():
                    if isinstance(m, TiledLinearADC) and name in cfg.k_per_layer:
                        m.set_k(cfg.k_per_layer[name])
                        n_updated += 1
                logger.info(f"  Applied per-layer k to {n_updated} TiledLinearADC layers in-place")

            results = run_evaluation(model, tokenizer, cfg, device)

        else:
            # BestPTQConfig: full ADC hardware model, no LoRA
            results = run_evaluation(model, tokenizer, cfg, device)

    elapsed = time.time() - t0
    logger.info(f"Config '{cfg.name}' finished in {elapsed / 60:.1f} min")

    wandb_project = getattr(cfg, "wandb_project", "")
    if wandb_project and _WANDB_AVAILABLE:
        try:
            import dataclasses
            cfg_dict = dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else vars(cfg)
            wandb.init(project=wandb_project, name=cfg.name, config=cfg_dict, reinit=True)
            wandb.log({
                "ppl_wikitext2": results.get("wikitext2"),
                "ppl_c4":        results.get("c4"),
                "latency_ms":    results.get("latency_mean_ms"),
                "throughput":    results.get("throughput_tok_s"),
            })
            wandb.finish()
            logger.info(f"  WandB logged to project '{wandb_project}', run '{cfg.name}'")
        except Exception as e:
            logger.warning(f"  WandB logging failed: {e}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Results table
# ─────────────────────────────────────────────────────────────────────────────

def print_results_table(all_results: dict) -> None:
    """Print a formatted comparison table to stdout."""
    header = f"{'Config':<14}  {'Wiki PPL':>8}  {'C4 PPL':>8}  {'Lat (ms)':>9}  {'Tok/s':>8}"
    print()
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name, r in all_results.items():
        wiki = f"{r['wikitext2']:.2f}" if "wikitext2" in r else "  —"
        c4   = f"{r['c4']:.2f}"        if "c4"        in r else "  —"
        lat  = f"{r.get('latency_mean_ms', 0):.1f}"
        tput = f"{r.get('throughput_tok_s', 0):.0f}"
        print(f"{name:<14}  {wiki:>8}  {c4:>8}  {lat:>9}  {tput:>8}")
    print("=" * len(header))
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="ADC-aware INT4 Llama quantization — reference implementation"
    )
    parser.add_argument(
        "--configs", nargs="+",
        choices=["fp", "int4_no_adc", "best_ptq", "best_lora",
                 "best_ptq_k", "best_ptq_k_recal", "all"],
        default=["all"],
        help="Which configs to run (default: all)",
    )
    parser.add_argument("--output_dir", default="./outputs",
                        help="Where to save results.json")
    parser.add_argument("--fq_cache_dir", default="./outputs/fq_cache",
                        help="Directory to cache FlatQuant+ADC models (skips ~1h training on re-runs)")
    parser.add_argument("--no_fq_cache", action="store_true",
                        help="Disable FlatQuant caching (always retrain from scratch)")
    args = parser.parse_args()

    cache_dir = None if args.no_fq_cache else args.fq_cache_dir

    # Select configs to run
    config_map = {c.name: c for c in ALL_CONFIGS}
    if args.configs == ["all"] or "all" in args.configs:
        selected = ALL_CONFIGS
    else:
        selected = [config_map[n] for n in args.configs]

    # Override output dir
    for cfg in selected:
        cfg.output_dir = args.output_dir

    all_results = {}
    for cfg in selected:
        try:
            all_results[cfg.name] = run_config(cfg, cache_dir=cache_dir)
        except Exception as e:
            logger.error(f"Config '{cfg.name}' failed: {e}", exc_info=True)
            all_results[cfg.name] = {"error": str(e)}

    print_results_table({k: v for k, v in all_results.items() if "error" not in v})

    # Save to JSON
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
