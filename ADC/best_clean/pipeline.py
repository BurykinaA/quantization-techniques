#!/usr/bin/env python3
"""
ADC-aware INT4 quantization of Llama — clean reference implementation.

Pipeline (for the INT4 configs):
  1. Load FP16 Llama
  2. FlatQuant calibration  - learn invertible Kronecker transforms that flatten
                              the activation distribution before INT4 quantization
  3. ADC conversion         - replace nn.Linear with TiledLinearADC (hardware model)
  4. ADC calibration        - set per-channel quantization scales (percentile method)
  5. (best_lora) Post-ADC LoRA - learn residual correction y = ADC(x) + B(A(x))
  6. Evaluate PPL on WikiText2 + C4, measure latency

Run:
  python pipeline.py --configs all
"""

import json
import logging
import os
import sys
import time

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# Resolve `core` imports whether run from ADC/best_clean/ or the repo root.
sys.path.insert(0, os.path.dirname(__file__))

from configs import (
    BaseConfig, FPConfig, INT4NoADCConfig, BestPTQConfig, BestLoRAConfig,
    ALL_CONFIGS, _SharedFlatQuantConfig,
)
from eval import compute_perplexity, load_eval_encodings, measure_latency
from core.adc_layers import TiledLinearADC
from core.adc_lora import apply_adc_lora, calibrate_adc_lora
from core.flat_quant import (
    apply_flatquant_to_model,
    calibrate_flat_quant,
    reparameterize_model as fq_reparameterize_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(cfg: BaseConfig):
    """Load Llama from HuggingFace in the dtype set by cfg.torch_dtype."""
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
# Calibration data loader
# ─────────────────────────────────────────────────────────────────────────────

def build_calibration_loader(tokenizer, cfg: _SharedFlatQuantConfig) -> DataLoader:
    """DataLoader over WikiText2/C4 training samples for FlatQuant and ADC calibration."""
    logger.info(f"Loading calibration data from {cfg.calibration_dataset} ...")

    is_streaming = (cfg.calibration_dataset == "c4")
    if cfg.calibration_dataset == "wikitext2":
        raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    else:
        raw = load_dataset("allenai/c4", "en", split="train", streaming=True)

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
# FlatQuant calibration + ADC conversion + ADC calibration
# ─────────────────────────────────────────────────────────────────────────────

def _replace_linear_with_adc(model: nn.Module, cfg: _SharedFlatQuantConfig) -> nn.Module:
    """
    Replace every nn.Linear (except embed_tokens and lm_head) with TiledLinearADC:

        z_int = x_int . W_int^T            (integer dot product, tiled to mvm_limit columns)
        z     = clamp(floor(z_int / delta), na, pa) * delta
        delta = 2 * tile_in * q_x * q_w / (2^ba * k)
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
                    bx=cfg.bx, bw=cfg.bw, ba=cfg.ba, k=cfg.k,
                    ashift=False,
                    signed_activations=True,
                    mvm_limit=cfg.mvm_limit,
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
    Set quantization scales for all TiledLinearADC layers.

    Runs the model in bypass mode (FP16 forward, no ADC floor) while recording
    activation/weight statistics, then sets per-channel scales. The "percentile"
    method uses the 99.9th percentile of observed |activations|, clipping the top
    0.1% of outliers (absmax is dominated by a single large activation).
    """

    def __init__(self, model: nn.Module, cfg: _SharedFlatQuantConfig):
        self.model = model
        self.cfg = cfg
        self.stats: dict = {}

    def _make_hook(self, name: str):
        stats = self.stats

        def hook(module, inp, _out):
            if name not in stats:
                stats[name] = {"act_absmax": [], "w_absmax": [], "module": module}
            x = inp[0].detach().float()
            stats[name]["act_absmax"].append(x.abs().max().item())
            stats[name]["w_absmax"].append(module.weight.detach().float().abs().max().item())
        return hook

    def run(self, loader: DataLoader) -> None:
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
        q_x = 2 ** (self.cfg.bx - 1) - 1
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
                        aq.scale.data = per_ch.to(aq.scale.device)
                    else:
                        aq.scale.data.fill_(w_scale)
                updated += 1

        logger.info(f"  Applied calibration scales to {updated} layer tiles")


def _fq_cache_key(cfg: _SharedFlatQuantConfig) -> str:
    """Short hash of the FlatQuant+ADC config params that affect the trained model."""
    import hashlib
    key_str = (f"{cfg.fq_epochs}_{cfg.fq_stage_b_epochs}_{cfg.fq_nsamples}"
               f"_{cfg.fq_lr}_{cfg.fq_w_bits}_{cfg.fq_a_bits}"
               f"_{cfg.bx}_{cfg.bw}_{cfg.ba}_{cfg.k}_{cfg.mvm_limit}"
               f"_{cfg.fq_add_diag}_{cfg.fq_lwc}_{cfg.fq_lac}"
               f"_{cfg.fq_stage_b_prop_alpha}_{cfg.fq_stage_b_diag_attn}")
    return hashlib.md5(key_str.encode()).hexdigest()[:10]


def apply_flatquant(model: nn.Module, loader: DataLoader, cfg: _SharedFlatQuantConfig,
                    device: torch.device, cache_dir: str | None = None) -> nn.Module:
    """
    Full FlatQuant pipeline:
      Stage A       - 30 epochs, MLP diagonal scaling, no propagation
      Stage B       - 10 epochs (lr x 0.1), adds attention diagonal, propagation alpha=0.5
      Reparameterize - bake learned transforms into weight matrices
      ADC conversion - replace nn.Linear with TiledLinearADC
      ADC calibration - set per-channel quantization scales

    If cache_dir is set, the fully-calibrated model is saved on the first run and
    reloaded on later runs, skipping the ~1 hour of FlatQuant training.
    """
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"fq_adc_{_fq_cache_key(cfg)}.pt")
        if os.path.exists(cache_path):
            logger.info(f"Loading cached FlatQuant+ADC model: {cache_path}")
            model = torch.load(cache_path, weights_only=False, map_location=device)
            return model.to(device)
        logger.info(f"FlatQuant cache not found — will save to: {cache_path}")

    logger.info("─── FlatQuant Stage A ───────────────────────────────────────")
    logger.info(f"  {cfg.fq_epochs} epochs, lr={cfg.fq_lr}, nsamples={cfg.fq_nsamples}")

    fq_adc_config = dict(bx=cfg.bx, bw=cfg.bw, ba=cfg.ba, k=cfg.k,
                         mvm_limit=cfg.mvm_limit, signed_activations=True)

    model = apply_flatquant_to_model(
        model,
        w_bits=cfg.fq_w_bits, a_bits=cfg.fq_a_bits,
        add_diag=cfg.fq_add_diag, lwc=cfg.fq_lwc, lac=cfg.fq_lac,
        adc_config=fq_adc_config,
    )

    model = calibrate_flat_quant(
        model, dataloader=loader, device=device,
        nsamples=cfg.fq_nsamples, cali_bsz=cfg.fq_cali_bsz,
        epochs=cfg.fq_epochs, flat_lr=cfg.fq_lr,
        add_diag=cfg.fq_add_diag, lwc=cfg.fq_lwc, lac=cfg.fq_lac,
        propagate_quant_inputs=False,
        diag_attn=False,
        diag_mlp=True,
        diag_mlp_up=True, diag_mlp_down=True,
    )

    logger.info("─── FlatQuant Stage B ───────────────────────────────────────")
    logger.info(f"  {cfg.fq_stage_b_epochs} epochs, lr={cfg.fq_lr * 0.1}, "
                f"propagation α={cfg.fq_stage_b_prop_alpha}")

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
    model = fq_reparameterize_model(model.to(device))

    logger.info("─── Convert to TiledLinearADC ────────────────────────────────")
    model = _replace_linear_with_adc(model, cfg).to(device)

    logger.info("─── ADC calibration (percentile scales) ─────────────────────")
    _set_bypass_adc(model, bypass=True)
    calibrator = _ADCCalibrator(model, cfg)
    calibrator.run(loader)
    _set_bypass_adc(model, bypass=False)
    calibrator.apply()

    for _, m in model.named_modules():
        if hasattr(m, "set_quantizer_mode"):
            m.set_quantizer_mode("fixed")

    if cache_path:
        logger.info(f"Saving FlatQuant+ADC model to cache: {cache_path}")
        torch.save(model, cache_path)

    return model


def _set_bypass_adc(model: nn.Module, bypass: bool) -> None:
    """Enable/disable the ADC floor (z = floor(z_int / delta)) on all TiledLinearADC layers."""
    for _, m in model.named_modules():
        if isinstance(m, TiledLinearADC):
            if hasattr(m, "set_bypass_adc"):
                m.set_bypass_adc(bypass)
            if hasattr(m, "set_bypass_all"):
                m.set_bypass_all(bypass)


# ─────────────────────────────────────────────────────────────────────────────
# Post-ADC LoRA correction (best_lora)
# ─────────────────────────────────────────────────────────────────────────────

def apply_lora(model: nn.Module, loader: DataLoader, cfg: BestLoRAConfig,
               device: torch.device) -> nn.Module:
    """
    Wrap target layers with ResidualLoRATiledLinearADC and train:

        y = TiledLinearADC(x) + (lora_alpha / lora_rank) * B(A(x.float()))

    LoRA matrices A and B are FP32 and added AFTER the ADC floor, so training is
    stable (gradients never flow through floor()).
    Loss: L = L_CE + lambda * KL(student || teacher_fp).
    """
    logger.info(f"─── Post-ADC LoRA (rank={cfg.lora_rank}, α={cfg.lora_alpha}) ────")
    logger.info(f"  Targets: {list(cfg.lora_target_modules)}")
    logger.info(f"  Loss: {cfg.lora_loss}, KL weight={cfg.lora_kl_weight}, T={cfg.lora_kl_temperature}")

    model = apply_adc_lora(
        model,
        target_modules=list(cfg.lora_target_modules),
        rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        layer_indices=None,
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
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(model: nn.Module, tokenizer, cfg: BaseConfig,
                   device: torch.device) -> dict:
    """Evaluate PPL on WikiText2 + C4 and measure latency."""
    results = {}
    model.eval()

    for ds in cfg.eval_datasets:
        split = "test" if ds == "wikitext2" else "validation"
        encodings = load_eval_encodings(ds, split, tokenizer, max_samples=cfg.max_eval_samples)
        if encodings is None:
            logger.warning(f"  Skipping {ds} (could not load)")
            continue
        m = compute_perplexity(
            model, encodings, device,
            max_length=cfg.max_length, stride=cfg.stride,
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
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_config(cfg: BaseConfig, cache_dir: str | None = None) -> dict:
    """Run a single configuration end-to-end and return evaluation results."""
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"CONFIG: {cfg.name}")
    logger.info("=" * 70)
    t0 = time.time()
    set_seed(cfg.seed)

    model, tokenizer, device = load_model(cfg)

    if isinstance(cfg, FPConfig):
        logger.info("Full precision — no quantization")
        results = run_evaluation(model, tokenizer, cfg, device)
    else:
        loader = build_calibration_loader(tokenizer, cfg)
        model = apply_flatquant(model, loader, cfg, device, cache_dir=cache_dir)

        if isinstance(cfg, INT4NoADCConfig):
            logger.info("Evaluating in bypass mode (INT4 quantization, NO ADC floor)")
            _set_bypass_adc(model, bypass=True)
            results = run_evaluation(model, tokenizer, cfg, device)
            _set_bypass_adc(model, bypass=False)
        elif isinstance(cfg, BestLoRAConfig):
            model = apply_lora(model, loader, cfg, device)
            results = run_evaluation(model, tokenizer, cfg, device)
        else:  # BestPTQConfig
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


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="ADC-aware INT4 Llama quantization — reference implementation"
    )
    parser.add_argument(
        "--configs", nargs="+",
        choices=["fp", "int4_no_adc", "best_ptq", "best_lora", "all"],
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

    config_map = {c.name: c for c in ALL_CONFIGS}
    if "all" in args.configs:
        selected = ALL_CONFIGS
    else:
        selected = [config_map[n] for n in args.configs]

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

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
