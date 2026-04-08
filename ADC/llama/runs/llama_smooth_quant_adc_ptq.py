#!/usr/bin/env python3
"""
Post-Training Quantization (PTQ) for ADC-based LLaMA models with optional
SmoothQuant or FlatQuant preprocessing.

Pipeline: Preprocess (SmoothQuant/FlatQuant/None) → ADC Convert → Calibrate → Evaluate → Visualize

Supports:
- meta-llama/Llama-3.1-8B
- meta-llama/Llama-3.2-3B
- meta-llama/Llama-3.2-1B
"""

import argparse
import os
import logging
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
)
from torch.utils.data import DataLoader
from datetime import datetime

from ADC.llama.core.adc_layers import TiledLinearADC, QATLinearADC
from ADC.llama.core.smooth_quant import (
    calibrate_smooth_scales,
    apply_smooth_quant,
)
from ADC.llama.core.adc_lora import apply_adc_lora, calibrate_adc_lora
from ADC.llama.core.flat_quant import (
    apply_flatquant_to_model,
    calibrate_flat_quant,
    reparameterize_model as fq_reparameterize_model,
    strip_flatquant_wrappers,
    save_flat_transforms,
    load_flat_transforms,
    capture_layer_outputs,
    compare_layer_outputs,
    propagate_alpha_adc_to_tiled,
)

import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================================================================
# LLaMA ADC Converter (identical to llama_adc_ptq.py)
# =========================================================================

class LlamaADCConverter:
    """Convert LLaMA model to use ADC QAT layers for causal LM."""

    @staticmethod
    def is_after_silu(name: str) -> bool:
        """
        Detect if this linear layer follows a SiLU activation.
        In LLaMA MLP: output = down_proj(silu(gate_proj(x)) * up_proj(x))
        So down_proj receives SiLU output.
        """
        return "down_proj" in name

    @staticmethod
    def replace_linear_with_adc(
        model: nn.Module,
        bx: int = 8,
        bw: int = 8,
        ba: int = 8,
        k: int = 4,
        ashift: bool = False,
        signed_activations: bool = None,
        exclude_patterns: list[str] | None = None,
        mvm_limit: int = 256,
        use_kurtosis_loss: bool = False,
        kurtosis_weight: float = 0.0,
        target_kurtosis: float = 1.8,
    ) -> nn.Module:
        """Replace all nn.Linear layers in the LLaMA model with TiledLinearADC."""
        if exclude_patterns is None:
            exclude_patterns = ["embed_tokens", "lm_head"]

        def should_exclude(name: str) -> bool:
            return any(pat in name for pat in exclude_patterns)

        def replace_recursive(module: nn.Module, name: str = ""):
            for child_name, child_module in module.named_children():
                full_name = f"{name}.{child_name}" if name else child_name

                if isinstance(child_module, nn.Linear) and not should_exclude(full_name):
                    layer_ashift = ashift and LlamaADCConverter.is_after_silu(full_name)

                    if signed_activations is not None:
                        layer_signed_activations = signed_activations
                    else:
                        layer_signed_activations = not layer_ashift

                    adc_layer = TiledLinearADC(
                        in_features=child_module.in_features,
                        out_features=child_module.out_features,
                        bias=(child_module.bias is not None),
                        bx=bx,
                        bw=bw,
                        ba=ba,
                        k=k,
                        ashift=layer_ashift,
                        signed_activations=layer_signed_activations,
                        mvm_limit=mvm_limit,
                        use_kurtosis_loss=use_kurtosis_loss,
                        kurtosis_weight=kurtosis_weight,
                        target_kurtosis=target_kurtosis,
                    )
                    adc_layer.load_weights(child_module)

                    quant_type = "A-shift (asymmetric)" if layer_ashift else "symmetric"
                    logger.info(f"Replaced {full_name} with TiledLinearADC ({quant_type}, bx={bx}, bw={bw}, ba={ba}, k={k})")

                    setattr(module, child_name, adc_layer)
                else:
                    replace_recursive(child_module, full_name)

        replace_recursive(model)
        return model

    @staticmethod
    def count_adc_layers(model: nn.Module) -> dict:
        """Count ADC and regular linear layers in the model."""
        counts = {"adc_linear": 0, "regular_linear": 0, "total_params": 0}
        for _, module in model.named_modules():
            if isinstance(module, TiledLinearADC):
                counts["adc_linear"] += 1
            elif isinstance(module, nn.Linear):
                counts["regular_linear"] += 1
            if hasattr(module, "parameters"):
                counts["total_params"] += sum(p.numel() for p in module.parameters())
        return counts


# =========================================================================
# Utility functions (shared with llama_adc_ptq.py)
# =========================================================================

def append_current_date_to_path(path_base: str) -> str:
    """Append current date to a path (for output directories)"""
    current_date = datetime.now().strftime("%Y%m%d")
    path_with_date = f"{path_base}_{current_date}"
    logger.info(f"Output directory with current date: {path_with_date}")
    return path_with_date


def show_model_with_adc_hooks(model, visualize_patterns):
    """Print the full module tree and highlight ADC hooks and visualization targets."""
    def will_visualise(name):
        return any(pat in name for pat in visualize_patterns)

    def is_tile_child(name):
        return '.tiles.' in name

    def format_line(level, name, module):
        if is_tile_child(name):
            return None
        bullet = "└─ " if level > 0 else ""
        indent = "   " * max(level - 1, 0) + bullet
        module_type = module.__class__.__name__
        tag = ""
        if isinstance(module, TiledLinearADC):
            tag = " 📊"
            if will_visualise(name):
                tag += "⭐"
        if tag or not list(module.children()):
            return f"{indent}{name or 'model'} ({module_type}){tag}"
        return None

    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("MODEL STRUCTURE WITH ADC HOOKS")
    output_lines.append("=" * 80)
    output_lines.append("Legend: 📊 = calibration hook,  ⭐ = visualization")
    output_lines.append("-" * 80)

    for name, module in model.named_modules():
        level = len(name.split(".")) if name else 0
        line = format_line(level, name, module)
        if line:
            output_lines.append(line)

    output_lines.append("=" * 80)
    output_text = "\n".join(output_lines)
    logger.info("\n" + output_text)
    return output_text


def load_dataset_by_name(dataset_name: str, split: str = "train"):
    """Load a dataset by name."""
    if dataset_name == "wikitext2":
        raw = load_dataset("wikitext", "wikitext-2-raw-v1")
        return raw[split]
    elif dataset_name == "c4":
        if split == "train":
            raw = load_dataset("allenai/c4", "en", split="train", streaming=True)
            return raw
        else:
            raw = load_dataset("allenai/c4", "en", split="validation", streaming=True)
            return raw
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: wikitext2, c4")


def load_and_tokenize_for_sliding_window(
    dataset_name: str,
    split: str,
    tokenizer,
    max_samples: int = 1000
):
    """
    Load and tokenize a dataset into one long sequence for sliding window evaluation.
    This is the standard approach used in papers like GPTQ, AWQ, FlatQuant.
    """
    logger.info(f"Loading {dataset_name} ({split} split) for sliding window evaluation...")

    if dataset_name == "wikitext2":
        raw = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        text = "\n\n".join([t for t in raw["text"] if t.strip()])
    elif dataset_name == "c4":
        raw = load_dataset("allenai/c4", "en", split=split, streaming=True)
        texts = []
        for i, example in enumerate(raw):
            if max_samples and i >= max_samples:
                break
            if example["text"].strip():
                texts.append(example["text"])
        text = "\n\n".join(texts)
        logger.info(f"Loaded {len(texts)} samples from C4")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    logger.info(f"Total text length: {len(text):,} characters")

    encodings = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )

    logger.info(f"Total tokens: {encodings['input_ids'].size(1):,}")
    return encodings


def compute_perplexity_sliding_window(
    model,
    encodings,
    device,
    max_length: int = 2048,
    stride: int = None,
    desc: str = "Evaluating"
):
    """Compute perplexity using sliding window approach (standard for papers)."""
    if stride is None:
        stride = max_length // 2

    model.eval()

    input_ids = encodings["input_ids"]
    seq_len = input_ids.size(1)

    logger.info(f"  Total tokens in corpus: {seq_len:,}")
    logger.info(f"  Context window: {max_length}, Stride: {stride}")

    nlls = []
    total_tokens = 0

    num_windows = max(1, (seq_len - max_length) // stride + 1)

    prev_end_loc = 0
    with torch.no_grad():
        for begin_loc in tqdm(range(0, seq_len, stride), desc=desc, total=num_windows):
            end_loc = min(begin_loc + max_length, seq_len)

            input_ids_window = input_ids[:, begin_loc:end_loc].to(device)
            target_len = end_loc - prev_end_loc

            labels = input_ids_window.clone()
            labels[:, :-target_len] = -100

            outputs = model(input_ids=input_ids_window, labels=labels)

            neg_log_likelihood = outputs.loss * target_len
            nlls.append(neg_log_likelihood.item())
            total_tokens += target_len

            prev_end_loc = end_loc
            if end_loc >= seq_len:
                break

    avg_loss = sum(nlls) / total_tokens
    perplexity = math.exp(avg_loss)

    return {
        "perplexity": perplexity,
        "avg_loss": avg_loss,
        "total_tokens": total_tokens,
        "num_windows": len(nlls),
        "max_length": max_length,
        "stride": stride,
    }


def prepare_dataset_for_lm(dataset, tokenizer, max_length: int, max_samples: int = None,
                           min_text_length: int = 50, streaming: bool = False):
    """Prepare a dataset for language modeling evaluation."""
    def tokenize_function(examples):
        texts = [t for t in examples["text"] if t.strip() and len(t.strip()) > min_text_length]
        if not texts:
            return {"input_ids": [], "attention_mask": []}
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )
        return tokenized

    if streaming:
        samples = []
        for i, example in enumerate(dataset):
            if max_samples and i >= max_samples:
                break
            if len(example["text"].strip()) > min_text_length:
                samples.append(example)
        from datasets import Dataset
        dataset = Dataset.from_list(samples)
    else:
        dataset = dataset.filter(lambda x: len(x["text"].strip()) > min_text_length)
        if max_samples and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))

    dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )
    dataset = dataset.filter(lambda x: len(x["input_ids"]) > 0)
    return dataset


# =========================================================================
# ADC Calibrator (identical to llama_adc_ptq.py)
# =========================================================================

class ADCCalibrator:
    """Calibrates ADC quantizers using activation statistics"""

    def __init__(self, model: nn.Module, method: str = "minmax", bx: int = 8, bw: int = 8):
        self.model = model
        self.method = method
        self.bx = bx
        self.bw = bw
        self.stats = {}
        self._current_attention_mask = None

    def register_hooks(self):
        """Register forward hooks to collect activation statistics."""
        hooks = []
        calibrator = self

        def make_hook(name):
            def hook(module, input, output):
                if name not in calibrator.stats:
                    calibrator.stats[name] = {
                        'act_min': [], 'act_max': [], 'act_absmax': [],
                        'w_min': [], 'w_max': [], 'w_absmax': [],
                        'y_int_min': [], 'y_int_max': [], 'y_int_absmax': [],
                        'module': module,
                    }

                x = input[0].detach()

                mask = calibrator._current_attention_mask
                if mask is not None:
                    if x.ndim == 3 and mask.ndim == 2 and mask.shape[0] == x.shape[0] and mask.shape[1] == x.shape[1]:
                        bool_mask = mask.bool()
                        x_valid = x[bool_mask]
                    elif x.ndim == 2 and mask.ndim == 2:
                        flat_mask = mask.reshape(-1).bool()
                        if flat_mask.shape[0] == x.shape[0]:
                            x_valid = x[flat_mask]
                        else:
                            x_valid = x
                    else:
                        x_valid = x
                else:
                    x_valid = x

                if x_valid.numel() == 0:
                    return

                calibrator.stats[name]['act_min'].append(x_valid.min().item())
                calibrator.stats[name]['act_max'].append(x_valid.max().item())
                calibrator.stats[name]['act_absmax'].append(x_valid.abs().max().item())

                w = module.weight.detach()
                calibrator.stats[name]['w_min'].append(w.min().item())
                calibrator.stats[name]['w_max'].append(w.max().item())
                calibrator.stats[name]['w_absmax'].append(w.abs().max().item())

                with torch.no_grad():
                    act_q = module.activation_quantizer
                    s_x = act_q.scale.to(x_valid.device)
                    if act_q.symmetric:
                        code_x = torch.clamp(torch.round(x_valid / s_x), act_q.qmin, act_q.qmax)
                    else:
                        zp_x = act_q.zero_point.to(x_valid.device)
                        code_x_temp = torch.clamp(torch.round(x_valid / s_x + zp_x), 0, act_q.qmax)
                        if hasattr(module, 'ashift') and module.ashift:
                            code_x = code_x_temp - module.C
                        else:
                            code_x = code_x_temp - zp_x

                    w_q = module.weight_quantizer
                    s_w_vec = w_q.scale.to(w.device)
                    s_w_b = s_w_vec.view(-1, 1)
                    code_w = torch.clamp(torch.round(w / s_w_b), w_q.qmin, w_q.qmax)

                    y_int = F.linear(code_x, code_w, bias=None)

                    calibrator.stats[name]['y_int_min'].append(y_int.min().item())
                    calibrator.stats[name]['y_int_max'].append(y_int.max().item())
                    calibrator.stats[name]['y_int_absmax'].append(y_int.abs().max().item())

            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, QATLinearADC):
                hook = module.register_forward_hook(make_hook(name))
                hooks.append(hook)
            elif isinstance(module, TiledLinearADC):
                for tile_idx, tile in enumerate(module.tiles):
                    tile_name = f"{name}.tiles.{tile_idx}"
                    hook = tile.register_forward_hook(make_hook(tile_name))
                    hooks.append(hook)

        logger.info(f"Registered {len(hooks)} calibration hooks")
        return hooks

    def calibrate(self, dataloader, num_batches: int = 100):
        """Run calibration on dataloader."""
        logger.info(f"Running calibration on {num_batches} batches...")

        self.model.eval()
        hooks = self.register_hooks()

        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, total=num_batches, desc="Calibrating")):
                if i >= num_batches:
                    break
                batch = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                self._current_attention_mask = batch.get("attention_mask", None)
                try:
                    _ = self.model(**batch)
                except Exception as e:
                    logger.warning(f"Error in batch {i}: {e}")
                    continue
                finally:
                    self._current_attention_mask = None

        for hook in hooks:
            hook.remove()
        logger.info(f"Collected stats for {len(self.stats)} layers")

    def compute_optimal_params(self, log_to_wandb: bool = False) -> dict[str, dict]:
        """Compute optimal quantization scales from collected statistics."""
        optimal_params = {}
        all_act_scales = []
        all_w_scales = []
        all_y_int_targets = []

        for name, stats in self.stats.items():
            if not stats['y_int_absmax']:
                continue

            act_absmax_arr = np.array(stats['act_absmax'])
            act_min_arr = np.array(stats['act_min'])
            act_max_arr = np.array(stats['act_max'])
            w_absmax_arr = np.array(stats['w_absmax'])
            y_int_absmax_arr = np.array(stats['y_int_absmax'])

            module = stats.get('module')
            if module and hasattr(module, 'activation_quantizer'):
                is_symmetric = module.activation_quantizer.symmetric
            else:
                is_symmetric = True

            if self.method == "minmax":
                act_absmax = act_absmax_arr.max()
                act_min_val = act_min_arr.min()
                act_max_val = act_max_arr.max()
                w_absmax = w_absmax_arr.max()
                y_int_target = y_int_absmax_arr.max()
            elif self.method == "percentile":
                act_absmax = np.percentile(act_absmax_arr, 99.9)
                act_min_val = np.percentile(act_min_arr, 0.1)
                act_max_val = np.percentile(act_max_arr, 99.9)
                w_absmax = np.percentile(w_absmax_arr, 99.9)
                y_int_target = np.percentile(y_int_absmax_arr, 99.9)
            elif self.method == "mse":
                act_absmax = self._find_mse_optimal_threshold(act_absmax_arr)
                act_min_val = act_min_arr.min()
                act_max_val = act_max_arr.max()
                w_absmax = self._find_mse_optimal_threshold(w_absmax_arr)
                y_int_target = self._find_mse_optimal_threshold(y_int_absmax_arr)
            else:
                act_absmax = act_absmax_arr.max()
                act_min_val = act_min_arr.min()
                act_max_val = act_max_arr.max()
                w_absmax = w_absmax_arr.max()
                y_int_target = y_int_absmax_arr.max()

            w_levels = 2 ** (self.bw - 1) - 1
            optimal_w_scale = max(w_absmax, 1e-8) / float(w_levels)

            if is_symmetric:
                act_levels = 2 ** (self.bx - 1) - 1
                optimal_act_scale = max(act_absmax, 1e-8) / float(act_levels)
                optimal_act_zp = 0.0
            else:
                act_range = max(act_max_val - act_min_val, 1e-8)
                act_qmax = 2 ** self.bx - 1
                optimal_act_scale = act_range / float(act_qmax)
                optimal_act_zp = float(np.clip(
                    np.round(-act_min_val / optimal_act_scale), 0, act_qmax
                ))

            optimal_params[name] = {
                'act_scale': optimal_act_scale,
                'act_zero_point': optimal_act_zp,
                'act_symmetric': is_symmetric,
                'w_scale': optimal_w_scale,
                'y_int_target': y_int_target,
            }

            all_act_scales.append(optimal_act_scale)
            all_w_scales.append(optimal_w_scale)
            all_y_int_targets.append(y_int_target)

        if log_to_wandb and wandb.run is not None:
            wandb.log({
                "calibration/num_layers": len(optimal_params),
                "calibration/act_scale_mean": np.mean(all_act_scales),
                "calibration/act_scale_std": np.std(all_act_scales),
                "calibration/act_scale_min": np.min(all_act_scales),
                "calibration/act_scale_max": np.max(all_act_scales),
                "calibration/w_scale_mean": np.mean(all_w_scales),
                "calibration/w_scale_std": np.std(all_w_scales),
                "calibration/w_scale_min": np.min(all_w_scales),
                "calibration/w_scale_max": np.max(all_w_scales),
                "calibration/y_int_target_mean": np.mean(all_y_int_targets),
                "calibration/y_int_target_std": np.std(all_y_int_targets),
                "calibration/y_int_target_max": np.max(all_y_int_targets),
            })

        return optimal_params

    def _find_mse_optimal_threshold(self, values: np.ndarray) -> float:
        """Find threshold that minimizes MSE"""
        candidates = np.percentile(values, [90, 95, 99, 99.5, 99.9, 100])
        best_mse = float('inf')
        best_threshold = candidates[-1]
        for threshold in candidates:
            clipped = np.clip(values, -threshold, threshold)
            mse = np.mean((values - clipped) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_threshold = threshold
        return best_threshold

    def apply_calibration(self, optimal_params: dict[str, dict]):
        """Apply calibrated scales to the model. Does NOT change delta (hardware constant)."""
        logger.info("Applying calibrated scales to model...")
        updated_act = 0
        updated_w = 0

        for name, module in self.model.named_modules():
            if isinstance(module, QATLinearADC) and name in optimal_params:
                params = optimal_params[name]
                with torch.no_grad():
                    if hasattr(module, 'activation_quantizer'):
                        act_q = module.activation_quantizer
                        old_scale = act_q.scale.item()
                        act_q.scale.copy_(torch.tensor(params['act_scale'], dtype=torch.float32))
                        act_q._scale_initialized = True

                        if not act_q.symmetric:
                            old_zp = act_q.zero_point.item()
                            act_q.zero_point.copy_(torch.tensor(params['act_zero_point'], dtype=torch.float32))
                            act_q._zp_initialized = True
                            logger.info(
                                f"{name} [ACT asym]: scale {old_scale:.6f} -> {params['act_scale']:.6f}, "
                                f"zp {old_zp:.2f} -> {params['act_zero_point']:.2f}"
                            )
                        else:
                            logger.info(f"{name} [ACT sym]: scale {old_scale:.6f} -> {params['act_scale']:.6f}")
                        updated_act += 1

                    if hasattr(module, 'weight_quantizer'):
                        w_q = module.weight_quantizer
                        old_scale_mean = w_q.scale.mean().item() if w_q.scale.numel() > 0 else 0.01

                        if w_q.per_channel:
                            weight = module.weight.detach()
                            if w_q.channel_dim == 0:
                                per_channel_absmax = weight.abs().max(dim=1)[0]
                            else:
                                weight_transposed = weight.transpose(w_q.channel_dim, 0)
                                per_channel_absmax = weight_transposed.contiguous().view(weight_transposed.shape[0], -1).abs().max(dim=1)[0]
                            per_channel_absmax = torch.clamp(per_channel_absmax, min=1e-6)
                            w_levels = 2 ** (self.bw - 1) - 1
                            new_scales = per_channel_absmax / float(w_levels)
                            if w_q.scale.numel() != new_scales.numel():
                                w_q.scale.data = w_q.scale.data.new_zeros(new_scales.shape)
                            w_q.scale.data.copy_(new_scales)
                            w_q._scale_initialized = True
                        else:
                            w_q.scale.copy_(torch.tensor(params['w_scale'], dtype=torch.float32))
                            w_q._scale_initialized = True

                        new_scale_mean = w_q.scale.mean().item()
                        logger.info(f"{name} [W sym]: scale {old_scale_mean:.6f} -> {new_scale_mean:.6f}")
                        updated_w += 1

        logger.info(f"Updated {updated_act} activation quantizers and {updated_w} weight quantizers")
        logger.info("NOTE: ADC delta values remain as hardware-defined constants")


# =========================================================================
# Diagnostics (identical to llama_adc_ptq.py)
# =========================================================================

def diagnose_quantized_model(model, tokenizer, device, num_layers_to_print: int = 5):
    """Run diagnostics on the quantized model to verify calibration health."""
    logger.info("=" * 80)
    logger.info("DIAGNOSTICS: Checking quantized model health")
    logger.info("=" * 80)

    act_scales = []
    w_scales_mean = []
    n_default_act = 0
    n_default_w = 0
    layer_info = []

    for name, module in model.named_modules():
        if isinstance(module, QATLinearADC):
            aq = module.activation_quantizer
            wq = module.weight_quantizer
            a_s = aq.scale.detach().float()
            w_s = wq.scale.detach().float()

            a_val = a_s.item() if a_s.numel() == 1 else a_s.mean().item()
            w_val = w_s.mean().item()
            act_scales.append(a_val)
            w_scales_mean.append(w_val)

            if abs(a_val - 0.01) < 1e-6 or abs(a_val - 0.02) < 1e-6:
                n_default_act += 1
            if w_s.numel() == 1 and abs(w_val - 0.01) < 1e-6:
                n_default_w += 1

            layer_info.append((name, a_val, w_val, w_s.shape, module.delta,
                               aq.symmetric, wq.per_channel))

    total = len(layer_info)
    logger.info(f"Total QATLinearADC layers: {total}")
    logger.info(f"Layers with DEFAULT act scale (likely uncalibrated): {n_default_act}/{total}")
    logger.info(f"Layers with DEFAULT w scale   (likely uncalibrated): {n_default_w}/{total}")

    if act_scales:
        logger.info(f"Activation scale range: [{min(act_scales):.6f}, {max(act_scales):.6f}]")
    if w_scales_mean:
        logger.info(f"Weight scale mean range: [{min(w_scales_mean):.6f}, {max(w_scales_mean):.6f}]")

    for name, a_s, w_s, w_shape, delta, sym, pc in layer_info[:num_layers_to_print]:
        logger.info(
            f"  {name}: act_s={a_s:.6f} ({'sym' if sym else 'asym'}), "
            f"w_s_mean={w_s:.6f} (shape={list(w_shape)}, pc={pc}), delta={delta:.2f}"
        )

    if n_default_act > 0 or n_default_w > 0:
        logger.warning(
            ">>> SOME LAYERS STILL HAVE DEFAULT SCALES! "
            "Calibration probably did not reach these layers."
        )

    logger.info("Running diagnostic forward pass...")
    text = "The quick brown fox jumps over the lazy dog. " * 5
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    has_nan = logits.isnan().any().item()
    has_inf = logits.isinf().any().item()
    logit_min = logits.min().item()
    logit_max = logits.max().item()
    logit_std = logits.float().std().item()

    logger.info(f"Logits shape: {list(logits.shape)}, dtype: {logits.dtype}")
    logger.info(f"Logits range: [{logit_min:.2f}, {logit_max:.2f}], std: {logit_std:.2f}")
    logger.info(f"Contains NaN: {has_nan}, Contains Inf: {has_inf}")

    if has_nan or has_inf:
        logger.error(">>> CRITICAL: Model produces NaN/Inf! Quantization is numerically broken.")
    elif logit_std < 0.01:
        logger.warning(">>> WARNING: Logit std is near zero -- model output is collapsed.")
    elif logit_std > 1000:
        logger.warning(">>> WARNING: Logit std is huge -- possible scale miscalibration.")
    else:
        logger.info("Logits look healthy (no NaN/Inf, reasonable std).")

    pred_ids = logits[0, -1, :].argmax().item()
    pred_token = tokenizer.decode([pred_ids])
    logger.info(f"Next-token prediction for test sentence: '{pred_token}' (id={pred_ids})")

    logger.info("=" * 80)
    return {
        "n_layers": total,
        "n_default_act": n_default_act,
        "n_default_w": n_default_w,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "logit_range": (logit_min, logit_max),
        "logit_std": logit_std,
    }


# =========================================================================
# Latency Measurement
# =========================================================================

def measure_latency(
    model: nn.Module,
    input_ids: torch.Tensor,
    n_warmup: int = 5,
    n_runs: int = 20,
) -> dict:
    """Measure forward-pass latency in milliseconds.

    Uses CUDA events on GPU (accurate, accounts for async kernel launch),
    falls back to ``time.perf_counter`` on CPU.

    Returns mean, std, p50, p95 latency in ms, plus tokens/sec throughput.
    """
    import time

    device = input_ids.device
    use_cuda = device.type == "cuda"
    seq_len = input_ids.shape[1]
    batch_size = input_ids.shape[0]
    n_tokens = batch_size * seq_len

    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(n_warmup):
            _ = model(input_ids=input_ids)
        if use_cuda:
            torch.cuda.synchronize(device)

        # Timed runs
        latencies_ms = []
        for _ in range(n_runs):
            if use_cuda:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model(input_ids=input_ids)
                end.record()
                torch.cuda.synchronize(device)
                latencies_ms.append(start.elapsed_time(end))
            else:
                t0 = time.perf_counter()
                _ = model(input_ids=input_ids)
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(latencies_ms)
    mean_ms = float(arr.mean())
    return {
        "latency_mean_ms":   mean_ms,
        "latency_std_ms":    float(arr.std()),
        "latency_p50_ms":    float(np.percentile(arr, 50)),
        "latency_p95_ms":    float(np.percentile(arr, 95)),
        "throughput_tok_s":  float(n_tokens / (mean_ms / 1000.0)),
        "batch_size":        batch_size,
        "seq_len":           seq_len,
    }


# =========================================================================
# ADC Layer Diagnostics
# =========================================================================

def collect_adc_layer_diagnostics(
    model: nn.Module,
    dataloader,
    num_batches: int,
    device: torch.device,
) -> dict:
    """Collect per-tile ADC quantization diagnostics over *num_batches* batches.

    Metrics per ``QATLinearADC`` tile
    ----------------------------------
    dead_rate            fraction of outputs where floor(y_int/Δ) == 0
                         (positive y_int < Δ → ADC sees nothing)
    clip_rate            fraction clipped by ADC clamp(na, pa)
    std_z                std of z = y_int / Δ  (1.0 = one ADC step on average)
    mean_z               mean of z
    bin_usage            fraction of 2^ba ADC bins actually used
    act_sat_rate         fraction of activation codes at ±qmax
    w_sat_rate           fraction of weight codes at ±qmax (static, from weights)
    reconstruction_mse   MSE(y_quantized, y_fp16) per tile element
    reconstruction_rel   reconstruction_mse / mean(y_fp16²)
    """
    tile_stats: dict = {}
    hooks: list = []

    def _make_hook(tile_name: str):
        def _hook(module, inp, output):
            x = inp[0].detach().float()
            w = module.weight.detach().float()

            # FP reference output (no quantization)
            with torch.no_grad():
                bias = module.bias.float() if module.bias is not None else None
                y_fp = F.linear(x, w, bias)

            # Recompute activation codes (per-token dynamic scale)
            act_q = module.activation_quantizer
            act_levels = float(act_q.qmax)
            s_x = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / act_levels
            code_x = torch.round(x / s_x).clamp(act_q.qmin, act_q.qmax)

            # Recompute weight codes
            w_q = module.weight_quantizer
            s_w = w_q.scale.detach().float().view(-1, 1)
            code_w = torch.round(w / s_w).clamp(w_q.qmin, w_q.qmax)

            # Integer MVM
            dev_type = "cuda" if x.is_cuda else "cpu"
            with torch.amp.autocast(dev_type, enabled=False):
                y_int = F.linear(code_x.float(), code_w.float())

            delta = float(module.delta)
            na, pa = int(module.na), int(module.pa)
            z = y_int / delta
            adc_codes = z.floor().long().clamp(na, pa)

            n = y_int.numel()
            dead = int((adc_codes == 0).sum())
            clipped = int(((z.floor() < na) | (z.floor() > pa)).sum())
            z_flat = z.reshape(-1)

            y_q = output.detach().float()
            diff = y_q - y_fp

            # Z-distribution buckets (by |z|)
            abs_z = z_flat.abs()
            z_b0 = int((abs_z < 1).sum())            # dead zone
            z_b1 = int(((abs_z >= 1)   & (abs_z < 10)).sum())
            z_b2 = int(((abs_z >= 10)  & (abs_z < 50)).sum())
            z_b3 = int(((abs_z >= 50)  & (abs_z < 128)).sum())
            z_b4 = int((abs_z >= 128).sum())          # clipping zone

            if tile_name not in tile_stats:
                tile_stats[tile_name] = {
                    "dead": 0, "clip": 0, "total": 0,
                    "z_sum": 0.0, "z_sq_sum": 0.0, "z_quad_sum": 0.0,
                    "diff_sq_sum": 0.0, "fp_sq_sum": 0.0,
                    "act_sat": 0, "act_n": 0,
                    "bins": set(),
                    "na": na, "pa": pa,
                    "zb0": 0, "zb1": 0, "zb2": 0, "zb3": 0, "zb4": 0,
                }
            s = tile_stats[tile_name]
            s["dead"] += dead
            s["clip"] += clipped
            s["total"] += n
            s["z_sum"] += float(z_flat.sum())
            s["z_sq_sum"] += float((z_flat ** 2).sum())
            s["z_quad_sum"] += float((z_flat ** 4).sum())
            s["diff_sq_sum"] += float((diff ** 2).sum())
            s["fp_sq_sum"] += float((y_fp ** 2).sum())
            s["act_sat"] += int((code_x.abs() == act_levels).sum())
            s["act_n"] += code_x.numel()
            s["bins"].update(adc_codes.unique().tolist())
            s["zb0"] += z_b0
            s["zb1"] += z_b1
            s["zb2"] += z_b2
            s["zb3"] += z_b3
            s["zb4"] += z_b4
        return _hook

    for name, module in model.named_modules():
        if isinstance(module, QATLinearADC):
            hooks.append(module.register_forward_hook(_make_hook(name)))

    # Weight saturation (static — computed once from current weights)
    w_sat: dict = {}
    for name, module in model.named_modules():
        if isinstance(module, QATLinearADC):
            w = module.weight.detach().float()
            s_w = module.weight_quantizer.scale.detach().float().view(-1, 1)
            code_w = torch.round(w / s_w).clamp(
                module.weight_quantizer.qmin, module.weight_quantizer.qmax
            )
            w_sat[name] = float((code_w.abs() == float(module.weight_quantizer.qmax)).float().mean())

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            try:
                model(**batch)
            except Exception as e:
                logger.warning(f"ADC diagnostics batch {i}: {e}")

    for h in hooks:
        h.remove()

    # Aggregate per-tile stats
    results: dict = {}
    for name, s in tile_stats.items():
        n = max(s["total"], 1)
        mean_z = s["z_sum"] / n
        var_z = max(s["z_sq_sum"] / n - mean_z ** 2, 0.0)
        n_bins = s["pa"] - s["na"] + 1
        fp_sq = max(s["fp_sq_sum"], 1e-10)
        results[name] = {
            "dead_rate":           s["dead"] / n,
            "clip_rate":           s["clip"] / n,
            "std_z":               var_z ** 0.5,
            "mean_z":              mean_z,
            "bin_usage":           len(s["bins"]) / n_bins,
            "act_sat_rate":        s["act_sat"] / max(s["act_n"], 1),
            "w_sat_rate":          w_sat.get(name, 0.0),
            "reconstruction_mse":  s["diff_sq_sum"] / n,
            "reconstruction_rel":  s["diff_sq_sum"] / fp_sq,
            # Z distribution buckets (fraction of values in each |z| range)
            "z_frac_dead":   s["zb0"] / n,               # |z| < 1   (ADC = 0)
            "z_frac_low":    s["zb1"] / n,               # 1 ≤ |z| < 10
            "z_frac_mid":    s["zb2"] / n,               # 10 ≤ |z| < 50
            "z_frac_high":   s["zb3"] / n,               # 50 ≤ |z| < 128
            "z_frac_clip":   s["zb4"] / n,               # |z| ≥ 128 (clipped)
            # Excess kurtosis of z: 0 = normal, >0 = heavy-tailed peak at 0
            "kurtosis_z":    (s["z_quad_sum"] / n) / max(var_z ** 2, 1e-10) - 3.0,
        }
    return results


def log_adc_diagnostics(results: dict, use_wandb: bool = False) -> None:
    """Log aggregated ADC diagnostics to logger and optionally to WandB."""
    if not results:
        logger.warning("ADC diagnostics: no results collected")
        return

    metrics = [
        "dead_rate", "clip_rate", "std_z", "mean_z",
        "bin_usage", "act_sat_rate", "w_sat_rate",
        "reconstruction_mse", "reconstruction_rel",
        "kurtosis_z",
        "z_frac_dead", "z_frac_low", "z_frac_mid", "z_frac_high", "z_frac_clip",
    ]

    agg: dict = {}
    for m in metrics:
        vals = [r[m] for r in results.values() if m in r]
        if not vals:
            continue
        agg[m] = {"mean": float(np.mean(vals)), "max": float(np.max(vals)),
                  "min": float(np.min(vals)), "vals": vals}

    logger.info("=" * 80)
    logger.info("ADC LAYER DIAGNOSTICS")
    logger.info("=" * 80)
    for m, a in agg.items():
        logger.info(f"  {m:25s}  mean={a['mean']:.4f}  max={a['max']:.4f}  min={a['min']:.4f}")

    # Z-distribution bucket summary (mean across all tiles)
    if "z_frac_dead" in agg:
        logger.info("  Z-distribution buckets (mean fraction across tiles):")
        logger.info(f"    |z|<1   (dead):    {agg['z_frac_dead']['mean']:.3f}  ← ADC outputs 0")
        logger.info(f"    1≤|z|<10 (low):   {agg['z_frac_low']['mean']:.3f}  ← coarse resolution")
        logger.info(f"    10≤|z|<50 (mid):  {agg['z_frac_mid']['mean']:.3f}")
        logger.info(f"    50≤|z|<128 (high):{agg['z_frac_high']['mean']:.3f}")
        logger.info(f"    |z|≥128 (clip):   {agg['z_frac_clip']['mean']:.3f}")
    if "kurtosis_z" in agg:
        k = agg["kurtosis_z"]["mean"]
        interp = "≈normal" if abs(k) < 1 else ("heavy-tailed peak at 0 → dead zones" if k > 0 else "flat")
        logger.info(f"  Kurtosis z:  mean={k:.2f}  ({interp})")
        logger.info(f"               max={agg['kurtosis_z']['max']:.2f}  min={agg['kurtosis_z']['min']:.2f}")

    # Worst 3 layers by dead_rate and reconstruction_rel
    for sort_key in ("dead_rate", "reconstruction_rel", "kurtosis_z"):
        ranked = sorted(results.items(), key=lambda x: x[1].get(sort_key, 0), reverse=True)[:3]
        logger.info(f"  Top-3 by {sort_key}:")
        for name, r in ranked:
            logger.info(f"    {name}: {r.get(sort_key, 0):.4f}")
    logger.info("=" * 80)

    if use_wandb and wandb.run is not None:
        log_dict: dict = {}
        for m, a in agg.items():
            log_dict[f"adc_diag/mean_{m}"] = a["mean"]
            log_dict[f"adc_diag/max_{m}"] = a["max"]
            log_dict[f"adc_diag/{m}_hist"] = wandb.Histogram(a["vals"])
        wandb.log(log_dict)


# =========================================================================
# KD Fine-tuning (post-ADC-calibration)
# =========================================================================

def perform_kd_finetuning(
    model: nn.Module,
    model_name_or_path: str,
    dataloader,
    device: torch.device,
    epochs: int = 5,
    lr: float = 1e-4,
    temperature: float = 2.0,
    num_batches: int = 64,
    teacher_on_cpu: bool = False,
    use_wandb: bool = False,
) -> None:
    """End-to-end KD fine-tuning of activation quantizer scales.

    Loads a FP16 teacher model and fine-tunes each QATLinearADC tile's
    activation_quantizer.scale in the student model using KL divergence
    on final logits (token-level, summed over the sequence).

    Only activation scales are trained — weight scales are left at their
    calibrated values to avoid disturbing the per-channel weight quantization.
    """
    logger.info("=" * 80)
    logger.info(f"KD FINE-TUNING: epochs={epochs}  lr={lr}  T={temperature}  batches={num_batches}")
    logger.info("=" * 80)

    # -----------------------------------------------------------------
    # 1. Load FP16 teacher
    # -----------------------------------------------------------------
    teacher_device = torch.device("cpu") if teacher_on_cpu else device
    logger.info(f"Loading FP teacher from {model_name_or_path} → {teacher_device}...")
    teacher = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(teacher_device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    logger.info("Teacher loaded and frozen.")

    # -----------------------------------------------------------------
    # 2. Collect trainable params: per-channel weight scales in QATLinearADC.
    #
    # NOTE: activation_quantizer.scale is intentionally NOT used in
    # QATLinearADC.forward() — activations are scaled dynamically per-token
    # (s_x = amax(x, dim=-1) / act_levels), so there is no learned
    # per-tensor activation scale to tune.
    #
    # weight_quantizer.scale IS used in forward via round_ste(W / s_w),
    # and gradients flow through STE. KD can shift these scales to
    # minimise logit KL rather than the calibration percentile criterion.
    # -----------------------------------------------------------------
    kd_params = []
    for _, m in model.named_modules():
        if isinstance(m, QATLinearADC):
            m.set_quantizer_mode('fixed')
            m.weight_quantizer.scale.requires_grad_(True)
            kd_params.append(m.weight_quantizer.scale)

    if not kd_params:
        logger.warning("KD: no QATLinearADC weight scales found — skipping.")
        del teacher
        torch.cuda.empty_cache()
        return

    logger.info(f"KD trainable params: {len(kd_params)} per-channel weight scales")
    optimizer = torch.optim.Adam(kd_params, lr=lr)

    # -----------------------------------------------------------------
    # 3. Training loop
    # -----------------------------------------------------------------
    model.eval()  # keep eval mode — avoids dropout, doesn't disable autograd

    best_loss = float("inf")
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches_done = 0

        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            # Teacher forward — no gradient, possibly on CPU
            with torch.no_grad():
                t_kwargs = {"input_ids": input_ids.to(teacher_device)}
                if attention_mask is not None:
                    t_kwargs["attention_mask"] = attention_mask.to(teacher_device)
                teacher_logits = teacher(**t_kwargs).logits.to(device).float()

            # Student forward — gradient flows through STE in activation scales
            student_out = model(input_ids=input_ids, attention_mask=attention_mask)
            student_logits = student_out.logits.float()  # [B, T, V]

            V = teacher_logits.shape[-1]

            # KL(teacher || student): minimise -sum(p_teacher * log(p_student))
            # scaled by T² so gradients are temperature-independent
            loss = F.kl_div(
                F.log_softmax(student_logits.view(-1, V) / temperature, dim=-1),
                F.softmax(teacher_logits.view(-1, V) / temperature, dim=-1),
                reduction="batchmean",
                log_target=False,
            ) * (temperature ** 2)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(kd_params, max_norm=1.0)
            optimizer.step()

            # Ensure weight scales stay positive
            with torch.no_grad():
                for p in kd_params:
                    p.clamp_(min=1e-6)

            epoch_loss += loss.item()
            n_batches_done += 1

        avg_loss = epoch_loss / max(n_batches_done, 1)
        best_loss = min(best_loss, avg_loss)
        logger.info(f"KD epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}")
        if use_wandb:
            wandb.log({"kd/loss": avg_loss, "kd/epoch": epoch + 1})

    # -----------------------------------------------------------------
    # 4. Restore fixed mode, free teacher
    # -----------------------------------------------------------------
    for _, m in model.named_modules():
        if isinstance(m, QATLinearADC):
            m.set_quantizer_mode('fixed')

    del teacher
    torch.cuda.empty_cache()

    logger.info(f"KD fine-tuning complete. Best loss={best_loss:.4f}.")
    if use_wandb:
        wandb.run.summary["kd/best_loss"] = best_loss


# =========================================================================
# SmoothQuant 3D Visualization
# =========================================================================

def _capture_activations_for_layer(model, sample_input, layer_name, device):
    """Run one forward pass and capture input activations for a specific nn.Linear layer.

    Padding positions (where attention_mask == 0) are stripped so that
    the returned tensor contains only real-token activations.
    """
    captured = {}

    def hook(module, input, output):
        captured['x'] = input[0].detach().cpu().float()

    target = None
    for name, module in model.named_modules():
        if name == layer_name and isinstance(module, nn.Linear):
            target = module
            break

    if target is None:
        return None

    h = target.register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        model(**{k: v.to(device) for k, v in sample_input.items()})
    h.remove()

    x = captured.get('x', None)
    if x is None:
        return None

    # Strip padding positions using attention_mask
    mask = sample_input.get('attention_mask', None)
    if mask is not None:
        mask_cpu = mask.cpu()
        if x.dim() == 3 and mask_cpu.dim() == 2:
            # x: [batch, seq_len, hidden], mask: [batch, seq_len]
            # Keep only real tokens (mask == 1) for the first batch element
            valid = mask_cpu[0].bool()  # [seq_len]
            x = x[:, valid, :]  # [1, num_real_tokens, hidden]

    return x


def _generate_smooth_quant_3d_visualization(
    act_original: torch.Tensor,
    act_smoothed: torch.Tensor,
    weight_original: torch.Tensor,
    weight_smoothed: torch.Tensor,
    layer_name: str,
    filepath: str,
    grid_size: int = 64,
):
    """
    Generate 2x2 grid of 3D surface plots showing SmoothQuant effect.

    Layout:
        Top-left:     Activation (Original)    — X=Channel, Y=Token, Z=|value|
        Top-right:    Activation (SmoothQuant)  — X=Channel, Y=Token, Z=|value|
        Bottom-left:  Weight (Original)         — X=In Channel, Y=Out Channel, Z=|value|
        Bottom-right: Weight (SmoothQuant)       — X=In Channel, Y=Out Channel, Z=|value|
    """
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f"SmoothQuant: {layer_name}", fontsize=14, fontweight='bold', y=0.98)

    def _subsample_2d(tensor, rows, cols):
        """Subsample a 2D tensor to at most (rows, cols)."""
        if tensor.shape[0] > rows:
            idx_r = torch.linspace(0, tensor.shape[0] - 1, rows).long()
            tensor = tensor[idx_r]
        if tensor.shape[1] > cols:
            idx_c = torch.linspace(0, tensor.shape[1] - 1, cols).long()
            tensor = tensor[:, idx_c]
        return tensor

    def _plot_surface(ax, data, title, xlabel, ylabel):
        data_np = data.numpy()
        rows, cols = data_np.shape
        X = np.arange(cols)
        Y = np.arange(rows)
        X, Y = np.meshgrid(X, Y)
        ax.plot_surface(X, Y, data_np, cmap='viridis', alpha=0.9,
                        edgecolor='none', rcount=100, ccount=100)
        ax.set_title(title, fontsize=10, pad=10)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_zlabel('Absolute Value', fontsize=8)
        ax.tick_params(labelsize=7)

    # --- Activations ---
    # act shape: [batch, seq_len, hidden_dim] -> take first batch, [seq_len, hidden_dim]
    if act_original.dim() == 3:
        act_orig_2d = act_original[0].abs()
        act_smooth_2d = act_smoothed[0].abs()
    else:
        act_orig_2d = act_original.abs()
        act_smooth_2d = act_smoothed.abs()

    act_orig_2d = _subsample_2d(act_orig_2d, grid_size, grid_size)
    act_smooth_2d = _subsample_2d(act_smooth_2d, grid_size, grid_size)

    # Use same Z scale for both activation plots
    act_zmax = max(act_orig_2d.max().item(), act_smooth_2d.max().item())

    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    _plot_surface(ax1, act_orig_2d, 'Activation (Original)', 'Channel', 'Token')
    ax1.set_zlim(0, act_zmax * 1.05)

    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    _plot_surface(ax2, act_smooth_2d, 'Activation (SmoothQuant)', 'Channel', 'Token')
    ax2.set_zlim(0, act_zmax * 1.05)

    # --- Weights ---
    # weight shape: [out_features, in_features]
    w_orig_2d = weight_original.abs()
    w_smooth_2d = weight_smoothed.abs()

    w_orig_2d = _subsample_2d(w_orig_2d, grid_size, grid_size)
    w_smooth_2d = _subsample_2d(w_smooth_2d, grid_size, grid_size)

    w_zmax = max(w_orig_2d.max().item(), w_smooth_2d.max().item())

    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    _plot_surface(ax3, w_orig_2d, 'Weight (Original)', 'In Channel', 'Out Channel')
    ax3.set_zlim(0, w_zmax * 1.05)

    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    _plot_surface(ax4, w_smooth_2d, 'Weight (SmoothQuant)', 'In Channel', 'Out Channel')
    ax4.set_zlim(0, w_zmax * 1.05)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved SmoothQuant 3D visualization: {filepath}")
    return filepath


# =========================================================================
# ADC Visualization (identical to llama_adc_ptq.py)
# =========================================================================

def _generate_adc_visualizations(model, sample_input, layer_patterns, title_prefix="", output_subdir="./viz"):
    """
    Generate 3x4 grid visualizations for ADC layers (12 plots per layer).

    Layout matches the QAT reference style::

        Row 1: Raw Activation X | Activation Codes | Dequant Activation | Act Quant Error
        Row 2: Raw Weights W    | Weight Codes     | Dequant Weights    | W Quant Error
        Row 3: FP Output        | Quantized Output | FP vs Quant scatter| Output Error
    """
    os.makedirs(output_subdir, exist_ok=True)
    logger.info(f"Generating ADC visualizations: {title_prefix}")

    layers_to_viz = []
    for name, module in model.named_modules():
        if isinstance(module, QATLinearADC):
            if any(pattern in name for pattern in layer_patterns):
                layers_to_viz.append((name, module))
        elif isinstance(module, TiledLinearADC) and len(module.tiles) > 0:
            if any(pattern in name for pattern in layer_patterns):
                layers_to_viz.append((name + ".tiles.0", module.tiles[0]))

    if not layers_to_viz:
        logger.warning(f"No ADC layers found matching patterns: {layer_patterns}")
        return {}

    logger.info(f"Found {len(layers_to_viz)} layers to visualize")

    captured_data: dict = {}

    def make_hook(layer_name):
        def hook(module, input, output):
            with torch.no_grad():
                x = input[0].float()
                w = module.weight.float()
                bias = module.bias.float() if module.bias is not None else None

                act_q = module.activation_quantizer
                w_q = module.weight_quantizer

                # Per-token dynamic scale — mirrors QATLinearADC.forward() exactly.
                act_levels = float(act_q.qmax)  # 127 (signed) or 255 (unsigned)
                s_x = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / act_levels
                code_x = torch.clamp(torch.round(x / s_x), act_q.qmin, act_q.qmax)
                x_dequant = code_x * s_x

                s_w_vec = w_q.scale.to(w.device)
                s_w_b = s_w_vec.view(-1, 1)
                code_w = torch.clamp(torch.round(w / s_w_b), w_q.qmin, w_q.qmax)
                w_dequant = code_w * s_w_b

                y_fp = F.linear(x, w, bias)
                y_quant = output.float()

                y_int = F.linear(code_x, code_w, bias=None)
                delta = module.delta
                na, pa = module.na, module.pa
                y_adc_codes = torch.clamp(torch.floor(y_int / delta), na, pa)

                captured_data[layer_name] = {
                    'x_raw': x.detach().cpu().numpy(),
                    'code_x': code_x.detach().cpu().numpy(),
                    'x_dequant': x_dequant.detach().cpu().numpy(),
                    'w_raw': w.detach().cpu().numpy(),
                    'code_w': code_w.detach().cpu().numpy(),
                    'w_dequant': w_dequant.detach().cpu().numpy(),
                    'y_fp': y_fp.detach().cpu().numpy(),
                    'y_quant': y_quant.detach().cpu().numpy(),
                    'y_int_before_adc': y_int.detach().cpu().numpy(),
                    'y_adc_codes': y_adc_codes.detach().cpu().numpy(),
                    's_x': s_x.detach().cpu().mean().item(),  # mean per-token scale for viz
                    's_w': s_w_vec.detach().cpu().numpy(),
                    'act_qmin': act_q.qmin,
                    'act_qmax': act_q.qmax,
                    'w_qmin': w_q.qmin,
                    'w_qmax': w_q.qmax,
                    'delta': delta,
                    'na': na,
                    'pa': pa,
                }
        return hook

    hooks = []
    for name, module in layers_to_viz:
        hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        _ = model(**sample_input)

    for h in hooks:
        h.remove()

    result_paths = {}
    for name, _ in layers_to_viz:
        if name not in captured_data:
            continue
        try:
            clean_name = name.replace(".", "_").replace("/", "_")
            filename = f"{clean_name}_{title_prefix.replace(' ', '_')}.png"
            filepath = os.path.join(output_subdir, filename)
            _plot_adc_pipeline(captured_data[name], name, title_prefix, filepath)
            result_paths[clean_name] = filepath
            logger.info(f"  saved {name}")
        except Exception as e:
            logger.error(f"  failed {name}: {e}")
            import traceback
            traceback.print_exc()

    logger.info(f"Generated {len(result_paths)} visualizations in {output_subdir}")
    return result_paths


def _plot_adc_pipeline(data: dict, layer_name: str, title_prefix: str, filepath: str):
    """
    3x4 grid visualization matching the QAT reference style.

    Row 1: Raw Activation X | Activation Codes | Dequant Activation | Act Quant Error
    Row 2: Raw Weights W    | Weight Codes     | Dequant Weights    | W Quant Error
    Row 3: FP Output        | Quantized Output | FP vs Quant scatter| Output Error
    """
    N = 2000

    x_raw = data['x_raw'].flatten()[:N]
    code_x = data['code_x'].flatten()[:N]
    x_dequant = data['x_dequant'].flatten()[:N]
    x_err = x_raw - x_dequant

    w_raw = data['w_raw'].flatten()[:N]
    code_w = data['code_w'].flatten()[:N]
    w_dequant = data['w_dequant'].flatten()[:N]
    w_err = w_raw - w_dequant

    y_fp = data['y_fp'].flatten()[:N]
    y_quant = data['y_quant'].flatten()[:N]
    y_err = y_fp - y_quant

    act_qmin, act_qmax = data['act_qmin'], data['act_qmax']
    w_qmin, w_qmax = data['w_qmin'], data['w_qmax']
    s_x = data['s_x']
    s_w = data['s_w']

    bins = 50

    fig, axes = plt.subplots(3, 4, figsize=(22, 14))
    fig.suptitle(f"ADC Layer: {layer_name}", fontsize=14, fontweight='bold', y=0.995)

    # Row 1: Activations
    ax = axes[0, 0]
    ax.hist(x_raw, bins=bins, alpha=0.8, color='royalblue', edgecolor='black', linewidth=0.3)
    ax.set_title(f"1. Raw Activation X\nMean: {x_raw.mean():.4f}, Std: {x_raw.std():.4f}", fontsize=9)
    ax.set_xlabel('Value'); ax.set_ylabel('Count')

    ax = axes[0, 1]
    act_range_str = f"[{act_qmin}, {act_qmax}]"
    n_act_bits = int(np.log2(act_qmax - act_qmin + 1)) if (act_qmax - act_qmin + 1) > 0 else 0
    ax.hist(code_x, bins=bins, alpha=0.8, color='goldenrod', edgecolor='black', linewidth=0.3)
    ax.axvline(act_qmin, color='red', ls='--', lw=1.5, label='qmin')
    ax.axvline(act_qmax, color='red', ls='--', lw=1.5, label='qmax')
    ax.set_title(f"2. Activation Codes ({n_act_bits}-bit)\nRange {act_range_str}", fontsize=9)
    ax.set_xlabel('Code Value'); ax.set_ylabel('Count')
    ax.legend(fontsize=7)

    ax = axes[0, 2]
    ax.hist(x_dequant, bins=bins, alpha=0.8, color='darkcyan', edgecolor='black', linewidth=0.3)
    ax.set_title(f"3. Dequantized Activation\nMean: {x_dequant.mean():.4f}", fontsize=9)
    ax.set_xlabel('Value'); ax.set_ylabel('Count')

    ax = axes[0, 3]
    act_mae = float(np.mean(np.abs(x_err)))
    ax.hist(x_err, bins=bins, alpha=0.8, color='firebrick', edgecolor='black', linewidth=0.3)
    ax.set_title(f"4. Activation Quant Error\nMAE: {act_mae:.4f}", fontsize=9)
    ax.set_xlabel('Error'); ax.set_ylabel('Count')

    # Row 2: Weights
    ax = axes[1, 0]
    ax.hist(w_raw, bins=bins, alpha=0.8, color='mediumorchid', edgecolor='black', linewidth=0.3)
    ax.set_title(f"5. Raw Weights W\nMean: {w_raw.mean():.4f}, Std: {w_raw.std():.4f}", fontsize=9)
    ax.set_xlabel('Value'); ax.set_ylabel('Count')

    ax = axes[1, 1]
    w_range_str = f"[{w_qmin}, {w_qmax}]"
    n_w_bits = int(np.log2(w_qmax - w_qmin + 1)) if (w_qmax - w_qmin + 1) > 0 else 0
    ax.hist(code_w, bins=bins, alpha=0.8, color='orange', edgecolor='black', linewidth=0.3)
    ax.axvline(w_qmin, color='red', ls='--', lw=1.5, label='qmin')
    ax.axvline(w_qmax, color='red', ls='--', lw=1.5, label='qmax')
    ax.set_title(f"6. Weight Codes ({n_w_bits}-bit)\nRange {w_range_str}", fontsize=9)
    ax.set_xlabel('Code Value'); ax.set_ylabel('Count')
    ax.legend(fontsize=7)

    ax = axes[1, 2]
    ax.hist(w_dequant, bins=bins, alpha=0.8, color='darkcyan', edgecolor='black', linewidth=0.3)
    ax.set_title(f"7. Dequantized Weights\nMean: {w_dequant.mean():.4f}", fontsize=9)
    ax.set_xlabel('Value'); ax.set_ylabel('Count')

    ax = axes[1, 3]
    w_mae = float(np.mean(np.abs(w_err)))
    ax.hist(w_err, bins=bins, alpha=0.8, color='firebrick', edgecolor='black', linewidth=0.3)
    ax.set_title(f"8. Weight Quant Error\nMAE: {w_mae:.4f}", fontsize=9)
    ax.set_xlabel('Error'); ax.set_ylabel('Count')

    # Row 3: Output
    ax = axes[2, 0]
    ax.hist(y_fp, bins=bins, alpha=0.8, color='royalblue', edgecolor='black', linewidth=0.3)
    ax.set_title(f"9. FP Output\nMean: {y_fp.mean():.4f}, Std: {y_fp.std():.4f}", fontsize=9)
    ax.set_xlabel('Value'); ax.set_ylabel('Count')

    ax = axes[2, 1]
    ax.hist(y_quant, bins=bins, alpha=0.8, color='orange', edgecolor='black', linewidth=0.3)
    # ADC saturation limits in dequantized-output space:
    #   limit = adc_code * delta * s_x * mean(s_w)
    delta = data['delta']
    na, pa = data['na'], data['pa']
    s_w_mean = float(np.mean(np.abs(s_w)))
    adc_min_out = na * delta * s_x * s_w_mean
    adc_max_out = pa * delta * s_x * s_w_mean
    ax.axvline(adc_min_out, color='red', ls='--', lw=1.5, label=f'ADC min ({na})')
    ax.axvline(adc_max_out, color='red', ls='--', lw=1.5, label=f'ADC max ({pa})')
    ax.set_title(f"10. Quantized Output\nMean: {y_quant.mean():.4f}", fontsize=9)
    ax.set_xlabel('Value'); ax.set_ylabel('Count')
    ax.legend(fontsize=7)

    ax = axes[2, 2]
    subsample = min(len(y_fp), 500)
    idx = np.random.choice(len(y_fp), subsample, replace=False)
    ax.scatter(y_fp[idx], y_quant[idx], alpha=0.4, s=6, color='steelblue')
    lims = [min(y_fp[idx].min(), y_quant[idx].min()),
            max(y_fp[idx].max(), y_quant[idx].max())]
    ax.plot(lims, lims, 'r--', lw=1.5, label='y=x')
    ax.set_title("11. FP vs Quantized", fontsize=9)
    ax.set_xlabel('Full Precision'); ax.set_ylabel('Quantized')
    ax.legend(fontsize=7)

    ax = axes[2, 3]
    out_mse = float(np.mean(y_err ** 2))
    ax.hist(y_err, bins=bins, alpha=0.8, color='firebrick', edgecolor='black', linewidth=0.3)
    ax.set_title(f"12. Output Error\nMSE: {out_mse:.6f}", fontsize=9)
    ax.set_xlabel('Error (FP - Quant)'); ax.set_ylabel('Count')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return filepath


# =========================================================================
# Layer-by-layer ablation helper
# =========================================================================

def _run_layer_ablation(model, tokenizer, device, args, use_wandb):
    """
    For each decoder layer i: bypass ADC in that layer → eval perplexity → restore.
    High perplexity improvement when bypassing layer i means that layer is the bottleneck.
    """
    enc = load_and_tokenize_for_sliding_window(
        args.eval_datasets[0], args.eval_split, tokenizer,
    )
    _eff_stride = args.stride if args.stride is not None else args.max_length // 2
    _max_tokens = args.layer_ablation_max_windows * _eff_stride + args.max_length
    if enc["input_ids"].size(1) > _max_tokens:
        enc = {
            "input_ids": enc["input_ids"][:, :_max_tokens],
            "attention_mask": enc["attention_mask"][:, :_max_tokens],
        }

    n_layers = model.config.num_hidden_layers
    logger.info("=" * 80)
    logger.info(f"LAYER ABLATION: bypass ADC one decoder layer at a time ({n_layers} layers, "
                f"~{args.layer_ablation_max_windows} windows each)")
    logger.info("Interpretation: lower PPL when layer i bypassed → that layer hurts most in ADC mode")
    logger.info("=" * 80)

    results = {}
    for i in range(n_layers):
        pattern = f"layers.{i}."
        bypassed = []
        for name, m in model.named_modules():
            if isinstance(m, TiledLinearADC) and pattern in name:
                m.set_bypass_all(True)
                bypassed.append(name)

        metrics = compute_perplexity_sliding_window(
            model, enc, device,
            max_length=args.max_length, stride=args.stride,
            desc=f"Ablation layer {i:2d}/{n_layers-1}",
        )
        ppl = metrics["perplexity"]
        results[i] = ppl
        logger.info(f"  Layer {i:2d} bypassed → PPL = {ppl:.4f}  ({len(bypassed)} tiles bypassed)")

        if use_wandb:
            wandb.log({
                f"layer_ablation/layer_{i:02d}/perplexity": ppl,
                f"layer_ablation/layer_{i:02d}/n_bypassed_tiles": len(bypassed),
            })

        for name, m in model.named_modules():
            if isinstance(m, TiledLinearADC) and pattern in name:
                m.set_bypass_all(False)

    best = min(results, key=lambda x: results[x])
    logger.info("=" * 80)
    logger.info("LAYER ABLATION SUMMARY")
    logger.info("  Sorted by PPL when bypassed (lower = bigger improvement = worse ADC layer):")
    for i in sorted(results, key=lambda x: results[x]):
        improvement = 0.0  # baseline not tracked here, just show PPL
        logger.info(f"  Layer {i:2d}: PPL = {results[i]:.4f} when bypassed")
    logger.info(f"  → Worst ADC layer: layer {best} "
                f"(PPL drops to {results[best]:.4f} when bypassed = biggest improvement)")
    logger.info("=" * 80)

    if use_wandb:
        wandb.log({"layer_ablation/best_layer": best,
                   "layer_ablation/best_layer_ppl": results[best]})
    return results


# =========================================================================
# Projection-type ablation helper
# =========================================================================

_PROJ_TYPES = ["q_proj", "k_proj", "v_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj"]

def _run_proj_ablation(model, tokenizer, device, args, use_wandb):
    """
    For each projection type (q_proj, k_proj, …, down_proj): bypass ADC in
    ALL tiles of that type across all layers → eval perplexity → restore.
    Shows which projection type contributes most to ADC error.
    """
    enc = load_and_tokenize_for_sliding_window(
        args.eval_datasets[0], args.eval_split, tokenizer,
    )
    _eff_stride = args.stride if args.stride is not None else args.max_length // 2
    _max_tokens = args.layer_ablation_max_windows * _eff_stride + args.max_length
    if enc["input_ids"].size(1) > _max_tokens:
        enc = {
            "input_ids": enc["input_ids"][:, :_max_tokens],
            "attention_mask": enc["attention_mask"][:, :_max_tokens],
        }

    logger.info("=" * 80)
    logger.info("PROJECTION ABLATION: bypass ADC by projection type across all layers")
    logger.info("Interpretation: lower PPL when proj X bypassed → that proj type hurts most in ADC mode")
    logger.info("=" * 80)

    results = {}
    for proj in _PROJ_TYPES:
        bypassed = []
        for name, m in model.named_modules():
            if isinstance(m, TiledLinearADC) and f".{proj}." in name:
                m.set_bypass_all(True)
                bypassed.append(name)

        if not bypassed:
            logger.info(f"  {proj:12s}: no tiles found, skipping")
            continue

        metrics = compute_perplexity_sliding_window(
            model, enc, device,
            max_length=args.max_length, stride=args.stride,
            desc=f"Proj ablation [{proj}]",
        )
        ppl = metrics["perplexity"]
        results[proj] = ppl
        logger.info(f"  {proj:12s} bypassed ({len(bypassed):3d} tiles) → PPL = {ppl:.4f}")

        if use_wandb:
            wandb.log({f"proj_ablation/{proj}/perplexity": ppl,
                       f"proj_ablation/{proj}/n_tiles": len(bypassed)})

        for name, m in model.named_modules():
            if isinstance(m, TiledLinearADC) and f".{proj}." in name:
                m.set_bypass_all(False)

    if results:
        best = min(results, key=lambda x: results[x])
        logger.info("=" * 80)
        logger.info("PROJECTION ABLATION SUMMARY")
        logger.info("  Sorted by PPL when bypassed (lower = bigger improvement = worse ADC proj):")
        for p in sorted(results, key=lambda x: results[x]):
            logger.info(f"  {p:12s}: PPL = {results[p]:.4f} when bypassed")
        logger.info(f"  → Worst ADC proj: {best} "
                    f"(PPL drops to {results[best]:.4f} when bypassed = biggest improvement)")
        logger.info("=" * 80)
        if use_wandb:
            wandb.log({"proj_ablation/best_proj": best,
                       "proj_ablation/best_proj_ppl": results[best]})
    return results


# =========================================================================
# Stage-by-stage diagnostic helper
# =========================================================================

def _run_stage_eval(model, encodings, device, stage_name, args, use_wandb):
    """Run a quick perplexity eval at one pipeline stage and log to WandB."""
    logger.info("=" * 80)
    logger.info(f"STAGE EVAL [{stage_name}]")
    logger.info("=" * 80)
    metrics = compute_perplexity_sliding_window(
        model, encodings, device,
        max_length=args.max_length, stride=args.stride,
        desc=f"[{stage_name}]",
    )
    logger.info(
        f"  [{stage_name}] Perplexity: {metrics['perplexity']:.4f}  "
        f"(Loss: {metrics['avg_loss']:.4f}, Windows: {metrics['num_windows']})"
    )
    if use_wandb:
        wandb.log({
            f"stage_eval/{stage_name}/perplexity": metrics["perplexity"],
            f"stage_eval/{stage_name}/avg_loss": metrics["avg_loss"],
            f"stage_eval/{stage_name}/num_windows": metrics["num_windows"],
        })
    logger.info("=" * 80)
    return metrics


# =========================================================================
# Main pipeline
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PTQ for ADC-based LLaMA models with configurable preprocessing"
    )

    # Model settings
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B",
                        help="HuggingFace model name")
    parser.add_argument("--output_dir", type=str, default="./outputs_llama_smooth_quant_adc_ptq",
                        help="Where to save calibrated model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch_dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Model dtype for loading")

    # SmoothQuant settings
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="SmoothQuant migration strength (0=all on weights, 1=all on activations)")
    parser.add_argument("--smooth_quant_batches", type=int, default=64,
                        help="Number of calibration batches for SmoothQuant activation stats")
    parser.add_argument("--disable_smooth_quant", action="store_true",
                        help="Skip SmoothQuant preprocessing (for A/B comparison)")
    parser.add_argument("--smooth_quant_layers", type=str, nargs="+", default=[],
                        help="Layer patterns for SmoothQuant 3D visualization "
                             "(e.g. layers.0.self_attn.q_proj layers.0.mlp.gate_proj)")
    parser.add_argument("--preprocess_method", type=str, default="smooth_quant",
                        choices=["smooth_quant", "flat_quant", "none"],
                        help="Preprocess before ADC conversion")

    # FlatQuant settings
    parser.add_argument("--fq_w_bits", type=int, default=8,
                        help="FlatQuant internal weight quantizer bits during calibration")
    parser.add_argument("--fq_a_bits", type=int, default=8,
                        help="FlatQuant internal activation quantizer bits during calibration")
    parser.add_argument("--fq_nsamples", type=int, default=128,
                        help="Number of calibration samples for FlatQuant")
    parser.add_argument("--fq_cali_bsz", type=int, default=4,
                        help="Batch size for FlatQuant layer-by-layer calibration")
    parser.add_argument("--fq_epochs", type=int, default=15,
                        help="Training epochs per layer for FlatQuant")
    parser.add_argument("--fq_lr", type=float, default=5e-3,
                        help="Learning rate for FlatQuant transform training")
    parser.add_argument("--fq_diag_alpha", type=float, default=0.5,
                        help="Diagonal scale initialization alpha (SmoothQuant-style)")
    parser.add_argument("--fq_add_diag", action="store_true", default=True,
                        help="Enable per-channel diagonal scaling in transforms")
    parser.add_argument("--fq_no_diag", dest="fq_add_diag", action="store_false",
                        help="Disable per-channel diagonal scaling")
    parser.add_argument("--fq_lwc", action="store_true", default=True,
                        help="Enable learnable weight clipping")
    parser.add_argument("--fq_no_lwc", dest="fq_lwc", action="store_false",
                        help="Disable learnable weight clipping")
    parser.add_argument("--fq_lac", action="store_true", default=True,
                        help="Enable learnable activation clipping")
    parser.add_argument("--fq_no_lac", dest="fq_lac", action="store_false",
                        help="Disable learnable activation clipping")
    parser.add_argument("--pact_inference", action="store_true",
                        help="Apply PACT (post-training alpha calibration) at ADC inference. "
                             "Default: per-token amax at inference (same as training).")
    parser.add_argument("--fq_lambda_clip", type=float, default=0.0,
                        help="Penalty weight for ADC clip loss (E7/E9)")
    parser.add_argument("--fq_lambda_dead", type=float, default=0.0,
                        help="Penalty weight for ADC dead-zone loss (E8/E9)")
    parser.add_argument("--fq_clip_margin", type=float, default=1.0,
                        help="Clip margin m: penalise |z| > pa - m")
    parser.add_argument("--fq_dead_threshold", type=float, default=1.0,
                        help="Dead-zone threshold τ: penalise |z| < τ")
    parser.add_argument("--fq_penalty_projections", nargs="+",
                        default=["o_proj", "down_proj"],
                        help="Projection names to apply ADC penalties to")
    parser.add_argument("--fq_lambda_band", type=float, default=0.0,
                        help="Band-occupancy loss weight: encourages z into (tau_lo, tau_hi)")
    parser.add_argument("--fq_band_tau_lo", type=float, default=1.0,
                        help="Lower bound of useful ADC band (dead zone boundary)")
    parser.add_argument("--fq_band_tau_hi", type=float, default=64.0,
                        help="Upper bound of useful ADC band (clip zone boundary)")
    parser.add_argument("--fq_band_beta", type=float, default=5.0,
                        help="Sharpness of sigmoid transitions in band-occupancy loss")
    parser.add_argument("--fq_band_topk_frac", type=float, default=0.2,
                        help="Fraction of worst tiles used for band-occupancy loss (top-k%%, 0.2=20%%)")
    parser.add_argument("--fq_freeze_clip", action="store_true",
                        help="Freeze LWC/LAC clip params during training (prevents dead penalty escaping into weight clipping)")
    parser.add_argument("--fq_kronecker_init", type=str, default="random",
                        choices=["random", "hadamard"],
                        help="Kronecker factor initialization: 'random' (default FlatQuant) or 'hadamard' (QuaRot-style)")
    parser.add_argument("--fq_loss_type", type=str, default="mse",
                        choices=["mse", "l1", "huber"],
                        help="Reconstruction loss for FlatQuant calibration: mse (default), l1, or huber")
    parser.add_argument("--fq_huber_delta", type=float, default=1.0,
                        help="Delta parameter for HuberLoss (only used when --fq_loss_type=huber)")
    parser.add_argument("--fq_lambda_center", type=float, default=0.0,
                        help="Bin-center penalty weight: cos²(π·z) pushes y_int toward ADC bin centres")
    parser.add_argument("--fq_propagate_quant", action="store_true",
                        help="Propagated calibration: train each layer on ADC-quantized inputs from previous layers")
    parser.add_argument("--fq_prop_alpha", type=float, default=1.0,
                        help="Propagation mixing: 1.0=full quant (default), 0.5=dual-forward (FP+quant), 0.0=FP only")
    parser.add_argument("--fq_stage_b_epochs", type=int, default=0,
                        help="Stage B epochs: after main calibration, run a second pass with propagation (0=disabled)")
    parser.add_argument("--fq_stage_b_prop_alpha", type=float, default=0.5,
                        help="Stage B propagation alpha (default 0.5)")
    parser.add_argument("--fq_prop_alpha_early", type=float, default=None,
                        help="Propagation alpha for early layers (0..fq_prop_late_start-1). If None, fq_prop_alpha used for all.")
    parser.add_argument("--fq_prop_late_start", type=int, default=8,
                        help="Layer index where 'late' alpha starts (default 8, halfway for 16-layer Llama).")
    parser.add_argument("--fq_diag_attn", action="store_true",
                        help="Train diag_scale only for attention blocks (requires --fq_add_diag). If neither --fq_diag_attn nor --fq_diag_mlp is set, both are trained.")
    parser.add_argument("--fq_diag_mlp", action="store_true",
                        help="Train diag_scale only for MLP blocks (requires --fq_add_diag). If neither --fq_diag_attn nor --fq_diag_mlp is set, both are trained.")
    parser.add_argument("--fq_diag_mlp_up", action="store_true",
                        help="Within MLP diag: train only up_gate_trans.diag_scale. If neither --fq_diag_mlp_up nor --fq_diag_mlp_down is set, both are trained.")
    parser.add_argument("--fq_diag_mlp_down", action="store_true",
                        help="Within MLP diag: train only down_trans.diag_scale. If neither --fq_diag_mlp_up nor --fq_diag_mlp_down is set, both are trained.")
    parser.add_argument("--fq_stage_b_diag_attn", action="store_true",
                        help="Stage B: train diag_scale for attention blocks (overrides Stage A diag_attn setting).")
    parser.add_argument("--fq_stage_b_diag_mlp", action="store_true",
                        help="Stage B: train diag_scale for MLP blocks (overrides Stage A diag_mlp setting).")
    parser.add_argument("--fq_stochastic_prop", action="store_true",
                        help="Use stochastic propagation instead of fixed alpha: per-batch random alpha.")
    parser.add_argument("--fq_stochastic_mode", type=str, default="bernoulli",
                        choices=["bernoulli", "beta"],
                        help="Stochastic mode: 'bernoulli' (randomly pick fp or quant, single forward) or 'beta' (sample alpha~Beta(β,β), dual forward).")
    parser.add_argument("--fq_beta_param", type=float, default=2.0,
                        help="Beta distribution parameter β for stochastic_mode=beta. β=1 → uniform[0,1]; β=2 → concentrated near 0.5.")
    # Knowledge Distillation fine-tuning (post-ADC-calibration)
    parser.add_argument("--kd_epochs", type=int, default=0,
                        help="KD fine-tuning epochs after ADC calibration (0 = disabled)")
    parser.add_argument("--kd_lr", type=float, default=1e-4,
                        help="Learning rate for KD activation scale fine-tuning")
    parser.add_argument("--kd_temperature", type=float, default=2.0,
                        help="Softmax temperature T for KD loss (higher = softer targets)")
    parser.add_argument("--kd_batches", type=int, default=64,
                        help="Number of calibration batches per KD epoch")
    parser.add_argument("--kd_teacher_on_cpu", action="store_true",
                        help="Load FP teacher model on CPU to save GPU memory")
    # ADC-LoRA post-correction (post-ADC-calibration)
    parser.add_argument("--lora_rank", type=int, default=0,
                        help="LoRA rank for ADC-LoRA post-correction (0 = disabled)")
    parser.add_argument("--lora_alpha", type=float, default=8.0,
                        help="LoRA scaling alpha (scaling = lora_alpha / lora_rank)")
    parser.add_argument("--lora_target_modules", nargs="+", default=["down_proj"],
                        help="Projection names to apply LoRA to, e.g. down_proj o_proj")
    parser.add_argument("--lora_epochs", type=int, default=5,
                        help="Training epochs for LoRA calibration")
    parser.add_argument("--lora_lr", type=float, default=1e-4,
                        help="Learning rate for LoRA AdamW optimizer")
    parser.add_argument("--lora_mode", type=str, default="residual",
                        choices=["residual", "pre_adc"],
                        help="LoRA mode: 'residual' (post-ADC, QLoRA-style) or "
                             "'pre_adc' (inside quantization, RAOQ-style)")
    parser.add_argument("--lora_layer_indices", nargs="+", type=int, default=None,
                        help="Transformer layer indices to apply LoRA to (None=all layers)")
    parser.add_argument("--lora_loss", type=str, default="ce",
                        choices=["ce", "ce_kl"],
                        help="LoRA training loss: 'ce' (cross-entropy) or "
                             "'ce_kl' (CE + KL divergence from FP teacher)")
    parser.add_argument("--lora_kl_weight", type=float, default=0.5,
                        help="Weight λ for KL term in ce_kl loss: L = CE + λ·KL")
    parser.add_argument("--lora_kl_temperature", type=float, default=2.0,
                        help="Softmax temperature T for KL divergence")

    parser.add_argument("--fq_save_transforms", action="store_true",
                        help="Save trained FlatQuant transforms to output_dir")
    parser.add_argument("--fq_reload_path", type=str, default=None,
                        help="Path to pre-trained FlatQuant transforms .pt file")

    # ADC settings
    parser.add_argument("--bx", type=int, default=8, help="Activation bits")
    parser.add_argument("--bw", type=int, default=8, help="Weight bits")
    parser.add_argument("--ba", type=int, default=8, help="ADC bits")
    parser.add_argument("--k", type=int, default=4, help="Hardware parameter")
    parser.add_argument("--ashift", action="store_true",
                        help="Enable A-shift quantization strategy")
    parser.add_argument("--activation_quant", type=str, default=None,
                        choices=["symmetric", "asymmetric"],
                        help="Activation quantization mode")
    parser.add_argument("--mvm_limit", type=int, default=256)

    # Calibration settings
    parser.add_argument("--calibration_method", type=str, default="percentile",
                        choices=["minmax", "percentile", "mse"],
                        help="Calibration method for quantizer scales")
    parser.add_argument("--num_calibration_batches", type=int, default=100,
                        help="Number of batches for ADC calibration")
    parser.add_argument("--calibration_batch_size", type=int, default=4)

    # Dataset settings
    parser.add_argument("--calibration_dataset", type=str, default="wikitext2",
                        choices=["wikitext2", "c4"])
    parser.add_argument("--eval_datasets", type=str, nargs="+", default=["wikitext2"],
                        choices=["wikitext2", "c4"])

    # Evaluation settings
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Context window size for perplexity evaluation")
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--eval_split", type=str, default="test",
                        choices=["train", "validation", "test"])
    parser.add_argument("--max_eval_samples", type=int, default=1000)
    parser.add_argument("--calibration_max_length", type=int, default=512)

    # WandB settings
    parser.add_argument("--wandb_project", type=str, default="llama-smooth-quant-adc-ptq")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--disable_wandb", action="store_true")

    # Results export
    parser.add_argument("--results_json_path", type=str, default=None,
                        help="Append a JSON record with key metrics to this file after each run. "
                             "Creates the file if it does not exist.")

    # Visualization settings
    parser.add_argument("--disable_visualizations", action="store_true",
                        help="Disable ADC visualizations")
    parser.add_argument("--visualize_layers", type=str, nargs="+", default=[],
                        help="Layer patterns for ADC 3x4 visualizations")

    # Debug / diagnostic settings
    parser.add_argument("--check_baseline", action="store_true",
                        help="Run FP16 baseline perplexity BEFORE any changes")
    parser.add_argument("--run_no_adc_eval", action="store_true",
                        help="Run extra evaluation WITHOUT ADC to isolate quantization vs ADC error")
    parser.add_argument("--run_e3_check", action="store_true",
                        help="E3 sanity-check: compare per-layer outputs of calibration path vs "
                             "inference path (after reparameterize + ADC conversion) on one sample")
    parser.add_argument("--stage_eval", action="store_true",
                        help="Stage-by-stage perplexity: FP baseline → post-FQ-reparameterize → "
                             "post-ADC-replace (bypass FP) → post-calibration. Pinpoints where "
                             "the pipeline degrades.")
    parser.add_argument("--stage_eval_max_windows", type=int, default=50,
                        help="Max sliding-window chunks for each stage_eval pass (default 50, ~fast)")
    parser.add_argument("--layer_ablation", action="store_true",
                        help="Layer-by-layer ablation: bypass ADC in one decoder layer at a time "
                             "and eval perplexity. Pinpoints which layer contributes most to ADC error.")
    parser.add_argument("--layer_ablation_max_windows", type=int, default=20,
                        help="Max sliding-window chunks per layer in ablation (default 20)")
    parser.add_argument("--proj_ablation", action="store_true",
                        help="Projection-type ablation: bypass ADC for each proj type "
                             "(q/k/v/o/gate/up/down) across all layers. Shows which projection "
                             "type is the bottleneck. Reuses layer_ablation_max_windows.")

    args = parser.parse_args()
    set_seed(args.seed)

    # Derive signed_activations flag
    if args.activation_quant == "symmetric":
        signed_activations = True
    elif args.activation_quant == "asymmetric":
        signed_activations = False
    else:
        signed_activations = None

    # Log quantization strategy
    logger.info(f"Quantization strategy: ashift={args.ashift}, activation_quant={args.activation_quant}")
    if signed_activations is True:
        logger.info("  -> Symmetric (signed) activation quantization for ALL layers")
    elif signed_activations is False:
        logger.info("  -> Asymmetric (unsigned) activation quantization for ALL layers")
    elif args.ashift:
        logger.info("  -> Asymmetric (unsigned) + A-shift for layers AFTER SiLU (down_proj)")
        logger.info("  -> Symmetric (signed) for all OTHER activations")
    else:
        logger.info("  -> Symmetric (signed) quantization for ALL activations (default)")
    logger.info("  -> Weights: always symmetric (signed) per-channel")

    if args.disable_smooth_quant and args.preprocess_method == "smooth_quant":
        logger.warning("--disable_smooth_quant is deprecated. Falling back to preprocess_method=none")
        args.preprocess_method = "none"

    if args.preprocess_method == "smooth_quant":
        logger.info(f"Preprocess: smooth_quant (alpha={args.alpha}, batches={args.smooth_quant_batches})")
    elif args.preprocess_method == "flat_quant":
        logger.info(
            "Preprocess: flat_quant "
            f"(w_bits={args.fq_w_bits}, a_bits={args.fq_a_bits}, epochs={args.fq_epochs}, "
            f"lr={args.fq_lr}, diag={args.fq_add_diag}, lwc={args.fq_lwc}, lac={args.fq_lac})"
        )
    else:
        logger.info("Preprocess: none")

    # Initialize WandB
    use_wandb = not args.disable_wandb
    model_short_name = args.model_name.split("/")[-1]
    if use_wandb:
        if args.preprocess_method == "smooth_quant":
            default_run_name = (
                f"sq_ptq_{model_short_name}_a{args.alpha}_bx{args.bx}_bw{args.bw}_"
                f"ba{args.ba}_k{args.k}_{args.calibration_method}"
            )
        elif args.preprocess_method == "flat_quant":
            default_run_name = (
                f"fq_ptq_{model_short_name}_w{args.fq_w_bits}a{args.fq_a_bits}_e{args.fq_epochs}_"
                f"bx{args.bx}_bw{args.bw}_ba{args.ba}_k{args.k}_{args.calibration_method}"
                f"_lc{args.fq_lambda_clip}_ld{args.fq_lambda_dead}"
                + (f"_lct{args.fq_lambda_center}" if args.fq_lambda_center > 0 else "")
                + ("_prop" if args.fq_propagate_quant else "")
                + (f"_pa{args.fq_prop_alpha}" if args.fq_propagate_quant and args.fq_prop_alpha < 1.0 else "")
                + (f"_pae{args.fq_prop_alpha_early}" if args.fq_prop_alpha_early is not None else "")
                + (f"_sb{args.fq_stage_b_epochs}" if args.fq_stage_b_epochs > 0 else "")
                + ("_diagattn" if args.fq_add_diag and args.fq_diag_attn and not args.fq_diag_mlp else "")
                + ("_diagmlp"  if args.fq_add_diag and args.fq_diag_mlp  and not args.fq_diag_attn else "")
                + ("_diagup"   if args.fq_add_diag and args.fq_diag_mlp_up   and not args.fq_diag_mlp_down else "")
                + ("_diagdown" if args.fq_add_diag and args.fq_diag_mlp_down and not args.fq_diag_mlp_up   else "")
                + ("_stagedb"  if args.fq_stage_b_epochs > 0 and (args.fq_stage_b_diag_attn or args.fq_stage_b_diag_mlp) else "")
                + (f"_stoch{args.fq_stochastic_mode}" if args.fq_stochastic_prop else "")
                + (f"_b{args.fq_beta_param}" if args.fq_stochastic_prop and args.fq_stochastic_mode == "beta" else "")
                + ("_pact" if args.pact_inference else "")
                + (f"_lora_{args.lora_mode}_r{args.lora_rank}" if args.lora_rank > 0 else "")
                + (f"_kl" if args.lora_rank > 0 and args.lora_loss == "ce_kl" else "")
                + (f"_top{len(args.lora_layer_indices)}" if args.lora_rank > 0 and args.lora_layer_indices else "")
            )
        else:
            default_run_name = (
                f"ptq_{model_short_name}_bx{args.bx}_bw{args.bw}_"
                f"ba{args.ba}_k{args.k}_{args.calibration_method}"
            )

        run_name = args.wandb_run_name or default_run_name
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model_name": args.model_name,
                "preprocess_method": args.preprocess_method,
                "smooth_quant_alpha": args.alpha,
                "smooth_quant_batches": args.smooth_quant_batches,
                "smooth_quant_enabled": args.preprocess_method == "smooth_quant",
                "fq_w_bits": args.fq_w_bits,
                "fq_a_bits": args.fq_a_bits,
                "fq_nsamples": args.fq_nsamples,
                "fq_epochs": args.fq_epochs,
                "fq_lr": args.fq_lr,
                "fq_add_diag": args.fq_add_diag,
                "fq_lwc": args.fq_lwc,
                "fq_lac": args.fq_lac,
                "fq_reload_path": args.fq_reload_path,
                "bx": args.bx,
                "bw": args.bw,
                "ba": args.ba,
                "k": args.k,
                "ashift": args.ashift,
                "activation_quant": args.activation_quant,
                "signed_activations": signed_activations,
                "mvm_limit": args.mvm_limit,
                "calibration_method": args.calibration_method,
                "calibration_dataset": args.calibration_dataset,
                "eval_datasets": args.eval_datasets,
                "num_calibration_batches": args.num_calibration_batches,
                "calibration_batch_size": args.calibration_batch_size,
                "calibration_max_length": args.calibration_max_length,
                "eval_max_length": args.max_length,
                "eval_stride": args.stride,
                "eval_split": args.eval_split,
                "perplexity_method": "sliding_window",
                "torch_dtype": args.torch_dtype,
                "seed": args.seed,
            }
        )
        logger.info(f"WandB initialized: project={args.wandb_project}, run={run_name}")
    else:
        logger.info("WandB logging disabled")

    # Load LLaMA model
    logger.info(f"Loading LLaMA model: {args.model_name}")

    args.output_dir = append_current_date_to_path(args.output_dir)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map.get(args.torch_dtype, torch.float16)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    if hasattr(model, 'device'):
        device = model.device
    elif hasattr(model, 'hf_device_map'):
        device = next(iter(model.hf_device_map.values())) if model.hf_device_map else device

    logger.info(f"Model loaded with dtype={torch_dtype}, device={device}")

    # =========================================================================
    # Optional: FP16 baseline perplexity check (before any changes)
    # =========================================================================
    if args.check_baseline:
        logger.info("=" * 80)
        logger.info("BASELINE CHECK: FP16 perplexity (no quantization, no SmoothQuant)")
        logger.info("=" * 80)
        baseline_enc = load_and_tokenize_for_sliding_window(
            args.eval_datasets[0], args.eval_split, tokenizer,
            max_samples=args.max_eval_samples if args.eval_datasets[0] == "c4" else None,
        )
        baseline_metrics = compute_perplexity_sliding_window(
            model, baseline_enc, device,
            max_length=args.max_length, stride=args.stride,
            desc="Baseline FP16",
        )
        logger.info(f"FP16 BASELINE Perplexity: {baseline_metrics['perplexity']:.4f}  "
                     f"(Loss: {baseline_metrics['avg_loss']:.4f})")
        if use_wandb:
            wandb.log({"baseline/perplexity": baseline_metrics["perplexity"],
                        "baseline/avg_loss": baseline_metrics["avg_loss"]})
        logger.info("=" * 80)

    # =========================================================================
    # Pre-load eval encodings for stage_eval (truncated to max_windows for speed)
    # =========================================================================
    stage_eval_enc = None
    if args.stage_eval:
        logger.info("Stage eval: pre-loading eval encodings...")
        stage_eval_enc = load_and_tokenize_for_sliding_window(
            args.eval_datasets[0], args.eval_split, tokenizer,
        )
        _eff_stride = args.stride if args.stride is not None else args.max_length // 2
        _max_tokens = args.stage_eval_max_windows * _eff_stride + args.max_length
        if stage_eval_enc["input_ids"].size(1) > _max_tokens:
            stage_eval_enc = {
                "input_ids": stage_eval_enc["input_ids"][:, :_max_tokens],
                "attention_mask": stage_eval_enc["attention_mask"][:, :_max_tokens],
            }
            logger.info(
                f"Stage eval: truncated to {_max_tokens} tokens "
                f"(~{args.stage_eval_max_windows} windows per stage)"
            )

    # =========================================================================
    # STEP 0: OPTIONAL PREPROCESSING (SmoothQuant / FlatQuant / None)
    # =========================================================================
    preprocess_summary = "DISABLED"
    if args.preprocess_method in ("smooth_quant", "flat_quant"):
        logger.info("=" * 80)
        logger.info(f"STEP 0: {args.preprocess_method.upper()} PREPROCESSING")
        logger.info("=" * 80)

        logger.info(f"Loading {args.calibration_dataset.upper()} for preprocessing calibration...")
        is_streaming = (args.calibration_dataset == "c4")
        pre_calib_raw = load_dataset_by_name(args.calibration_dataset, split="train")
        pre_calib_dataset = prepare_dataset_for_lm(
            pre_calib_raw,
            tokenizer,
            max_length=args.calibration_max_length,
            max_samples=2000,
            min_text_length=50,
            streaming=is_streaming,
        )

        def pre_collator(features):
            return {
                "input_ids": torch.tensor([f["input_ids"] for f in features]),
                "attention_mask": torch.tensor([f["attention_mask"] for f in features]),
            }

        pre_loader = DataLoader(
            pre_calib_dataset,
            batch_size=args.calibration_batch_size,
            shuffle=False,
            collate_fn=pre_collator,
        )

        if args.preprocess_method == "smooth_quant":
            logger.info(f"Alpha: {args.alpha}, Calibration batches: {args.smooth_quant_batches}")
            preprocess_summary = f"smooth_quant(alpha={args.alpha})"

            # Save original tensors for optional visualization.
            original_weights = {}
            sq_viz_layers = args.smooth_quant_layers
            if sq_viz_layers:
                for name, module in model.named_modules():
                    if isinstance(module, nn.Linear) and any(pat in name for pat in sq_viz_layers):
                        original_weights[name] = module.weight.detach().cpu().float().clone()
                        logger.info(f"  Saved original weights for viz: {name}")

            original_activations = {}
            sample_for_sq = None
            if sq_viz_layers and len(pre_calib_dataset) > 0:
                sample_for_sq = {
                    "input_ids": torch.tensor([pre_calib_dataset[0]["input_ids"]]).to(device),
                    "attention_mask": torch.tensor([pre_calib_dataset[0]["attention_mask"]]).to(device),
                }
                for name, module in model.named_modules():
                    if isinstance(module, nn.Linear) and any(pat in name for pat in sq_viz_layers):
                        act = _capture_activations_for_layer(model, sample_for_sq, name, device)
                        if act is not None:
                            original_activations[name] = act.clone()
                            logger.info(f"  Captured original activations for viz: {name} {act.shape}")

            logger.info("Calibrating SmoothQuant activation statistics...")
            act_maxes = calibrate_smooth_scales(
                model,
                pre_loader,
                num_batches=args.smooth_quant_batches,
                device=device,
            )

            logger.info("Applying SmoothQuant smoothing...")
            applied_scales = apply_smooth_quant(model, act_maxes, alpha=args.alpha)

            if use_wandb:
                for group_key, scales in applied_scales.items():
                    wandb.log({
                        f"smooth_quant/{group_key}/scale_mean": scales.mean().item(),
                        f"smooth_quant/{group_key}/scale_std": scales.std().item(),
                        f"smooth_quant/{group_key}/scale_min": scales.min().item(),
                        f"smooth_quant/{group_key}/scale_max": scales.max().item(),
                    })

            if sq_viz_layers and not args.disable_visualizations and sample_for_sq is not None:
                logger.info("Generating SmoothQuant 3D visualizations...")
                sq_viz_dir = os.path.join(args.output_dir, "viz_smooth_quant")
                os.makedirs(sq_viz_dir, exist_ok=True)

                for name, module in model.named_modules():
                    if isinstance(module, nn.Linear) and any(pat in name for pat in sq_viz_layers):
                        if name not in original_weights or name not in original_activations:
                            continue
                        smoothed_act = _capture_activations_for_layer(model, sample_for_sq, name, device)
                        if smoothed_act is None:
                            continue

                        clean_name = name.replace(".", "_").replace("/", "_")
                        filepath = os.path.join(sq_viz_dir, f"{clean_name}_smooth_quant_3d.png")
                        _generate_smooth_quant_3d_visualization(
                            act_original=original_activations[name],
                            act_smoothed=smoothed_act,
                            weight_original=original_weights[name],
                            weight_smoothed=module.weight.detach().cpu().float(),
                            layer_name=name,
                            filepath=filepath,
                        )
                        if use_wandb:
                            wandb.log({f"viz_smooth_quant/{clean_name}": wandb.Image(filepath)})
                logger.info(f"SmoothQuant 3D visualizations saved to {sq_viz_dir}")

            logger.info("=" * 80)
            logger.info("SmoothQuant preprocessing complete")
            logger.info("=" * 80)
        else:
            # --- FlatQuant: apply wrappers -> train -> reparameterize -> strip ---
            logger.info(
                f"FlatQuant config: w_bits={args.fq_w_bits}, a_bits={args.fq_a_bits}, "
                f"epochs={args.fq_epochs}, lr={args.fq_lr}, nsamples={args.fq_nsamples}, "
                f"diag={args.fq_add_diag}, lwc={args.fq_lwc}, lac={args.fq_lac}"
            )
            preprocess_summary = (
                f"flat_quant(w{args.fq_w_bits}a{args.fq_a_bits}, "
                f"epochs={args.fq_epochs}, lr={args.fq_lr})"
            )

            logger.info("Applying FlatQuant wrappers to model...")
            # signed_activations: None → True when ashift=False (default)
            fq_signed = signed_activations if signed_activations is not None else (not args.ashift)
            fq_adc_config = {
                "bx": args.bx,
                "bw": args.bw,
                "ba": args.ba,
                "k":  args.k,
                "mvm_limit": args.mvm_limit,
                "signed_activations": fq_signed,
            }
            logger.info(
                f"FlatQuant ADC config: bx={fq_adc_config['bx']}, bw={fq_adc_config['bw']}, "
                f"ba={fq_adc_config['ba']}, k={fq_adc_config['k']}, "
                f"mvm_limit={fq_adc_config['mvm_limit']}, "
                f"signed={fq_adc_config['signed_activations']}, "
                f"delta≈{2.0 * min(args.mvm_limit, 256) * (127 if fq_signed else 255) * 127 / (2**args.ba * args.k):.1f}"
            )
            model = apply_flatquant_to_model(
                model,
                w_bits=args.fq_w_bits,
                a_bits=args.fq_a_bits,
                add_diag=args.fq_add_diag,
                lwc=args.fq_lwc,
                lac=args.fq_lac,
                adc_config=fq_adc_config,  # Simulate ADC (floor_ste) during FQ training so transforms optimise for ADC
            )

            if args.fq_reload_path:
                logger.info(f"Loading pre-trained FlatQuant transforms from {args.fq_reload_path}")
                model = load_flat_transforms(model, args.fq_reload_path)
            else:
                logger.info("Starting FlatQuant layer-by-layer calibration...")
                model = calibrate_flat_quant(
                    model,
                    dataloader=pre_loader,
                    device=device,
                    nsamples=args.fq_nsamples,
                    cali_bsz=args.fq_cali_bsz,
                    epochs=args.fq_epochs,
                    flat_lr=args.fq_lr,
                    diag_alpha=args.fq_diag_alpha,
                    add_diag=args.fq_add_diag,
                    lwc=args.fq_lwc,
                    lac=args.fq_lac,
                    lambda_clip=args.fq_lambda_clip,
                    lambda_dead=args.fq_lambda_dead,
                    clip_margin=args.fq_clip_margin,
                    dead_threshold=args.fq_dead_threshold,
                    penalty_projections=args.fq_penalty_projections,
                    lambda_band=args.fq_lambda_band,
                    band_tau_lo=args.fq_band_tau_lo,
                    band_tau_hi=args.fq_band_tau_hi,
                    band_beta=args.fq_band_beta,
                    band_topk_frac=args.fq_band_topk_frac,
                    freeze_clip=args.fq_freeze_clip,
                    kronecker_init=args.fq_kronecker_init,
                    loss_type=args.fq_loss_type,
                    huber_delta=args.fq_huber_delta,
                    lambda_center=args.fq_lambda_center,
                    propagate_quant_inputs=args.fq_propagate_quant,
                    propagate_quant_alpha=args.fq_prop_alpha,
                    prop_alpha_early=args.fq_prop_alpha_early,
                    prop_late_start=args.fq_prop_late_start,
                    diag_attn=(not args.fq_diag_attn and not args.fq_diag_mlp) or args.fq_diag_attn,
                    diag_mlp=(not args.fq_diag_attn and not args.fq_diag_mlp) or args.fq_diag_mlp,
                    diag_mlp_up=(not args.fq_diag_mlp_up and not args.fq_diag_mlp_down) or args.fq_diag_mlp_up,
                    diag_mlp_down=(not args.fq_diag_mlp_up and not args.fq_diag_mlp_down) or args.fq_diag_mlp_down,
                    stochastic_prop=args.fq_stochastic_prop,
                    stochastic_mode=args.fq_stochastic_mode,
                    beta_param=args.fq_beta_param,
                )

            if args.fq_stage_b_epochs > 0:
                # Stage B diag selection:
                # If --fq_stage_b_diag_attn or --fq_stage_b_diag_mlp are set, use those.
                # Otherwise inherit Stage A diag settings.
                _sb_neither = not args.fq_stage_b_diag_attn and not args.fq_stage_b_diag_mlp
                _stage_a_diag_attn = (not args.fq_diag_attn and not args.fq_diag_mlp) or args.fq_diag_attn
                _stage_a_diag_mlp  = (not args.fq_diag_attn and not args.fq_diag_mlp) or args.fq_diag_mlp
                _sb_diag_attn = (_stage_a_diag_attn if _sb_neither else args.fq_stage_b_diag_attn)
                _sb_diag_mlp  = (_stage_a_diag_mlp  if _sb_neither else args.fq_stage_b_diag_mlp)
                logger.info(
                    f"Stage B: {args.fq_stage_b_epochs} epochs, "
                    f"prop=True, prop_alpha={args.fq_stage_b_prop_alpha}, "
                    f"diag_attn={_sb_diag_attn}, diag_mlp={_sb_diag_mlp}"
                )
                model = calibrate_flat_quant(
                    model,
                    dataloader=pre_loader,
                    device=device,
                    nsamples=args.fq_nsamples,
                    cali_bsz=args.fq_cali_bsz,
                    epochs=args.fq_stage_b_epochs,
                    flat_lr=args.fq_lr * 0.1,
                    diag_alpha=args.fq_diag_alpha,
                    add_diag=args.fq_add_diag,
                    lwc=args.fq_lwc,
                    lac=args.fq_lac,
                    kronecker_init=args.fq_kronecker_init,
                    loss_type=args.fq_loss_type,
                    propagate_quant_inputs=True,
                    propagate_quant_alpha=args.fq_stage_b_prop_alpha,
                    diag_attn=_sb_diag_attn,
                    diag_mlp=_sb_diag_mlp,
                    diag_mlp_up=True,
                    diag_mlp_down=True,
                    stochastic_prop=args.fq_stochastic_prop,
                    stochastic_mode=args.fq_stochastic_mode,
                    beta_param=args.fq_beta_param,
                )

            if args.fq_save_transforms:
                os.makedirs(args.output_dir, exist_ok=True)
                transforms_path = os.path.join(args.output_dir, "flat_quant_transforms.pt")
                save_flat_transforms(model, transforms_path)
                if use_wandb:
                    wandb.run.summary["fq_transforms_path"] = transforms_path

            # E3: capture calibration-path outputs BEFORE reparameterize.
            # After layer-by-layer FlatQuant calibration the model is partially
            # offloaded (each layer was moved to GPU then left scattered).
            # Move everything to device now — it will stay there for all
            # subsequent steps (reparameterize, ADC conversion, calibration).
            e3_calib_outputs = None
            e3_sample = None
            if args.run_e3_check:
                logger.info("E3: moving model to device for capture...")
                model = model.to(device)
                logger.info("E3: capturing calibration-path layer outputs (before reparameterize)...")
                e3_sample = {
                    "input_ids": torch.randint(0, model.config.vocab_size, (1, 64), device=device),
                    "attention_mask": torch.ones(1, 64, dtype=torch.long, device=device),
                }
                e3_calib_outputs = capture_layer_outputs(model, e3_sample, device)
                logger.info(f"E3: captured outputs for {len(e3_calib_outputs)} layers")

            # Ensure all parameters (including FlatQuantLinear clip_factor_*) are on
            # the correct device before reparameterize — especially needed when using
            # --fq_reload_path, where layer-by-layer calibration never runs and
            # model.to(device) above is gated on --run_e3_check / --stage_eval flags.
            model = model.to(device)
            logger.info("Reparameterizing FlatQuant transforms into weights...")
            model = fq_reparameterize_model(model)

            # NOTE: We intentionally do NOT call strip_flatquant_wrappers() here.
            # The Kronecker transform must remain active on activations at inference
            # so that W_stored = W/D @ Kron^{-T} (baked into weights) is used
            # correctly: (x * D @ Kron) @ (W/D @ Kron^{-T})^T = x @ W^T.
            # FlatQuantLinear._reparameterized=True ensures train_forward() delegates
            # directly to self.linear (nn.Linear or TiledLinearADC after conversion).

            # Stage eval B: FP model with FlatQuant transforms baked in (no ADC yet).
            # Good result here means FlatQuant reparameterization is correct.
            # Bad result here means FlatQuant training itself is broken.
            if args.stage_eval and stage_eval_enc is not None:
                model = model.to(device)
                _run_stage_eval(model, stage_eval_enc, device,
                                "B_post_fq_reparameterize", args, use_wandb)

            logger.info("=" * 80)
            logger.info("FlatQuant preprocessing complete")
            logger.info("=" * 80)
    else:
        logger.info("Preprocessing disabled, skipping STEP 0")

    # =========================================================================
    # STEP 1: Convert to ADC layers (on the now-smoothed model)
    # =========================================================================
    logger.info("Converting to ADC layers...")
    model = LlamaADCConverter.replace_linear_with_adc(
        model,
        bx=args.bx,
        bw=args.bw,
        ba=args.ba,
        k=args.k,
        ashift=args.ashift,
        signed_activations=signed_activations,
        exclude_patterns=["embed_tokens", "lm_head", "_orig_attn"],
        mvm_limit=args.mvm_limit,
        use_kurtosis_loss=False,
        kurtosis_weight=0.0,
        target_kurtosis=1.8,
    )

    model = model.to(device)
    logger.info(f"Model moved to {device}")

    # Propagate learned PACT alpha_adc from FlatQuantLinear → TiledLinearADC.
    # Only when --pact_inference is set; otherwise TiledLinearADC uses
    # per-token amax at inference (same quantization scheme as training).
    if args.pact_inference:
        propagate_alpha_adc_to_tiled(model)
        logger.info("PACT inference enabled: per-tile alpha propagated to TiledLinearADC")
    else:
        logger.info("PACT inference disabled: TiledLinearADC uses per-token amax at inference")

    stats = LlamaADCConverter.count_adc_layers(model)
    logger.info(f"Model: {stats['adc_linear']} ADC layers, {stats['regular_linear']} regular Linear, {stats['total_params']:,} params")

    # Stage eval C: ADC layers present, bypass_all=True → FP forward through ADC structure.
    # Should match stage B if replace_linear_with_adc is transparent in bypass mode.
    # Divergence here means the TiledLinearADC replacement itself breaks something.
    if args.stage_eval and stage_eval_enc is not None:
        logger.info("Stage eval C: setting bypass_all=True on all TiledLinearADC...")
        for _, _m in model.named_modules():
            if isinstance(_m, TiledLinearADC):
                _m.set_bypass_all(True)
        _run_stage_eval(model, stage_eval_enc, device,
                        "C_post_adc_replace_bypass", args, use_wandb)
        for _, _m in model.named_modules():
            if isinstance(_m, TiledLinearADC):
                _m.set_bypass_all(False)
        logger.info("Stage eval C: bypass_all restored to False")

    viz_patterns = args.visualize_layers if not args.disable_visualizations else []
    model_structure_text = show_model_with_adc_hooks(model, viz_patterns)

    if use_wandb:
        wandb.run.summary["model_structure"] = model_structure_text

    # Prepare sample input for ADC visualization
    sample_input = None
    if not args.disable_visualizations:
        logger.info("Preparing sample input for visualizations...")

    # =========================================================================
    # Load calibration dataset for ADC calibration
    # =========================================================================
    logger.info(f"Loading {args.calibration_dataset.upper()} dataset for ADC calibration...")

    is_streaming = (args.calibration_dataset == "c4")
    calib_raw = load_dataset_by_name(args.calibration_dataset, split="train")

    calibration_dataset = prepare_dataset_for_lm(
        calib_raw,
        tokenizer,
        max_length=args.calibration_max_length,
        max_samples=2000,
        min_text_length=50,
        streaming=is_streaming
    )

    logger.info(f"Calibration dataset: {len(calibration_dataset)} samples from {args.calibration_dataset}")

    if not args.disable_visualizations and len(calibration_dataset) > 0:
        sample_input = {
            'input_ids': torch.tensor([calibration_dataset[0]['input_ids']]).to(device),
            'attention_mask': torch.tensor([calibration_dataset[0]['attention_mask']]).to(device),
        }
        logger.info(f"Sample input prepared for visualization (shape: {sample_input['input_ids'].shape})")

    def calibration_collator(features):
        batch = {
            "input_ids": torch.tensor([f["input_ids"] for f in features]),
            "attention_mask": torch.tensor([f["attention_mask"] for f in features]),
        }
        return batch

    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=args.calibration_batch_size,
        shuffle=False,
        collate_fn=calibration_collator,
    )

    # =========================================================================
    # STEP 2: ADC Calibration
    # =========================================================================
    logger.info("=" * 80)
    logger.info("STEP 2: ADC CALIBRATION")
    logger.info("=" * 80)

    # Visualize BEFORE calibration
    viz_before = None
    if sample_input is not None and not args.disable_visualizations:
        viz_before = _generate_adc_visualizations(
            model, sample_input, args.visualize_layers,
            title_prefix="BEFORE Calibration",
            output_subdir=os.path.join(args.output_dir, "viz_before")
        )

    # Bypass ALL quantization during calibration
    logger.info("Enabling bypass_all mode for calibration (FP16 forward passes)...")
    for _, m in model.named_modules():
        if isinstance(m, TiledLinearADC):
            m.set_bypass_all(True)

    calibrator = ADCCalibrator(
        model,
        method=args.calibration_method,
        bx=args.bx,
        bw=args.bw
    )
    calibrator.calibrate(calibration_loader, num_batches=args.num_calibration_batches)

    # Disable bypass
    for _, m in model.named_modules():
        if isinstance(m, TiledLinearADC):
            m.set_bypass_all(False)
    logger.info("bypass_all disabled -- quantization re-enabled.")

    # Compute optimal parameters
    logger.info("Computing optimal quantization scales...")
    optimal_params = calibrator.compute_optimal_params(log_to_wandb=use_wandb)

    # Apply calibration
    calibrator.apply_calibration(optimal_params)

    # Set quantizers to 'fixed' mode
    for name, module in model.named_modules():
        if isinstance(module, (QATLinearADC, TiledLinearADC)):
            if hasattr(module, 'set_quantizer_mode'):
                module.set_quantizer_mode('fixed')
    logger.info("Quantizers set to 'fixed' mode after calibration")

    # =========================================================================
    # STEP 2.3 (optional): KD fine-tuning of activation scales
    # =========================================================================
    if args.kd_epochs > 0:
        perform_kd_finetuning(
            model=model,
            model_name_or_path=args.model_name,
            dataloader=calibration_loader,
            device=device,
            epochs=args.kd_epochs,
            lr=args.kd_lr,
            temperature=args.kd_temperature,
            num_batches=args.kd_batches,
            teacher_on_cpu=args.kd_teacher_on_cpu,
            use_wandb=use_wandb,
        )

    # =========================================================================
    # STEP 2.4 (optional): ADC-LoRA post-correction
    # =========================================================================
    if args.lora_rank > 0:
        logger.info(
            f"[LoRA] Applying rank={args.lora_rank} α={args.lora_alpha} "
            f"to {args.lora_target_modules}"
        )
        _lora_layer_indices = (set(args.lora_layer_indices)
                               if args.lora_layer_indices else None)
        model = apply_adc_lora(
            model,
            target_modules=args.lora_target_modules,
            rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            mode=args.lora_mode,
            layer_indices=_lora_layer_indices,
        )
        model = calibrate_adc_lora(
            model,
            dataloader=calibration_loader,
            device=device,
            nsamples=args.fq_nsamples,
            cali_bsz=args.fq_cali_bsz,
            epochs=args.lora_epochs,
            lora_lr=args.lora_lr,
            lora_loss=args.lora_loss,
            teacher_name_or_path=args.model_name if args.lora_loss == "ce_kl" else None,
            kl_weight=args.lora_kl_weight,
            kl_temperature=args.lora_kl_temperature,
        )
        if use_wandb:
            wandb.log({
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "lora_mode": args.lora_mode,
                "lora_target_modules": str(args.lora_target_modules),
                "lora_layer_indices": str(args.lora_layer_indices),
                "lora_epochs": args.lora_epochs,
                "lora_lr": args.lora_lr,
                "lora_loss": args.lora_loss,
                "lora_kl_weight": args.lora_kl_weight,
            })

    # E3: capture inference-path outputs and compare with calibration-path
    if args.run_e3_check and e3_calib_outputs is not None and e3_sample is not None:
        logger.info("=" * 80)
        logger.info("E3 SANITY CHECK: calibration path vs inference path")
        logger.info("=" * 80)
        e3_inf_outputs = capture_layer_outputs(model, e3_sample, device)
        e3_results = compare_layer_outputs(e3_calib_outputs, e3_inf_outputs)
        if use_wandb:
            flat = {}
            for layer_idx, parts in e3_results.items():
                for part, stats in parts.items():
                    flat[f"e3/{layer_idx}/{part}/mse"] = stats["mse"]
                    flat[f"e3/{layer_idx}/{part}/rel_error"] = stats["rel_error"]
            if flat:
                all_mse = [v for k, v in flat.items() if k.endswith("/mse")]
                all_rel = [v for k, v in flat.items() if k.endswith("/rel_error")]
                flat["e3/mean_mse"] = sum(all_mse) / len(all_mse)
                flat["e3/max_mse"] = max(all_mse)
                flat["e3/mean_rel_error"] = sum(all_rel) / len(all_rel)
                flat["e3/max_rel_error"] = max(all_rel)
                wandb.log(flat)
        logger.info("=" * 80)

    # Run diagnostics
    diag = diagnose_quantized_model(model, tokenizer, device)
    if diag["has_nan"] or diag["has_inf"]:
        logger.error("ABORTING: Model produces NaN/Inf after calibration!")
        if use_wandb:
            wandb.finish(exit_code=1)
        return

    if diag["n_default_act"] > 0:
        logger.warning(f"{diag['n_default_act']} layers have uncalibrated activation scales!")

    # ADC layer diagnostics (always runs)
    logger.info("Collecting ADC layer diagnostics...")
    adc_diag_batches = min(32, args.num_calibration_batches)
    adc_diag_results = collect_adc_layer_diagnostics(
        model, calibration_loader, adc_diag_batches, device
    )
    log_adc_diagnostics(adc_diag_results, use_wandb=use_wandb)

    # Latency measurement
    logger.info("Measuring forward-pass latency...")
    _lat_ids = torch.randint(
        0, model.config.vocab_size, (1, min(512, args.calibration_max_length)), device=device
    )
    lat = measure_latency(model, _lat_ids, n_warmup=5, n_runs=20)
    logger.info(
        f"Latency (bs=1, seq={lat['seq_len']}):  "
        f"mean={lat['latency_mean_ms']:.1f}ms  "
        f"p95={lat['latency_p95_ms']:.1f}ms  "
        f"throughput={lat['throughput_tok_s']:.0f} tok/s"
    )
    if use_wandb:
        wandb.log({f"latency/{k}": v for k, v in lat.items()})

    # =========================================================================
    # Layer-by-layer ablation (optional)
    # =========================================================================
    if args.layer_ablation:
        _run_layer_ablation(model, tokenizer, device, args, use_wandb)

    # =========================================================================
    # Projection-type ablation (optional)
    # =========================================================================
    if args.proj_ablation:
        _run_proj_ablation(model, tokenizer, device, args, use_wandb)

    # Visualize AFTER calibration
    viz_after = None
    if sample_input is not None and not args.disable_visualizations:
        viz_after = _generate_adc_visualizations(
            model, sample_input, args.visualize_layers,
            title_prefix="AFTER Calibration",
            output_subdir=os.path.join(args.output_dir, "viz_after")
        )

    # Log calibration stats to wandb
    if use_wandb and len(optimal_params) > 0:
        layer_names = list(optimal_params.keys())
        act_scales = [p['act_scale'] for p in optimal_params.values()]
        w_scales = [p['w_scale'] for p in optimal_params.values()]
        y_int_targets = [p['y_int_target'] for p in optimal_params.values()]

        wandb.log({
            "calibration/act_scale_histogram": wandb.Histogram(act_scales),
            "calibration/w_scale_histogram": wandb.Histogram(w_scales),
            "calibration/y_int_target_histogram": wandb.Histogram(y_int_targets),
        })

        if viz_before:
            for name, img_path in viz_before.items():
                wandb.log({f"viz_before/{name}": wandb.Image(img_path)})
        if viz_after:
            for name, img_path in viz_after.items():
                wandb.log({f"viz_after/{name}": wandb.Image(img_path)})

    # Check model's max context length
    model_max_length = getattr(model.config, 'max_position_embeddings', 4096)
    if args.max_length > model_max_length:
        logger.warning(f"max_length ({args.max_length}) > model's max ({model_max_length}), using {model_max_length}")
        args.max_length = model_max_length

    # =========================================================================
    # STEP 2.5 (optional): Diagnostic -- quantized tiling WITHOUT ADC
    # =========================================================================
    _diag_metrics = None  # bypass PPL (no ADC), set if --run_no_adc_eval

    if args.run_no_adc_eval:
        logger.info("=" * 80)
        logger.info("STEP 2.5: DIAGNOSTIC -- Tiling + Quantization WITHOUT ADC")
        logger.info("=" * 80)

        for _, module in model.named_modules():
            if isinstance(module, TiledLinearADC):
                module.set_bypass_adc(True)

        model.eval()

        _diag_ds = args.eval_datasets[0]
        _diag_enc = load_and_tokenize_for_sliding_window(
            _diag_ds, args.eval_split, tokenizer,
            max_samples=args.max_eval_samples if _diag_ds == "c4" else None,
        )
        _diag_metrics = compute_perplexity_sliding_window(
            model, _diag_enc, device,
            max_length=args.max_length, stride=args.stride,
            desc="NoADC eval",
        )
        logger.info(
            f"WITHOUT ADC -> {_diag_ds.upper()} Perplexity: {_diag_metrics['perplexity']:.4f}  "
            f"(Loss: {_diag_metrics['avg_loss']:.4f})"
        )
        if use_wandb:
            wandb.log({
                "diagnostic/no_adc_perplexity": _diag_metrics["perplexity"],
                "diagnostic/no_adc_avg_loss": _diag_metrics["avg_loss"],
            })

        for _, module in model.named_modules():
            if isinstance(module, TiledLinearADC):
                module.set_bypass_adc(False)
        logger.info("ADC re-enabled for full evaluation.")
        logger.info("=" * 80)

    # =========================================================================
    # STEP 3: Evaluate perplexity (Sliding Window) -- WITH ADC
    # =========================================================================
    logger.info("=" * 80)
    logger.info("STEP 3: PERPLEXITY EVALUATION (Sliding Window) -- WITH ADC")
    logger.info("=" * 80)
    logger.info(f"Evaluating on datasets: {args.eval_datasets}")
    logger.info(f"Method: Sliding window (standard for papers like GPTQ, AWQ, FlatQuant)")
    logger.info(f"Context window: {args.max_length}, Stride: {args.stride or args.max_length // 2}")
    logger.info(f"Evaluation split: {args.eval_split}")

    all_eval_metrics = {}
    model.eval()

    for eval_dataset_name in args.eval_datasets:
        logger.info("-" * 40)
        logger.info(f"Evaluating on {eval_dataset_name.upper()} ({args.eval_split} split)...")

        encodings = load_and_tokenize_for_sliding_window(
            eval_dataset_name,
            args.eval_split,
            tokenizer,
            max_samples=args.max_eval_samples if eval_dataset_name == "c4" else None
        )

        metrics = compute_perplexity_sliding_window(
            model, encodings, device,
            max_length=args.max_length, stride=args.stride,
            desc=f"Eval {eval_dataset_name}"
        )

        all_eval_metrics[eval_dataset_name] = metrics
        logger.info(f"  {eval_dataset_name.upper()} Perplexity: {metrics['perplexity']:.4f}")

    eval_metrics = all_eval_metrics[args.eval_datasets[0]]
    perplexity = eval_metrics["perplexity"]
    avg_loss = eval_metrics["avg_loss"]

    # Print summary
    logger.info("=" * 80)
    logger.info("RESULTS SUMMARY (Sliding Window Perplexity)")
    logger.info("=" * 80)
    logger.info(f"Preprocess: {preprocess_summary}")
    logger.info(f"Context window: {args.max_length}, Stride: {args.stride or args.max_length // 2}")
    for ds_name, metrics in all_eval_metrics.items():
        logger.info(
            f"{ds_name.upper():12s} Perplexity: {metrics['perplexity']:.4f}  "
            f"(Loss: {metrics['avg_loss']:.4f}, Tokens: {metrics['total_tokens']:,}, "
            f"Windows: {metrics['num_windows']})"
        )

    # Log all metrics to wandb
    if use_wandb:
        for ds_name, metrics in all_eval_metrics.items():
            wandb.log({
                f"eval/{ds_name}/perplexity": metrics["perplexity"],
                f"eval/{ds_name}/avg_loss": metrics["avg_loss"],
                f"eval/{ds_name}/total_tokens": metrics["total_tokens"],
                f"eval/{ds_name}/num_windows": metrics["num_windows"],
            })
            wandb.run.summary[f"{ds_name}_perplexity"] = metrics["perplexity"]
            wandb.run.summary[f"{ds_name}_avg_loss"] = metrics["avg_loss"]

        wandb.run.summary["num_calibrated_layers"] = len(optimal_params)
        wandb.run.summary["calibration_dataset"] = args.calibration_dataset
        wandb.run.summary["eval_datasets"] = ",".join(args.eval_datasets)
        wandb.run.summary["eval_max_length"] = args.max_length
        wandb.run.summary["eval_stride"] = args.stride or args.max_length // 2

    # Save calibrated model
    logger.info(f"Saving calibrated model to: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save calibration info
    with open(os.path.join(args.output_dir, "calibration_info.txt"), "w") as f:
        f.write("=" * 80 + "\n")
        f.write("LLaMA PREPROCESS + ADC POST-TRAINING QUANTIZATION (PTQ) RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Preprocess: {preprocess_summary}\n")
        f.write(f"Calibration method: {args.calibration_method}\n")
        f.write(f"Calibration batches: {args.num_calibration_batches}\n")
        f.write(f"ADC hardware config: bx={args.bx}, bw={args.bw}, ba={args.ba}, k={args.k}\n")
        f.write(f"A-shift: {args.ashift}\n")
        f.write(f"activation_quant: {args.activation_quant}\n")
        if signed_activations is True:
            f.write("Quantization strategy: Symmetric (signed) activations for all layers\n")
        elif signed_activations is False:
            f.write("Quantization strategy: Asymmetric (unsigned) activations for all layers\n")
        elif args.ashift:
            f.write("Quantization strategy: Per-layer (A-shift for SiLU outputs only)\n")
            f.write("  - Layers after SiLU (down_proj): Asymmetric + A-shift\n")
            f.write("  - All other layers: Symmetric (signed)\n")
        else:
            f.write("Quantization strategy: Symmetric (signed) for all activations (default)\n")
        f.write("Weights: always symmetric (signed) per-channel\n")
        f.write("\n")
        f.write("NOTE: ADC delta is a HARDWARE CONSTANT and cannot be changed!\n")
        f.write("      We calibrate activation/weight SCALES to optimally use the fixed ADC range.\n")
        f.write("\n")
        f.write("Results:\n")
        f.write(f"  Perplexity:  {eval_metrics['perplexity']:.2f}\n")
        f.write(f"  Avg Loss:    {eval_metrics['avg_loss']:.4f}\n")
        f.write("\n")
        f.write(f"Calibrated layers: {len(optimal_params)}\n")
        f.write("\n")
        f.write("Per-layer calibrated scales:\n")
        f.write("-" * 80 + "\n")
        for name, params in sorted(optimal_params.items()):
            f.write(f"\n{name}:\n")
            f.write(f"  Activation scale: {params['act_scale']:.6f}\n")
            f.write(f"  Weight scale:     {params['w_scale']:.6f}\n")
            f.write(f"  Y_int target:     {params['y_int_target']:.2f}\n")

    with open(os.path.join(args.output_dir, "eval_metrics.txt"), "w") as f:
        for k, v in sorted(eval_metrics.items()):
            f.write(f"{k}: {v}\n")

    logger.info("=" * 80)
    logger.info("PTQ COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Calibrated model saved to: {args.output_dir}")
    logger.info(f"Perplexity: {eval_metrics['perplexity']:.2f}")

    if viz_before or viz_after:
        logger.info("Visualizations:")
        if viz_before:
            logger.info(f"  Before calibration: {os.path.join(args.output_dir, 'viz_before')}")
        if viz_after:
            logger.info(f"  After calibration:  {os.path.join(args.output_dir, 'viz_after')}")

    if args.preprocess_method == "smooth_quant" and not args.disable_visualizations and args.smooth_quant_layers:
        logger.info(f"  SmoothQuant 3D:     {os.path.join(args.output_dir, 'viz_smooth_quant')}")

    # -------------------------------------------------------------------------
    # Export key metrics to JSON (for overnight batch runs)
    # -------------------------------------------------------------------------
    if args.results_json_path:
        import json as _json
        # Compute mean dead_rate and reconstruction_rel from adc_diag_results
        _dead_vals = [r["dead_rate"] for r in adc_diag_results.values() if "dead_rate" in r]
        _rel_vals  = [r["reconstruction_rel"] for r in adc_diag_results.values() if "reconstruction_rel" in r]
        _record = {
            "run_name":      getattr(args, "wandb_run_name", None) or args.output_dir,
            "timestamp":     __import__("datetime").datetime.now().isoformat(),
            "status":        "success",
            "config": {
                "bx":                    args.bx,
                "bw":                    args.bw,
                "ba":                    args.ba,
                "k":                     args.k,
                "fq_epochs":             args.fq_epochs,
                "fq_nsamples":           args.fq_nsamples,
                "fq_lambda_center":      getattr(args, "fq_lambda_center", 0.0),
                "fq_propagate_quant":    getattr(args, "fq_propagate_quant", False),
                "fq_kronecker_init":     getattr(args, "fq_kronecker_init", "random"),
            },
            "results": {
                "ppl_bypass":            float(_diag_metrics["perplexity"]) if _diag_metrics else None,
                "ppl_adc":               float(eval_metrics["perplexity"]),
                "dead_rate_mean":        float(np.mean(_dead_vals)) if _dead_vals else None,
                "dead_rate_max":         float(np.max(_dead_vals))  if _dead_vals else None,
                "reconstruction_rel_mean": float(np.mean(_rel_vals)) if _rel_vals else None,
            },
        }
        _existing = []
        try:
            with open(args.results_json_path) as _f:
                _existing = _json.load(_f)
        except (FileNotFoundError, _json.JSONDecodeError):
            pass
        _existing.append(_record)
        os.makedirs(os.path.dirname(os.path.abspath(args.results_json_path)), exist_ok=True)
        with open(args.results_json_path, "w") as _f:
            _json.dump(_existing, _f, indent=2)
        logger.info(f"Results appended to: {args.results_json_path}")

    # Final WandB logging
    if use_wandb:
        wandb.run.summary["output_dir"] = args.output_dir
        wandb.run.summary["model_path"] = os.path.join(args.output_dir, "pytorch_model.bin")

        if viz_before:
            wandb.run.summary["viz_before_count"] = len(viz_before)
        if viz_after:
            wandb.run.summary["viz_after_count"] = len(viz_after)

        logger.info("Logged all metrics to WandB")
        wandb.finish()
        logger.info("WandB run finished")


if __name__ == "__main__":
    main()
