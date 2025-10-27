#!/usr/bin/env python3
"""
Post-Training Quantization (PTQ) for ADC-based BERT QA
Calibrates ADC quantizers using a calibration dataset
"""

import argparse
import os
import logging
from typing import Dict, List
import numpy as np

import torch
import torch.nn as nn
import evaluate
from tqdm import tqdm

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    BertForQuestionAnswering,
    set_seed,
)
from torch.utils.data import DataLoader

from adc_layers import TiledLinearADC, QATLinearADC, ADCQuantizer
from bert_adc_integration import (
    BertADCConverter,
    load_qa_model_robust,
    find_last_checkpoint_dir,
    prepare_validation_features,
    postprocess_qa_predictions,
    MetricsComputer,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ADCCalibrator:
    """Calibrates ADC quantizers using activation statistics"""
    
    def __init__(self, model: nn.Module, method: str = "minmax"):
        self.model = model
        self.method = method
        self.stats = {}
        
    def register_hooks(self):
        """Register forward hooks to collect activation statistics"""
        hooks = []
        
        def make_hook(name):
            def hook(module, input, output):
                if name not in self.stats:
                    self.stats[name] = {
                        'act_min': [],
                        'act_max': [],
                        'act_absmax': [],
                        'w_min': [],
                        'w_max': [],
                        'w_absmax': [],
                        'y_int_min': [],
                        'y_int_max': [],
                        'y_int_absmax': [],
                    }
                
                # Collect activation stats
                x = input[0].detach()
                self.stats[name]['act_min'].append(x.min().item())
                self.stats[name]['act_max'].append(x.max().item())
                self.stats[name]['act_absmax'].append(x.abs().max().item())
                
                # Collect weight stats
                w = module.weight.detach()
                self.stats[name]['w_min'].append(w.min().item())
                self.stats[name]['w_max'].append(w.max().item())
                self.stats[name]['w_absmax'].append(w.abs().max().item())
                
                # Collect integer MM output (before ADC)
                # Simulate quantization to get y_int
                with torch.no_grad():
                    act_q = module.activation_quantizer
                    s_x = act_q.scale
                    if act_q.symmetric:
                        code_x = torch.clamp(torch.round(x / s_x), act_q.qmin, act_q.qmax)
                    else:
                        zp_x = act_q.zero_point
                        code_x = torch.clamp(torch.round(x / s_x + zp_x), act_q.qmin, act_q.qmax)
                    
                    w_q = module.weight_quantizer
                    s_w_vec = w_q.scale
                    s_w_b = s_w_vec.view(-1, 1)
                    code_w = torch.clamp(torch.round(w / s_w_b), w_q.qmin, w_q.qmax)
                    
                    y_int = torch.nn.functional.linear(code_x, code_w, bias=None)
                    
                    self.stats[name]['y_int_min'].append(y_int.min().item())
                    self.stats[name]['y_int_max'].append(y_int.max().item())
                    self.stats[name]['y_int_absmax'].append(y_int.abs().max().item())
            
            return hook
        
        # Register hooks on QATLinearADC layers
        for name, module in self.model.named_modules():
            if isinstance(module, QATLinearADC):
                hook = module.register_forward_hook(make_hook(name))
                hooks.append(hook)
                logger.info(f"Registered hook on {name}")
        
        return hooks
    
    def calibrate(self, dataloader, num_batches: int = 100):
        """Run calibration on dataloader"""
        logger.info(f"Running calibration on {num_batches} batches...")
        
        self.model.eval()
        hooks = self.register_hooks()
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, total=num_batches, desc="Calibrating")):
                if i >= num_batches:
                    break
                
                # Move batch to device
                batch = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                try:
                    _ = self.model(**batch)
                except Exception as e:
                    logger.warning(f"Error in batch {i}: {e}")
                    continue
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        logger.info(f"Collected stats for {len(self.stats)} layers")
    
    def compute_optimal_params(self) -> Dict[str, Dict]:
        """
        Compute optimal quantization SCALES (not delta!) from collected statistics.
        
        Key insight: Delta is a hardware constant and cannot be changed.
        We calibrate activation/weight scales to optimally use the fixed ADC range.
        
        Goal: Make y_int values fit well within the ADC dynamic range
        given fixed delta and ADC range [-na, pa].
        """
        optimal_params = {}
        
        for name, stats in self.stats.items():
            if not stats['y_int_absmax']:  # No data collected
                continue
            
            # Collect statistics
            act_absmax_arr = np.array(stats['act_absmax'])
            w_absmax_arr = np.array(stats['w_absmax'])
            y_int_absmax_arr = np.array(stats['y_int_absmax'])
            
            # Use percentile to be robust against outliers
            if self.method == "minmax":
                act_absmax = act_absmax_arr.max()
                w_absmax = w_absmax_arr.max()
                y_int_target = y_int_absmax_arr.max()
            elif self.method == "percentile":
                act_absmax = np.percentile(act_absmax_arr, 99.9)
                w_absmax = np.percentile(w_absmax_arr, 99.9)
                y_int_target = np.percentile(y_int_absmax_arr, 99.9)
            elif self.method == "mse":
                act_absmax = self._find_mse_optimal_threshold(act_absmax_arr)
                w_absmax = self._find_mse_optimal_threshold(w_absmax_arr)
                y_int_target = self._find_mse_optimal_threshold(y_int_absmax_arr)
            else:
                act_absmax = act_absmax_arr.max()
                w_absmax = w_absmax_arr.max()
                y_int_target = y_int_absmax_arr.max()
            
            # Compute optimal scales
            # For symmetric quantization: scale = absmax / (2^(n-1) - 1)
            optimal_act_scale = act_absmax / 127.0  # 127 = 2^7 - 1 for 8-bit signed
            optimal_w_scale = w_absmax / 127.0
            
            optimal_params[name] = {
                'act_scale': optimal_act_scale,
                'w_scale': optimal_w_scale,
                'y_int_target': y_int_target,  # For monitoring
            }
        
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
    
    def apply_calibration(self, optimal_params: Dict[str, Dict]):
        """
        Apply calibrated SCALES to the model.
        
        IMPORTANT: We do NOT change delta - it's a hardware constant!
        We only calibrate activation and weight scales to optimally use
        the fixed ADC range.
        """
        logger.info("Applying calibrated scales to model...")
        
        updated_act = 0
        updated_w = 0
        
        for name, module in self.model.named_modules():
            # Handle both QATLinearADC and TiledLinearADC tiles
            if isinstance(module, QATLinearADC) and name in optimal_params:
                params = optimal_params[name]
                
                with torch.no_grad():
                    # Update activation quantizer scale
                    if hasattr(module, 'activation_quantizer'):
                        old_scale = module.activation_quantizer.scale.item()
                        module.activation_quantizer.scale.copy_(
                            torch.tensor(params['act_scale'], dtype=torch.float32)
                        )
                        module.activation_quantizer._scale_initialized = True
                        
                        logger.info(f"{name} [ACT]: scale {old_scale:.6f} -> {params['act_scale']:.6f}")
                        updated_act += 1
                    
                    # Update weight quantizer scale (per-channel)
                    if hasattr(module, 'weight_quantizer'):
                        w_q = module.weight_quantizer
                        old_scale_mean = w_q.scale.mean().item() if w_q.scale.numel() > 0 else 0.01
                        
                        # For per-channel, compute scales from actual weights
                        if w_q.per_channel:
                            # Compute per-channel scales directly from weights
                            weight = module.weight.detach()  # [out_features, in_features]
                            
                            if w_q.channel_dim == 0:
                                # For each output channel, find absmax across input features
                                per_channel_absmax = weight.abs().max(dim=1)[0]  # [out_features]
                            else:
                                # For other channel dims
                                weight_transposed = weight.transpose(w_q.channel_dim, 0)
                                per_channel_absmax = weight_transposed.contiguous().view(weight_transposed.shape[0], -1).abs().max(dim=1)[0]
                            
                            # Avoid division by zero
                            per_channel_absmax = torch.clamp(per_channel_absmax, min=1e-6)
                            
                            # Compute per-channel scales for symmetric quantization
                            # scale = absmax / (2^(n-1) - 1) = absmax / 127
                            new_scales = per_channel_absmax / 127.0
                            
                            # Update scales
                            w_q.scale.data.copy_(new_scales)
                            w_q._scale_initialized = True
                        else:
                            # Per-tensor: use scalar scale
                            w_q.scale.copy_(
                                torch.tensor(params['w_scale'], dtype=torch.float32)
                            )
                            w_q._scale_initialized = True
                        
                        new_scale_mean = w_q.scale.mean().item()
                        
                        logger.info(f"{name} [W]: scale {old_scale_mean:.6f} -> {new_scale_mean:.6f}")
                        updated_w += 1
                    
                    # NOTE: ADC delta stays as analytical value - it's a hardware constant!
                    # We just ensure that with calibrated scales, y_int fits well in ADC range
        
        logger.info(f"Updated {updated_act} activation quantizers and {updated_w} weight quantizers")
        logger.info("NOTE: ADC delta values remain as hardware-defined constants")


def main():
    parser = argparse.ArgumentParser(description="PTQ for ADC-based BERT QA")
    
    # Model paths
    parser.add_argument("--qat_checkpoint_dir", type=str, required=True,
                       help="Path to QAT checkpoint (will be converted to ADC)")
    parser.add_argument("--output_dir", type=str, default="./outputs_adc_ptq",
                       help="Where to save calibrated model")
    parser.add_argument("--seed", type=int, default=42)
    
    # ADC settings
    parser.add_argument("--bx", type=int, default=8, help="Activation bits")
    parser.add_argument("--bw", type=int, default=8, help="Weight bits")
    parser.add_argument("--ba", type=int, default=8, help="ADC bits")
    parser.add_argument("--k", type=int, default=4, help="Hardware parameter")
    parser.add_argument("--ashift", action="store_true")
    parser.add_argument("--signed_activations", action="store_true",
                       help="Use signed activation quantization (RECOMMENDED)")
    parser.add_argument("--mvm_limit", type=int, default=256)
    
    # Calibration settings
    parser.add_argument("--calibration_method", type=str, default="percentile",
                       choices=["minmax", "percentile", "mse"],
                       help="Calibration method for ADC delta")
    parser.add_argument("--num_calibration_batches", type=int, default=100,
                       help="Number of batches for calibration")
    parser.add_argument("--calibration_batch_size", type=int, default=8)
    
    # Evaluation settings
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--doc_stride", type=int, default=128)
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    # Load checkpoint
    logger.info(f"Loading QAT checkpoint from: {args.qat_checkpoint_dir}")
    checkpoint_dir = find_last_checkpoint_dir(args.qat_checkpoint_dir)
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
    tokenizer.padding_side = "right"
    
    model = load_qa_model_robust(checkpoint_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Convert to ADC with analytical delta (will be calibrated)
    logger.info("Converting to ADC QAT layers...")
    model = BertADCConverter.replace_linear_with_adc_qat(
        model,
        bx=args.bx,
        bw=args.bw,
        ba=args.ba,
        k=args.k,
        ashift=args.ashift,
        signed_activations=args.signed_activations,
        exclude_patterns=["embeddings", "pooler", "qa_outputs"],
        mvm_limit=args.mvm_limit,
        use_dynamic_delta=False,  # PTQ: use fixed delta after calibration
        use_delta_anneal=False,
        delta_loss_weight=0.0,
    )
    model = model.to(device)
    
    stats = BertADCConverter.count_adc_qat_layers(model)
    logger.info(f"Model: {stats['adc_qat_linear']} ADC layers, {stats['total_params']:,} params")
    
    # Load calibration data
    logger.info("Loading SQuAD dataset for calibration...")
    raw = load_dataset("squad")
    
    # Use train split for calibration
    calibration_dataset = raw["train"].select(range(min(1000, len(raw["train"]))))
    calibration_dataset = calibration_dataset.map(
        lambda x: prepare_validation_features(x, tokenizer, args.max_length, args.doc_stride),
        batched=True,
        remove_columns=calibration_dataset.column_names,
        desc="Preparing calibration data",
    )
    
    # Custom collator to only take model inputs
    def calibration_collator(features):
        batch = {
            "input_ids": torch.tensor([f["input_ids"] for f in features]),
            "attention_mask": torch.tensor([f["attention_mask"] for f in features]),
        }
        if "token_type_ids" in features[0]:
            batch["token_type_ids"] = torch.tensor([f["token_type_ids"] for f in features])
        return batch
    
    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=args.calibration_batch_size,
        shuffle=False,
        collate_fn=calibration_collator,
    )
    
    # Run calibration
    logger.info("="*80)
    logger.info("STEP 1: CALIBRATION")
    logger.info("="*80)
    
    calibrator = ADCCalibrator(model, method=args.calibration_method)
    calibrator.calibrate(calibration_loader, num_batches=args.num_calibration_batches)
    
    # Compute optimal parameters
    logger.info("Computing optimal quantization scales...")
    optimal_params = calibrator.compute_optimal_params()
    
    # Apply calibration
    calibrator.apply_calibration(optimal_params)
    
    # Evaluate
    logger.info("="*80)
    logger.info("STEP 2: EVALUATION")
    logger.info("="*80)
    
    eval_examples = raw["validation"]
    eval_dataset_full = eval_examples.map(
        lambda x: prepare_validation_features(x, tokenizer, args.max_length, args.doc_stride),
        batched=True,
        remove_columns=eval_examples.column_names,
        desc="Preparing validation data",
    )
    
    # Custom collator for evaluation (only model inputs)
    def eval_collator(features):
        batch = {
            "input_ids": torch.tensor([f["input_ids"] for f in features]),
            "attention_mask": torch.tensor([f["attention_mask"] for f in features]),
        }
        if "token_type_ids" in features[0]:
            batch["token_type_ids"] = torch.tensor([f["token_type_ids"] for f in features])
        return batch
    
    eval_loader = DataLoader(
        eval_dataset_full,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=eval_collator,
    )
    
    logger.info("Running evaluation...")
    model.eval()
    
    all_start_logits = []
    all_end_logits = []
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            outputs = model(**batch)
            
            all_start_logits.append(outputs.start_logits.cpu().numpy())
            all_end_logits.append(outputs.end_logits.cpu().numpy())
    
    # Concatenate predictions
    all_start_logits = np.concatenate(all_start_logits, axis=0)
    all_end_logits = np.concatenate(all_end_logits, axis=0)
    
    # Post-process predictions (use full dataset with all metadata)
    formatted_predictions = postprocess_qa_predictions(
        examples=eval_examples,
        features=eval_dataset_full,
        predictions=(all_start_logits, all_end_logits),
    )
    
    # Compute metrics
    squad_metric = evaluate.load("squad")
    references = [{"id": ex_id, "answers": ans} 
                 for ex_id, ans in zip(eval_examples["id"], eval_examples["answers"])]
    predictions_for_metric = [{"id": k, "prediction_text": v} 
                              for k, v in formatted_predictions.items()]
    
    eval_metrics = squad_metric.compute(
        predictions=predictions_for_metric,
        references=references
    )
    
    logger.info("="*80)
    logger.info("RESULTS")
    logger.info("="*80)
    logger.info(f"F1 Score:      {eval_metrics['f1']:.2f}")
    logger.info(f"Exact Match:   {eval_metrics['exact_match']:.2f}")
    
    # Save calibrated model
    logger.info(f"Saving calibrated model to: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Save calibration info
    with open(os.path.join(args.output_dir, "calibration_info.txt"), "w") as f:
        f.write("="*80 + "\n")
        f.write("ADC POST-TRAINING QUANTIZATION (PTQ) CALIBRATION RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Calibration method: {args.calibration_method}\n")
        f.write(f"Calibration batches: {args.num_calibration_batches}\n")
        f.write(f"ADC hardware config: bx={args.bx}, bw={args.bw}, ba={args.ba}, k={args.k}\n")
        f.write(f"Signed activations: {args.signed_activations}\n")
        f.write(f"A-shift: {args.ashift}\n")
        f.write(f"\n")
        f.write("NOTE: ADC delta is a HARDWARE CONSTANT and cannot be changed!\n")
        f.write("      We calibrate activation/weight SCALES to optimally use the fixed ADC range.\n")
        f.write(f"\n")
        f.write(f"Results:\n")
        f.write(f"  F1:          {eval_metrics['f1']:.2f}\n")
        f.write(f"  Exact Match: {eval_metrics['exact_match']:.2f}\n")
        f.write(f"\n")
        f.write(f"Calibrated layers: {len(optimal_params)}\n")
        f.write(f"\n")
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
    
    logger.info("="*80)
    logger.info("PTQ COMPLETE!")
    logger.info("="*80)
    logger.info(f"Calibrated model saved to: {args.output_dir}")
    logger.info(f"F1: {eval_metrics['f1']:.2f}, EM: {eval_metrics['exact_match']:.2f}")


if __name__ == "__main__":
    main()

