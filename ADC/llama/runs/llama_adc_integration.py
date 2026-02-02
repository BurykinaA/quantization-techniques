import argparse
import os
import time
import collections
import logging
import random
from typing import Dict, Any, Optional, List, Tuple
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import evaluate

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    BertForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
    set_seed,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)

from ADC.llama.core.adc_layers import (
    QATLinearADC, 
    TiledLinearADC, 
    LearnableQuantizer,
    LoRAQATLinearADC,
    LoRATiledLinearADC,
)  # noqa: F401

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl


from .adc_monitoring_integration import create_adc_training_monitor, add_adc_monitoring_to_model
ADC_MONITORING_AVAILABLE = True


# WandB import (same as PTQ script)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, logging will be disabled")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) 


def get_bitaug_neighbors(target_ba: int, neighbor_range: int = 1, min_bits: int = 4, max_bits: int = 12) -> list[int]:
    """
    Get neighboring bit precisions for BitAug (Paper Section: Bit Augmentation).
    
    Following paper insights: neighbors should be close to target to avoid adding noise.
    
    Args:
        target_ba: Target ADC bit precision
        neighbor_range: How many bits above/below target to include (default: 1 for ±1 bits)
        min_bits: Minimum allowed bit precision
        max_bits: Maximum allowed bit precision
        
    Returns:
        List of valid neighboring bit precisions (excluding target itself)
    """
    neighbors = []
    for offset in range(-neighbor_range, neighbor_range + 1):
        if offset == 0:
            continue  # Skip target itself
        bit = target_ba + offset
        if min_bits <= bit <= max_bits:
            neighbors.append(bit)
    return neighbors


def set_model_adc_bits(model: nn.Module, ba: int):
    """
    Set ADC bit precision for all ADC layers in the model.
    
    Args:
        model: Model containing TiledLinearADC/QATLinearADC layers
        ba: New ADC bit precision
    """
    for module in model.modules():
        if isinstance(module, (TiledLinearADC, QATLinearADC, LoRATiledLinearADC)):
            if hasattr(module, 'set_adc_bits'):
                module.set_adc_bits(ba)


def warmup_lora_mse(
    model: nn.Module,
    dataloader,
    num_steps: int = 100,
    lr: float = 1e-3,
    device: str = 'cuda'
) -> float:
    """
    Warmup LoRA parameters using MSE optimization (Paper Equation 12).
    
    Minimizes: ||Qx(X)Qw(W) - Y||^2_F
    where Y = QA(Qx(X)Qw(W + AB))
    
    This finds A, B such that the ADC-quantized output with LoRA
    matches the quantized output WITHOUT ADC as closely as possible.
    The goal is to compensate for ADC quantization error via LoRA.
    
    Args:
        model: Model with LoRA layers (LoRATiledLinearADC)
        dataloader: DataLoader for warmup data
        num_steps: Number of optimization steps
        lr: Learning rate for warmup optimization
        device: Device to run on
        
    Returns:
        Final average MSE loss
    """
    # Collect LoRA parameters
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            param.requires_grad = True
            lora_params.append(param)
    
    if not lora_params:
        logger.warning("No LoRA parameters found for warmup")
        return 0.0
    
    # Collect LoRA layers for per-layer MSE computation
    lora_layers = []
    lora_layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, (LoRATiledLinearADC, LoRAQATLinearADC)):
            lora_layers.append(module)
            lora_layer_names.append(name)
    
    if not lora_layers:
        logger.warning("No LoRA layers found for warmup")
        return 0.0
    
    logger.info(f"LoRA MSE warmup (Eq. 12): {len(lora_params)} params, "
               f"{len(lora_layers)} layers, {num_steps} steps, lr={lr}")
    
    optimizer = torch.optim.Adam(lora_params, lr=lr)
    model.train()
    
    # Hook storage for layer inputs/outputs
    layer_inputs = {}
    
    def make_input_hook(layer_name):
        def hook(module, args, kwargs):
            # Store the input tensor for this layer
            if len(args) > 0:
                layer_inputs[layer_name] = args[0].detach().clone()
        return hook
    
    # Register forward pre-hooks to capture inputs
    hooks = []
    for name, layer in zip(lora_layer_names, lora_layers):
        hook = layer.register_forward_pre_hook(make_input_hook(name), with_kwargs=True)
        hooks.append(hook)
    
    total_loss = 0.0
    step = 0
    
    try:
        for batch in dataloader:
            if step >= num_steps:
                break
            
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            optimizer.zero_grad()
            layer_inputs.clear()
            
            # Forward pass to capture layer inputs
            with torch.no_grad():
                _ = model(**batch)
            
            # Now compute per-layer MSE loss (Equation 12)
            # ||Qx(X)Qw(W) - QA(Qx(X)Qw(W + AB))||^2_F
            mse_loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            for name, layer in zip(lora_layer_names, lora_layers):
                if name not in layer_inputs:
                    continue
                
                x = layer_inputs[name].requires_grad_(False)
                
                # Reference: Qx(X) @ Qw(W) - no ADC, no LoRA
                with torch.no_grad():
                    ref_output = layer.compute_reference_output(x)
                
                # LoRA output: QA(Qx(X) @ Qw(W + AB)) - with ADC, with LoRA
                lora_output = layer(x)
                
                # MSE loss for this layer (Frobenius norm squared)
                layer_mse = ((ref_output - lora_output) ** 2).mean()
                mse_loss = mse_loss + layer_mse
            
            mse_loss.backward()
            optimizer.step()
            
            total_loss += mse_loss.item()
            step += 1
            
            if step % 20 == 0:
                logger.info(f"LoRA warmup step {step}/{num_steps}, MSE loss: {mse_loss.item():.6f}")
    
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()
    
    avg_loss = total_loss / max(step, 1)
    logger.info(f"LoRA MSE warmup complete. Average MSE loss: {avg_loss:.6f}")
    
    return avg_loss


def get_lora_param_count(model: nn.Module) -> dict:
    """
    Count trainable and total parameters in a model with LoRA.
    
    Returns:
        Dictionary with 'lora_params', 'frozen_params', 'total_params', 'compression_ratio'
    """
    lora_params = 0
    frozen_params = 0
    trainable_other = 0
    
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            lora_params += param.numel()
        elif not param.requires_grad:
            frozen_params += param.numel()
        else:
            trainable_other += param.numel()
    
    total_trainable = lora_params + trainable_other
    total_params = lora_params + frozen_params + trainable_other
    
    # Compression ratio: how many times fewer trainable params vs full fine-tuning
    full_trainable = frozen_params + lora_params + trainable_other  # If nothing was frozen
    compression_ratio = full_trainable / total_trainable if total_trainable > 0 else 1.0
    
    return {
        'lora_params': lora_params,
        'frozen_params': frozen_params,
        'trainable_other': trainable_other,
        'total_trainable': total_trainable,
        'total_params': total_params,
        'compression_ratio': compression_ratio,
    }


class ADCLossTrainer(Trainer):
    """
    Custom trainer that handles:
    - ADC auxiliary losses (kurtosis/W-reshape) from get_auxiliary_losses()
    - BitAug: Bit Augmentation for improved training (Paper Equation 10)
    """

    def __init__(
        self, 
        *args, 
        adc_step_monitor=None,
        # BitAug parameters (Paper: Bit Augmentation)
        use_bitaug: bool = False,
        bitaug_lambda: float = 0.5,
        target_ba: int = 8,
        bitaug_neighbors: list[int] | None = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.adc_step_monitor = adc_step_monitor
        
        # BitAug configuration
        self.use_bitaug = use_bitaug
        self.bitaug_lambda = bitaug_lambda
        self.target_ba = target_ba
        
        # Get neighbors if not provided
        if bitaug_neighbors is None and use_bitaug:
            self.bitaug_neighbors = get_bitaug_neighbors(target_ba, neighbor_range=1)
        else:
            self.bitaug_neighbors = bitaug_neighbors or []
        
        if use_bitaug:
            logger.info(f"BitAug enabled: target_ba={target_ba}, lambda={bitaug_lambda}, "
                       f"neighbors={self.bitaug_neighbors}")

    def _compute_task_loss(self, model, inputs) -> tuple[torch.Tensor, any]:
        """
        Compute task-specific loss (QA loss for BERT).
        
        Returns:
            Tuple of (loss, outputs)
        """
        outputs = model(**inputs)
        
        if hasattr(outputs, 'loss'):
            loss = outputs.loss
        elif isinstance(outputs, tuple):
            loss = outputs[0]
        elif isinstance(outputs, dict):
            loss = outputs.get('loss', None)
        else:
            loss = None
        
        if loss is None:
            raise ValueError("Could not extract loss from model outputs")
        
        return loss, outputs

    def _compute_auxiliary_loss(self, model, device, dtype) -> torch.Tensor:
        """
        Aggregate auxiliary losses from all ADC layers.
        Includes kurtosis loss (W-reshape, Paper Eq. 6 & 7).
        """
        auxiliary_loss = torch.tensor(0.0, device=device, dtype=dtype)
        
        for module in model.modules():
            if isinstance(module, (TiledLinearADC, QATLinearADC)):
                if hasattr(module, 'get_auxiliary_losses'):
                    try:
                        aux_losses = module.get_auxiliary_losses()
                        if 'total' in aux_losses:
                            layer_loss = aux_losses['total']
                            if isinstance(layer_loss, torch.Tensor):
                                auxiliary_loss = auxiliary_loss + layer_loss.to(device)
                    except Exception:
                        pass
        
        return auxiliary_loss

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute total loss with optional BitAug (Paper Equation 10).
        
        BitAug loss formula:
            L_A = L(θ, ba) + λ_b * L(θ, ẽba)
        
        where:
            - L(θ, ba) is the loss with target ADC bit precision
            - L(θ, ẽba) is the loss with a randomly sampled neighbor bit precision
            - λ_b is the BitAug coefficient
        """
        # === Step 1: Forward pass with target ADC bits ===
        main_loss, outputs = self._compute_task_loss(model, inputs)
        auxiliary_loss = self._compute_auxiliary_loss(model, main_loss.device, main_loss.dtype)
        
        total_loss = main_loss + auxiliary_loss
        
        # === Step 2: BitAug - forward pass with augmented bit precision ===
        if self.use_bitaug and model.training and self.bitaug_neighbors:
            # Randomly sample one bit precision from neighbors (Paper Eq. 10)
            aug_ba = random.choice(self.bitaug_neighbors)
            
            # Temporarily change ADC bits
            set_model_adc_bits(model, aug_ba)
            
            # Forward pass with augmented bits (no need to keep outputs)
            aug_loss, _ = self._compute_task_loss(model, inputs)
            
            # Add BitAug loss component (task loss only)
            total_loss += self.bitaug_lambda * aug_loss
            
            # Restore target ADC bits
            set_model_adc_bits(model, self.target_ba)

        # Call ADC monitoring after each batch
        if self.adc_step_monitor:
            try:
                self.adc_step_monitor()
            except Exception as e:
                print(f"ADC monitoring error: {e}")

        return (total_loss, outputs) if return_outputs else total_loss


class EvalMetricsLogger(TrainerCallback):
    """Callback to log F1/EM at each evaluation."""

    def __init__(self, use_wandb: bool = False):
        self.use_wandb = use_wandb

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        try:
            if metrics is None:
                logger.warning("EvalMetricsLogger: metrics is None")
                return
            
            # Debug: print all available metrics keys
            logger.info(f"EvalMetricsLogger: Available metric keys: {list(metrics.keys())}")
            
            f1 = metrics.get("eval_f1", metrics.get("f1"))
            em = metrics.get("eval_exact_match", metrics.get("exact_match"))
            
            if f1 is not None and em is not None:
                logger.info(f"Eval F1: {float(f1):.2f}, EM: {float(em):.2f}")
                if self.use_wandb and wandb.run is not None:
                    wandb.log(
                        {
                            "eval/f1": float(f1),
                            "eval/exact_match": float(em),
                        },
                        step=state.global_step,
                    )
            else:
                logger.warning(f"EvalMetricsLogger: F1 or EM not found. f1={f1}, em={em}")
        except Exception as e:
            logger.error(f"EvalMetricsLogger error: {e}")
            import traceback
            traceback.print_exc()

class BertADCConverter:
    """Convert BERT model to use ADC QAT layers for QA."""

    @staticmethod
    def replace_linear_with_adc_qat(
        model: nn.Module,
        bx: int = 8,  # Activation bits
        bw: int = 8,  # Weight bits
        ba: int = 8,  # ADC bits
        k: int = 4,   # Hardware design parameter
        ashift: bool = False,
        exclude_patterns: Optional[List[str]] = None,
        mvm_limit: int = 256,  # Default to 256 to match CLI default
        # W-reshape (kurtosis) parameters from paper Equation 6 & 7
        use_kurtosis_loss: bool = True,
        kurtosis_weight: float = 0.0006,
        target_kurtosis: float = 1.8,
        # LoRA parameters (Paper Section 3.4: Training Overhead Reduction)
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        lora_target_modules: Optional[List[str]] = None,
    ) -> nn.Module:
        """
        Replace all nn.Linear layers in the model with TiledLinearADC, except excluded.

        Args:
            model: Model to convert.
            bx: Bits for activation quantization.
            bw: Bits for weight quantization.
            ba: Bits for ADC quantization.
            k: Hardware design parameter for ADC.
            ashift: Enable A-shift for layers after GeLU. When True, layer.X.output.dense
                    uses asymmetric quantization + A-shift, all others use symmetric.
            exclude_patterns: List of substrings of module names to exclude.
            mvm_limit: Memory vector multiplication limit for tiling.
            use_lora: Enable ADC-LoRA for reduced trainable parameters.
            lora_r: LoRA rank (dimension of low-rank matrices).
            lora_alpha: LoRA scaling factor (scaling = alpha / r).
            lora_dropout: Dropout rate for LoRA path.
            lora_target_modules: List of module name patterns to apply LoRA to.
                                 Default: ["query", "value"] (standard for transformers).
            
        Note:
            signed_activations is now determined per-layer automatically:
            - Layers after GeLU (when ashift=True): asymmetric/unsigned
            - All other layers: symmetric/signed
        """
        if exclude_patterns is None:
            # Default: don't quantize embeddings/pooler and QA output head unless requested
            exclude_patterns = ["embeddings", "pooler", "qa_outputs"]
        
        if lora_target_modules is None:
            # Default LoRA targets: query and value projections (standard for transformers)
            lora_target_modules = ["query", "value"]

        def should_exclude(name: str) -> bool:
            return any(pat in name for pat in exclude_patterns)
        
        def should_apply_lora(name: str) -> bool:
            """Check if this layer should have LoRA applied."""
            if not use_lora:
                return False
            return any(target in name for target in lora_target_modules)
        
        def is_after_gelu(name: str) -> bool:
            """Detect if this linear layer follows a GeLU. Only used for logging now."""
            return "output.dense" in name and "layer." in name

        def replace_recursive(module: nn.Module, name: str = ""):
            for child_name, child_module in module.named_children():
                full_name = f"{name}.{child_name}" if name else child_name

                if isinstance(child_module, nn.Linear) and not should_exclude(full_name):
                    # Apply A-shift ONLY to layers that receive GeLU outputs
                    layer_ashift = ashift and is_after_gelu(full_name)
                    # A-shift requires asymmetric quantization, others use symmetric
                    layer_signed_activations = not layer_ashift
                    
                    adc_qat_layer = TiledLinearADC(
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
                    # Use the load_weights method instead of manual copying
                    adc_qat_layer.load_weights(child_module)
                    
                    # Wrap with LoRA if this layer should have LoRA applied
                    if should_apply_lora(full_name):
                        adc_qat_layer = LoRATiledLinearADC(
                            tiled_layer=adc_qat_layer,
                            r=lora_r,
                            alpha=lora_alpha,
                            dropout=lora_dropout,
                        )
                        quant_type = "A-shift (asymmetric)" if layer_ashift else "symmetric"
                        logger.info(f"Replaced {full_name} with LoRATiledLinearADC "
                                   f"({quant_type}, bx={bx}, bw={bw}, ba={ba}, k={k}, "
                                   f"lora_r={lora_r}, lora_alpha={lora_alpha})")
                    else:
                        quant_type = "A-shift (asymmetric)" if layer_ashift else "symmetric"
                        logger.info(f"Replaced {full_name} with TiledLinearADC ({quant_type}, bx={bx}, bw={bw}, ba={ba}, k={k})")
                    
                    setattr(module, child_name, adc_qat_layer)
                else:
                    replace_recursive(child_module, full_name)

        replace_recursive(model)
        return model

    @staticmethod
    def count_adc_qat_layers(model: nn.Module) -> Dict[str, int]:
        counts = {"adc_qat_linear": 0, "lora_linear": 0, "regular_linear": 0, "total_params": 0, "lora_params": 0}
        for _, module in model.named_modules():
            if isinstance(module, (LoRATiledLinearADC, LoRAQATLinearADC)):
                counts["lora_linear"] += 1
                counts["lora_params"] += module.get_num_trainable_params()
            elif isinstance(module, (QATLinearADC, TiledLinearADC)):
                counts["adc_qat_linear"] += 1
            elif isinstance(module, nn.Linear):
                counts["regular_linear"] += 1
            if hasattr(module, "parameters"):
                counts["total_params"] += sum(p.numel() for p in module.parameters())
        return counts


def find_last_checkpoint_dir(fp_output_dir: str) -> str:
    """
    Given a base fine-tuning output directory, pick the latest 'checkpoint-*' subdir.
    If fp_output_dir itself is a checkpoint dir, return it. If none found, return base dir.
    """
    if not os.path.isdir(fp_output_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {fp_output_dir}")

    base = os.path.basename(fp_output_dir.rstrip("/"))
    if base.startswith("checkpoint-"):
        return fp_output_dir

    candidates = []
    for name in os.listdir(fp_output_dir):
        path = os.path.join(fp_output_dir, name)
        if os.path.isdir(path) and name.startswith("checkpoint-"):
            try:
                step = int(name.split("-")[-1])
            except Exception:
                step = -1
            candidates.append((step, path))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[-1][1]

    # Fallback: use the directory itself if there is no sub-checkpoint
    return fp_output_dir


# ====== SQuAD QA pipeline helpers (same as FP script) ======
def load_qa_model_robust(checkpoint_dir: str) -> BertForQuestionAnswering:
    """
    Load BertForQuestionAnswering from a checkpoint that may contain extra
    keys from QAT layers. Falls back to strict=False state_dict load.
    """
    try:
        return BertForQuestionAnswering.from_pretrained(checkpoint_dir)
    except Exception as e:
        logger.warning(f"Standard from_pretrained failed, retrying with strict=False. Error: {e}")
        state_path = os.path.join(checkpoint_dir, 'pytorch_model.bin')
        state = torch.load(state_path, map_location='cpu')
        # Build fresh model from config and load weights with strict=False (ignore extra QAT keys)
        config = AutoConfig.from_pretrained(checkpoint_dir)
        model = BertForQuestionAnswering(config)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            logger.info(f"Ignored {len(unexpected)} unexpected keys (likely QAT quantizer params).")
        if missing:
            logger.info(f"Missing keys count: {len(missing)} (randomly initialized).")
        return model

def load_state_dict_flexible(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> Tuple[List[str], List[str]]:
    """
    Load state dict with flexible shape handling for quantizer parameters.
    Handles per-channel quantizer scales that need to be resized.
    Also handles LoRA key remapping (e.g., query.tiles.0 -> query.tiled_layer.tiles.0).
    
    Returns:
        Tuple of (missing_keys, unexpected_keys)
    """
    model_state = model.state_dict()
    missing_keys = []
    unexpected_keys = []
    loaded_keys = set()
    remapped_count = 0
    
    # Build a mapping for LoRA key remapping
    # When LoRA wraps TiledLinearADC, path changes from .tiles. to .tiled_layer.tiles.
    def remap_key_for_lora(key: str) -> str:
        """Try to remap checkpoint key to model key for LoRA layers."""
        # Check if the key needs LoRA remapping
        # Pattern: something.tiles.X.something -> something.tiled_layer.tiles.X.something
        if '.tiles.' in key and '.tiled_layer.' not in key:
            # Find modules that might be LoRA-wrapped in the model
            # Try adding .tiled_layer. before .tiles.
            parts = key.split('.tiles.')
            if len(parts) == 2:
                remapped = parts[0] + '.tiled_layer.tiles.' + parts[1]
                if remapped in model_state:
                    return remapped
        return key
    
    # Create a remapped state dict
    remapped_state_dict = {}
    for key, tensor in state_dict.items():
        remapped_key = remap_key_for_lora(key)
        if remapped_key != key:
            remapped_count += 1
        remapped_state_dict[remapped_key] = tensor
    
    if remapped_count > 0:
        logger.info(f"Remapped {remapped_count} keys for LoRA compatibility")
    
    # Now load using standard approach with strict=False
    # First, handle shape mismatches for quantizer parameters
    for key, checkpoint_tensor in remapped_state_dict.items():
        if key not in model_state:
            unexpected_keys.append(key)
            continue
        
        model_shape = model_state[key].shape
        
        if checkpoint_tensor.shape != model_shape:
            # Handle shape mismatch for quantizer parameters
            if 'quantizer.scale' in key or 'quantizer.zero_point' in key:
                # Find the actual parameter in the model and resize it
                try:
                    param = model
                    for attr in key.split('.'):
                        param = getattr(param, attr)
                    
                    with torch.no_grad():
                        if isinstance(param, nn.Parameter):
                            param.data = checkpoint_tensor.clone()
                        elif torch.is_tensor(param):
                            param.copy_(checkpoint_tensor)
                    loaded_keys.add(key)
                except Exception as e:
                    logger.warning(f"Failed to load {key}: {e}")
                    missing_keys.append(key)
            else:
                missing_keys.append(key)
                logger.warning(f"Shape mismatch for {key}: checkpoint {checkpoint_tensor.shape} vs model {model_shape}")
        else:
            loaded_keys.add(key)
    
    # Now do the actual load for matching keys
    # Filter state dict to only include keys that exist in model and have matching shapes
    filtered_state_dict = {}
    for key, tensor in remapped_state_dict.items():
        if key in model_state and key not in missing_keys:
            if tensor.shape == model_state[key].shape or key in loaded_keys:
                filtered_state_dict[key] = tensor
    
    # Load the filtered state dict
    load_result = model.load_state_dict(filtered_state_dict, strict=False)
    
    # Combine missing/unexpected with load_result
    for key in load_result.missing_keys:
        if key not in missing_keys and key not in loaded_keys:
            missing_keys.append(key)
    for key in load_result.unexpected_keys:
        if key not in unexpected_keys:
            unexpected_keys.append(key)
    
    # Check for missing keys (parameters in model but not in checkpoint)
    for key in model_state.keys():
        if key not in remapped_state_dict and key not in missing_keys:
            # Only add if not a LoRA parameter (those are expected to be missing)
            if 'lora_A' not in key and 'lora_B' not in key:
                missing_keys.append(key)
    
    logger.info(f"Loaded {len(filtered_state_dict)} keys into model")
    
    return missing_keys, unexpected_keys


def warm_start_adc_quantizers_from_qat(model: nn.Module, checkpoint_dir: str) -> int:
    """
    Initialize ADC quantizers (per-tile LearnableQuantizer scales/zero-points)
    from a QAT checkpoint's quantizer parameters. Returns number of modules updated.
    """
    # Support both .bin/.safetensors (single) and sharded variants
    state: Dict[str, torch.Tensor]
    bin_path = os.path.join(checkpoint_dir, 'pytorch_model.bin')
    safetensors_path = os.path.join(checkpoint_dir, 'model.safetensors')
    bin_index = os.path.join(checkpoint_dir, 'pytorch_model.bin.index.json')
    safe_index = os.path.join(checkpoint_dir, 'model.safetensors.index.json')
    state = {}
    if os.path.exists(bin_path):
        state = torch.load(bin_path, map_location='cpu')
    elif os.path.exists(safetensors_path):
        try:
            from safetensors.torch import load_file as safe_load_file  # type: ignore
        except Exception as e:
            logger.info(f"Safetensors present but not loadable ({e}). Skipping warm-start.")
            return 0
        state = safe_load_file(safetensors_path)
    elif os.path.exists(bin_index):
        try:
            with open(bin_index, 'r') as f:
                index = json.load(f)
            weight_map: Dict[str, str] = index.get('weight_map', {})
            shard_files = sorted(set(weight_map.values()))
            for shard in shard_files:
                shard_path = os.path.join(checkpoint_dir, shard)
                if os.path.exists(shard_path):
                    shard_sd = torch.load(shard_path, map_location='cpu')
                    state.update({k: v for k, v in shard_sd.items() if 'quantizer' in k})
        except Exception as e:
            logger.info(f"Failed loading sharded bin checkpoint for warm-start: {e}")
            return 0
    elif os.path.exists(safe_index):
        try:
            with open(safe_index, 'r') as f:
                index = json.load(f)
            weight_map: Dict[str, str] = index.get('weight_map', {})
            shard_files = sorted(set(weight_map.values()))
            from safetensors import safe_open  # type: ignore
            for shard in shard_files:
                shard_path = os.path.join(checkpoint_dir, shard)
                if os.path.exists(shard_path):
                    with safe_open(shard_path, framework="pt", device="cpu") as f:
                        for k in f.keys():
                            if 'quantizer' in k:
                                state[k] = f.get_tensor(k)
        except Exception as e:
            logger.info(f"Failed loading sharded safetensors checkpoint for warm-start: {e}")
            return 0
    else:
        logger.info("No state dict found to warm-start quantizers.")
        return 0

    # Build map from base module name -> quantizer params
    quant_map: Dict[str, Dict[str, torch.Tensor]] = {}
    def get_entry(base: str) -> Dict[str, torch.Tensor]:
        if base not in quant_map:
            quant_map[base] = {}
        return quant_map[base]

    for key, tensor in state.items():
        if key.endswith('.activation_quantizer.scale'):
            base = key[: -len('.activation_quantizer.scale')]
            get_entry(base)['act_scale'] = tensor
        elif key.endswith('.activation_quantizer.zero_point'):
            base = key[: -len('.activation_quantizer.zero_point')]
            get_entry(base)['act_zp'] = tensor
        elif key.endswith('.weight_quantizer.scale'):
            base = key[: -len('.weight_quantizer.scale')]
            get_entry(base)['w_scale'] = tensor
        elif key.endswith('.weight_quantizer.zero_point'):
            base = key[: -len('.weight_quantizer.zero_point')]
            get_entry(base)['w_zp'] = tensor

    def _safe_copy_(dst: torch.nn.Parameter, src: torch.Tensor) -> None:
        """Copy with shape safety: match, expand scalar, or reduce by mean."""
        with torch.no_grad():
            if dst.shape == src.shape:
                dst.copy_(src.to(dst.device).type_as(dst))
                return
            # Scalar source -> expand
            if src.numel() == 1 and dst.numel() > 1:
                dst.copy_(src.to(dst.device).type_as(dst).expand_as(dst))
                return
            # Larger source -> reduce to first dim if 1D dst
            if dst.dim() == 1 and src.numel() > dst.numel():
                reduced = src.to(dst.device).type_as(dst)
                # If first dim matches, take slice; else mean over all dims
                if reduced.dim() > 0 and reduced.shape[0] >= dst.shape[0]:
                    reduced = reduced.reshape(-1)[: dst.shape[0]]
                else:
                    reduced = reduced.reshape(-1).mean().expand_as(dst)
                dst.copy_(reduced)
                return
            # Fallback: broadcast if possible, else do mean
            try:
                dst.copy_(src.to(dst.device).type_as(dst))
            except Exception:
                dst.copy_(src.to(dst.device).type_as(dst).reshape(-1).mean().expand_as(dst))

    updated = 0
    # Assign into each TiledLinearADC's tiles
    for name, module in model.named_modules():
        if isinstance(module, TiledLinearADC):
            base_name = name  # matches original linear path in checkpoint
            if base_name in quant_map:
                params = quant_map[base_name]
                for tile in module.tiles:
                    # Activation quantizer (per-tensor, likely asymmetric)
                    if hasattr(tile, 'activation_quantizer'):
                        aq = tile.activation_quantizer
                        if 'act_scale' in params:
                            _safe_copy_(aq.scale, params['act_scale'])
                        if hasattr(aq, 'zero_point') and 'act_zp' in params:
                            # Only copy if not symmetric
                            if not getattr(aq, 'symmetric', False):
                                _safe_copy_(aq.zero_point, params['act_zp'])
                        if hasattr(aq, '_scale_initialized'):
                            aq._scale_initialized = True
                        if hasattr(aq, '_zp_initialized'):
                            aq._zp_initialized = True

                    # Weight quantizer (per-channel symmetric)
                    if hasattr(tile, 'weight_quantizer'):
                        wq = tile.weight_quantizer
                        if 'w_scale' in params:
                            _safe_copy_(wq.scale, params['w_scale'])
                        # Skip copying zero_point for symmetric weight quantizer
                        if hasattr(wq, 'zero_point') and 'w_zp' in params and not getattr(wq, 'symmetric', True):
                            _safe_copy_(wq.zero_point, params['w_zp'])
                        if hasattr(wq, '_scale_initialized'):
                            wq._scale_initialized = True
                        if hasattr(wq, '_zp_initialized'):
                            wq._zp_initialized = True

                updated += 1

    logger.info(f"Warm-started quantizers for {updated} TiledLinearADC modules from QAT checkpoint.")
    return updated

 
def prepare_train_features(examples, tokenizer, max_length=384, doc_stride=128):
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offsets_mapping = tokenized["offset_mapping"]  # Keep offset_mapping for compute_metrics

    tokenized["start_positions"] = []
    tokenized["end_positions"] = []
    tokenized["example_id"] = []  # Add example_id for compute_metrics

    for i, offsets in enumerate(offsets_mapping):
        input_ids = tokenized["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = tokenized.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        
        # Store example_id for mapping back to original examples
        tokenized["example_id"].append(examples["id"][sample_index])

        # Mask offset_mapping to only include context tokens (needed for postprocessing)
        tokenized["offset_mapping"][i] = [
            (o if sequence_ids[k] == 1 else None)  # context_index = 1
            for k, o in enumerate(offsets)
        ]

        if len(answers["answer_start"]) == 0:
            tokenized["start_positions"].append(cls_index)
            tokenized["end_positions"].append(cls_index)
            continue

        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])

        context_index = 1

        token_start_index = 0
        while sequence_ids[token_start_index] != context_index:
            token_start_index += 1
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != context_index:
            token_end_index -= 1

        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            tokenized["start_positions"].append(cls_index)
            tokenized["end_positions"].append(cls_index)
        else:
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char and sequence_ids[token_start_index] == context_index:
                token_start_index += 1
            start_position = token_start_index - 1

            while offsets[token_end_index][1] >= end_char and sequence_ids[token_end_index] == context_index:
                token_end_index -= 1
            end_position = token_end_index + 1

            tokenized["start_positions"].append(start_position)
            tokenized["end_positions"].append(end_position)

    return tokenized


def prepare_validation_features(examples, tokenizer, max_length=384, doc_stride=128):
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    tokenized["example_id"] = []

    for i in range(len(tokenized["input_ids"])):
        sequence_ids = tokenized.sequence_ids(i)
        context_index = 1

        tokenized["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized["offset_mapping"][i])
        ]

        sample_index = sample_mapping[i]
        tokenized["example_id"].append(examples["id"][sample_index])

    return tokenized


def postprocess_qa_predictions(
    examples,
    features,
    predictions,
    n_best_size=20,
    max_answer_length=30,
):
    all_start_logits, all_end_logits = predictions
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feat_id in enumerate(features["example_id"]):
        features_per_example[feat_id].append(i)

    predictions_dict = {}

    for example_id, feature_indices in features_per_example.items():
        context = examples["context"][example_id_to_index[example_id]]
        prelim_predictions = []

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features["offset_mapping"][feature_index]

            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    prelim_predictions.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "start": start_char,
                            "end": end_char,
                        }
                    )

        if len(prelim_predictions) == 0:
            predictions_dict[example_id] = ""
            continue

        best_pred = max(prelim_predictions, key=lambda x: x["score"])
        predictions_dict[example_id] = context[best_pred["start"] : best_pred["end"]]

    return predictions_dict


class MetricsComputer:
    def __init__(self, eval_examples, eval_dataset, tokenizer, squad_metric):
        self.eval_examples = eval_examples
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.squad_metric = squad_metric

    def compute_metrics(self, eval_pred):
        try:
            logger.info("MetricsComputer.compute_metrics: Starting metric computation...")
            predictions, _ = eval_pred
            formatted_predictions = postprocess_qa_predictions(
                examples=self.eval_examples,
                features=self.eval_dataset,
                predictions=predictions,
            )
            references = [{"id": ex_id, "answers": ans} for ex_id, ans in zip(self.eval_examples["id"], self.eval_examples["answers"])]
            predictions_for_metric = [{"id": k, "prediction_text": v} for k, v in formatted_predictions.items()]
            result = self.squad_metric.compute(predictions=predictions_for_metric, references=references)
            logger.info(f"MetricsComputer.compute_metrics: Computed F1={result['f1']:.2f}, EM={result['exact_match']:.2f}")
            return {"f1": result["f1"], "exact_match": result["exact_match"]}
        except Exception as e:
            logger.error(f"MetricsComputer.compute_metrics: Error - {e}")
            import traceback
            traceback.print_exc()
            return {"f1": 0.0, "exact_match": 0.0}


def add_gradient_hooks(model):
    """Add hooks to monitor gradients and detect infinite gradients.
    
    Also monitors quantizer scale parameters to verify STE gradient flow is working.
    """
    # Track scale gradient statistics for verification
    scale_grad_stats = {"count": 0, "nonzero_count": 0, "total_norm": 0.0}
    
    def grad_hook(name):
        def hook(grad):
            if grad is not None:
                grad_norm = grad.norm().item()
                has_nan = torch.isnan(grad).any()
                has_inf = torch.isinf(grad).any()
                
                # Track quantizer scale gradients specifically (verify STE fix)
                if "quantizer.scale" in name:
                    scale_grad_stats["count"] += 1
                    if grad_norm > 1e-10:
                        scale_grad_stats["nonzero_count"] += 1
                    scale_grad_stats["total_norm"] += grad_norm
                    
                    # Log first few scale gradients to verify they're non-zero
                    if scale_grad_stats["count"] <= 5:
                        logger.info(f"Scale gradient [{name}]: norm={grad_norm:.6e}")
                
                # Log at a slightly higher threshold to reduce noise; still clip
                if has_nan or has_inf or grad_norm > 500:
                    # More aggressive gradient clipping
                    if has_nan or has_inf:
                        grad = torch.nan_to_num(grad, nan=0.0, posinf=10.0, neginf=-10.0)
                    elif grad_norm > 500:
                        grad = grad / (grad_norm / 50.0)  # Scale down to norm≈50
                        
            return grad
        return hook
    
    # Add hooks to all parameters
    scale_param_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.register_hook(grad_hook(name))
            if "quantizer.scale" in name:
                scale_param_count += 1
    
    logger.info(f"Added gradient monitoring hooks to all parameters")
    logger.info(f"Monitoring {scale_param_count} quantizer scale parameters for STE gradient verification")


def debug_scale_values(model):
    """Log statistics about quantizer scale values to diagnose NaN issues."""
    scale_stats = {
        "total": 0,
        "nan": 0,
        "inf": 0, 
        "zero": 0,
        "tiny": 0,  # < 1e-6
        "negative": 0,
    }
    problem_scales = []
    
    for name, module in model.named_modules():
        if hasattr(module, 'scale') and isinstance(module.scale, nn.Parameter):
            scale = module.scale.data
            scale_stats["total"] += scale.numel()
            
            nan_count = torch.isnan(scale).sum().item()
            inf_count = torch.isinf(scale).sum().item()
            zero_count = (scale == 0).sum().item()
            tiny_count = ((scale.abs() > 0) & (scale.abs() < 1e-6)).sum().item()
            neg_count = (scale < 0).sum().item()
            
            scale_stats["nan"] += nan_count
            scale_stats["inf"] += inf_count
            scale_stats["zero"] += zero_count
            scale_stats["tiny"] += tiny_count
            scale_stats["negative"] += neg_count
            
            if nan_count > 0 or inf_count > 0 or zero_count > 0:
                problem_scales.append(f"{name}: nan={nan_count}, inf={inf_count}, zero={zero_count}, "
                                     f"min={scale.min().item():.2e}, max={scale.max().item():.2e}")
    
    logger.info(f"Scale statistics: {scale_stats}")
    if problem_scales:
        logger.warning(f"Problematic scales found:")
        for ps in problem_scales[:10]:  # Show first 10
            logger.warning(f"  {ps}")
    
    # Compare expected vs actual scales to detect miscalibration
    scale_comparisons = []
    for name, module in model.named_modules():
        # Check QATLinearADC tiles (inside TiledLinearADC)
        if isinstance(module, QATLinearADC):
            # For weight quantizer
            if hasattr(module, 'weight') and hasattr(module, 'weight_quantizer'):
                w = module.weight.data
                actual_scale = module.weight_quantizer.scale.data
                qmax = module.weight_quantizer.qmax
                expected_scale = w.abs().max() / qmax  # per-tensor expected
                ratio = (actual_scale.mean() / expected_scale).item() if expected_scale > 0 else 0
                if ratio < 0.1 or ratio > 10:  # Flag if off by 10x
                    scale_comparisons.append(
                        f"{name}.weight: expected={expected_scale:.2e}, actual={actual_scale.mean():.2e}, ratio={ratio:.2f}"
                    )
    
    if scale_comparisons:
        logger.warning(f"Scale mismatches (ratio < 0.1 or > 10) - {len(scale_comparisons)} found:")
        for sc in scale_comparisons[:20]:  # Show first 20
            logger.warning(f"  {sc}")
    else:
        logger.info("All scales are within expected range (0.1x to 10x of expected)")


def main():
    parser = argparse.ArgumentParser()
    # Where to load FP model checkpoint from (dir with checkpoint-* or the checkpoint dir itself)
    parser.add_argument("--fp_checkpoint_dir", type=str, required=False, help="Path to FP fine-tuning output dir or a specific checkpoint-* dir")
    parser.add_argument("--output_dir", type=str, default="./outputs_qa_adc_qat")
    parser.add_argument("--seed", type=int, default=42)

    # ADC QAT settings
    parser.add_argument("--bx", type=int, default=8, help="Activation bits")
    parser.add_argument("--bw", type=int, default=8, help="Weight bits")
    parser.add_argument("--ba", type=int, default=8, help="ADC bits")
    parser.add_argument("--k", type=int, default=4, help="Hardware design parameter for ADC")
    parser.add_argument("--ashift", action="store_true",
                       help="Enable A-shift quantization strategy: "
                            "asymmetric (unsigned) quantization + A-shift for GeLU outputs. "
                            "If False, uses symmetric (signed) quantization for all activations.")
    parser.add_argument("--exclude_head", action="store_true", help="Exclude qa_outputs from quantization")
    parser.add_argument("--exclude_pooler", action="store_true", help="Exclude pooler from quantization")
    parser.add_argument("--exclude_embeddings", action="store_true", help="Exclude embeddings from quantization")
    parser.add_argument("--mvm_limit", type=int, default=256, help="Memory vector multiplication limit for tiling")
    parser.add_argument("--fixed_delta", action="store_true", help="Use fixed analytical ADC delta (disable dynamic delta and annealing)")
    parser.add_argument("--adc_resume_dir", type=str, required=False, help="Path to ADC checkpoint dir to resume from")
    parser.add_argument("--disable_adc_monitoring", action="store_true", help="Disable ADC distribution monitoring and pipeline logs")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--wandb_project", type=str, default="bert-adc-qat", help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name (auto-generated if not provided)")
    parser.add_argument("--wandb_tags", type=str, nargs="+", default=None, help="Optional WandB tags")
    parser.add_argument("--wandb_notes", type=str, default=None, help="Optional WandB notes")

    # Data/Trainer settings (same pipeline as FP)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-8)  # Reduced from 1e-6
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--doc_stride", type=int, default=128)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--eval_only", action="store_true", help="Skip training and run evaluation only with analytical ADC delta")
    parser.add_argument("--kurtosis_lambda", type=float, default=0.0, help="Lambda for W-reshape kurtosis regularization (0 disables)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate (paper uses 0.2 for BERT-base)")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="LR scheduler type: linear, cosine, etc.")
    
    # BitAug parameters (Paper: Bit Augmentation, Equation 8-10)
    parser.add_argument("--bitaug", action="store_true", 
                       help="Enable BitAug: augment training with multiple ADC bit precisions")
    parser.add_argument("--bitaug_lambda", type=float, default=0.5, 
                       help="BitAug loss coefficient λ_b (default: 0.5)")
    parser.add_argument("--bitaug_neighbor_range", type=int, default=1, 
                       help="BitAug neighbor range: ±N bits around target ba (default: 1 for ±1 bit)")
    parser.add_argument("--bitaug_min_bits", type=int, default=4, 
                       help="Minimum bit precision for BitAug sampling")
    parser.add_argument("--bitaug_max_bits", type=int, default=12, 
                       help="Maximum bit precision for BitAug sampling")
    
    # ADC-LoRA parameters (Paper Section 3.4: Training Overhead Reduction)
    parser.add_argument("--use_lora", action="store_true",
                       help="Enable ADC-LoRA: reduce trainable parameters by using low-rank adaptation. "
                            "Implements Eq. 11: Y = QA(Qx(X)Qw(W + AB))")
    parser.add_argument("--lora_r", type=int, default=8,
                       help="LoRA rank r (dimension of low-rank matrices, default: 8)")
    parser.add_argument("--lora_alpha", type=float, default=16.0,
                       help="LoRA scaling factor alpha (scaling = alpha/r, default: 16.0)")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                       help="LoRA dropout rate (default: 0.0)")
    parser.add_argument("--lora_target_modules", type=str, nargs="+", 
                       default=["query", "value"],
                       help="Module name patterns to apply LoRA to (default: ['query', 'value'])")
    parser.add_argument("--lora_warmup_steps", type=int, default=0,
                       help="Number of MSE warmup steps for LoRA initialization (default: 0, disabled)")
    parser.add_argument("--lora_warmup_lr", type=float, default=1e-3,
                       help="Learning rate for LoRA MSE warmup (default: 1e-3)")
    
    args = parser.parse_args()

    set_seed(args.seed)
    
    # Initialize WandB (same as PTQ script)
    use_wandb = WANDB_AVAILABLE and not args.disable_wandb
    wandb_run = None

    if use_wandb:
        lora_suffix = f"_lora_r{args.lora_r}" if args.use_lora else ""
        default_run_name = f"qat_bx{args.bx}_bw{args.bw}_ba{args.ba}_k{args.k}{lora_suffix}"
        run_name = args.wandb_run_name or default_run_name
        wandb_config = {
            "bx": args.bx,
            "bw": args.bw,
            "ba": args.ba,
            "k": args.k,
            "ashift": args.ashift,
            "fixed_delta": args.fixed_delta,
            "num_train_epochs": args.num_train_epochs,
            "learning_rate": args.learning_rate,
            "train_batch_size": args.per_device_train_batch_size,
            "eval_batch_size": args.per_device_eval_batch_size,
            "warmup_ratio": args.warmup_ratio,
            "warmup_steps": args.warmup_steps,
            "seed": args.seed,
            "mvm_limit": args.mvm_limit,
            "delta_loss_weight": 0.0 if args.fixed_delta else 0.01,
            # LoRA parameters
            "use_lora": args.use_lora,
            "lora_r": args.lora_r if args.use_lora else None,
            "lora_alpha": args.lora_alpha if args.use_lora else None,
            "lora_dropout": args.lora_dropout if args.use_lora else None,
            "lora_target_modules": args.lora_target_modules if args.use_lora else None,
            "lora_warmup_steps": args.lora_warmup_steps if args.use_lora else None,
        }
        wandb_run = wandb.init(
                project=args.wandb_project,
                name=run_name,
                config=wandb_config,
                tags=args.wandb_tags,
                notes=args.wandb_notes or "",
        )
        logger.info(f"WandB initialized: project={args.wandb_project}, run={run_name}")
    else:
        logger.info("WandB logging disabled")

    
    # Note: signed_activations is now set PER-LAYER in BertADCConverter
    # based on whether the layer comes after GeLU
    logger.info(f"Quantization strategy: ashift={args.ashift}")
    if args.ashift:
        logger.info("  → Asymmetric (unsigned) + A-shift for layers AFTER GeLU (e.g., layer.X.output.dense)")
        logger.info("  → Symmetric (signed) for all OTHER activations")
    else:
        logger.info("  → Symmetric (signed) quantization for ALL activations")

    last_ckpt = None
    resume_ckpt = None
    if args.adc_resume_dir:
        resume_ckpt = find_last_checkpoint_dir(args.adc_resume_dir)
        logger.info(f"Resuming ADC training from: {resume_ckpt}")
        # Load the ADC model by first creating a base BERT model, then converting to ADC layers
        # and loading the state dict
        config = AutoConfig.from_pretrained(resume_ckpt)
        # Apply dropout from args (paper uses 0.2 for BERT-base)
        if args.dropout != config.hidden_dropout_prob:
            logger.info(f"Updating dropout: {config.hidden_dropout_prob} -> {args.dropout}")
            config.hidden_dropout_prob = args.dropout
            config.attention_probs_dropout_prob = args.dropout
        base_model = BertForQuestionAnswering(config)
        tokenizer = AutoTokenizer.from_pretrained(resume_ckpt, use_fast=True)
        tokenizer.padding_side = "right"

        # Convert to ADC layers using SAME parameters as checkpoint
        model = BertADCConverter.replace_linear_with_adc_qat(
            base_model,
            bx=args.bx,  # Use args, not hardcoded!
            bw=args.bw,
            ba=args.ba,
            k=args.k,
            ashift=args.ashift,  # CRITICAL: Must match checkpoint!
            exclude_patterns=["embeddings", "pooler", "qa_outputs"],
            mvm_limit=args.mvm_limit,
            # W-reshape (kurtosis) parameters
            use_kurtosis_loss=(args.kurtosis_lambda > 0),
            kurtosis_weight=args.kurtosis_lambda if args.kurtosis_lambda > 0 else 0.0006,
            target_kurtosis=1.8,
            # LoRA parameters (Paper Section 3.4)
            use_lora=args.use_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules=args.lora_target_modules,
        )

        # Load the state dict with strict=False to handle quantizer parameters
        bin_path = os.path.join(resume_ckpt, 'pytorch_model.bin')
        safe_path = os.path.join(resume_ckpt, 'model.safetensors')

        if os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location='cpu')
        elif os.path.exists(safe_path):
            try:
                from safetensors.torch import load_file as safe_load_file
                state_dict = safe_load_file(safe_path)
            except ImportError:
                raise ImportError("safetensors not available, but checkpoint uses safetensors format")
        else:
            raise FileNotFoundError(f"No model file found in {resume_ckpt}")

        # Load state dict with flexible shape handling for quantizer parameters
        logger.info("Loading state dict with flexible parameter matching...")
        missing_keys, unexpected_keys = load_state_dict_flexible(model, state_dict)
        
        if unexpected_keys:
            logger.info(f"Ignored {len(unexpected_keys)} unexpected keys from checkpoint")
        if missing_keys:
            logger.warning(f"Missing {len(missing_keys)} keys when loading checkpoint")
        
        logger.info(f"Successfully loaded ADC checkpoint: {len(state_dict)} keys loaded")
        
        # Mark all quantizers as initialized so they don't try to reinitialize
        for name, module in model.named_modules():
            # Handle LoRA-wrapped layers
            if isinstance(module, LoRATiledLinearADC):
                tiles = module.tiled_layer.tiles
            elif isinstance(module, LoRAQATLinearADC):
                tiles = [module.base_layer]
            elif isinstance(module, TiledLinearADC):
                tiles = module.tiles
            elif isinstance(module, QATLinearADC):
                tiles = [module]
            else:
                continue
            
            for tile in tiles:
                if hasattr(tile, 'activation_quantizer'):
                    tile.activation_quantizer._scale_initialized = True
                    if hasattr(tile.activation_quantizer, '_zp_initialized'):
                        tile.activation_quantizer._zp_initialized = True
                if hasattr(tile, 'weight_quantizer'):
                    tile.weight_quantizer._scale_initialized = True
                    if hasattr(tile.weight_quantizer, '_zp_initialized'):
                        tile.weight_quantizer._zp_initialized = True
        
        logger.info("Marked all quantizers as initialized")
        
        # Debug: Log scale statistics to identify problematic values
        debug_scale_values(model)
        
        # Set quantizers to 'qat' mode for gradient learning during training
        logger.info("Setting quantizers to 'qat' mode for gradient learning...")
        qat_layer_count = 0
        for name, module in model.named_modules():
            if isinstance(module, (QATLinearADC, TiledLinearADC)):
                if hasattr(module, 'set_quantizer_mode'):
                    module.set_quantizer_mode('qat')
                    qat_layer_count += 1
        logger.info(f"Set {qat_layer_count} layers to 'qat' mode (scales will be learned via gradients)")

        do_convert = False
        do_warm_start = False
    else:
        if not args.fp_checkpoint_dir:
            parser.error("Either --fp_checkpoint_dir or --adc_resume_dir is required")
        last_ckpt = find_last_checkpoint_dir(args.fp_checkpoint_dir)
        logger.info(f"Loading fine-tuned FP checkpoint from: {last_ckpt}")
        tokenizer = AutoTokenizer.from_pretrained(last_ckpt, use_fast=True)
        tokenizer.padding_side = "right"
        model = load_qa_model_robust(last_ckpt)
        
        # Apply dropout from args (paper uses 0.2 for BERT-base)
        if hasattr(model.config, 'hidden_dropout_prob') and args.dropout != model.config.hidden_dropout_prob:
            logger.info(f"Updating dropout: {model.config.hidden_dropout_prob} -> {args.dropout}")
            model.config.hidden_dropout_prob = args.dropout
            model.config.attention_probs_dropout_prob = args.dropout
            # Re-apply to model modules
            for module in model.modules():
                if hasattr(module, 'dropout') and isinstance(module.dropout, torch.nn.Dropout):
                    module.dropout.p = args.dropout
        
        do_convert = True
        do_warm_start = True

    if do_convert:
        exclude_patterns = []
        if args.exclude_embeddings:
            exclude_patterns.append("embeddings")
        if args.exclude_pooler:
            exclude_patterns.append("pooler")
        if args.exclude_head:
            exclude_patterns.append("qa_outputs")
        if not exclude_patterns:
            exclude_patterns = ["embeddings", "pooler", "qa_outputs"]

        model = BertADCConverter.replace_linear_with_adc_qat(
            model,
            bx=args.bx,
            bw=args.bw,
            ba=args.ba,
            k=args.k,
            ashift=args.ashift,
            exclude_patterns=exclude_patterns,
            mvm_limit=args.mvm_limit,
            # W-reshape (kurtosis) parameters
            use_kurtosis_loss=(args.kurtosis_lambda > 0),
            kurtosis_weight=args.kurtosis_lambda if args.kurtosis_lambda > 0 else 0.0006,
            target_kurtosis=1.8,
            # LoRA parameters (Paper Section 3.4)
            use_lora=args.use_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules=args.lora_target_modules,
        )

    if do_warm_start:
        try:
            warm_start_adc_quantizers_from_qat(model, last_ckpt)
        except Exception as e:
            logger.warning(f"Quantizer warm-start failed: {e}")
    
    # Disable gradient checkpointing - it may interfere with quantizer updates
    # if hasattr(model, 'gradient_checkpointing_enable'):
    #     model.gradient_checkpointing_enable()
    #     logger.info("Gradient checkpointing enabled to reduce memory usage")
    
    # Add gradient monitoring
    add_gradient_hooks(model)

    # Setup ADC distribution monitoring
    adc_step_monitor = None
    if ADC_MONITORING_AVAILABLE and (not args.disable_adc_monitoring):
        try:
            logger.info("Setting up ADC distribution monitoring...")
            adc_plotter, adc_step_monitor = create_adc_training_monitor(
                output_dir=os.path.join(args.output_dir, "adc_distributions"),
                plot_every_n_batches=50,  # Generate plots every 50 batches
                max_layers=3  # Monitor max 3 layers to avoid too many plots
            )
            
            # Add monitoring to key layers - enable full pipeline monitoring
            monitored_layers = add_adc_monitoring_to_model(
                model, adc_plotter, 
                layer_patterns=["attention.output.dense", "intermediate.dense", "output.dense"],
                max_layers=3,
                monitor_full_pipeline=True  # Enable full pipeline monitoring
            )
            
            if monitored_layers > 0:
                logger.info(f"ADC distribution monitoring enabled for {monitored_layers} layers")
            else:
                logger.warning("No ADC layers found for monitoring")
                adc_step_monitor = None
        except Exception as e:
            logger.warning(f"Failed to setup ADC monitoring: {e}")
            adc_step_monitor = None

    stats = BertADCConverter.count_adc_qat_layers(model)
    logger.info(f"ADC QAT conversion: {stats['adc_qat_linear']} TiledLinearADC, "
                f"{stats.get('lora_linear', 0)} LoRATiledLinearADC, "
                f"{stats['regular_linear']} remaining Linear, "
                f"{stats['total_params']:,} params")
    
    # Log LoRA-specific information
    if args.use_lora:
        lora_stats = get_lora_param_count(model)
        logger.info(f"LoRA enabled: {lora_stats['lora_params']:,} trainable LoRA params, "
                   f"{lora_stats['frozen_params']:,} frozen params, "
                   f"compression ratio: {lora_stats['compression_ratio']:.1f}x")
        if use_wandb and wandb_run is not None:
            wandb.run.summary["lora_params"] = lora_stats["lora_params"]
            wandb.run.summary["frozen_params"] = lora_stats["frozen_params"]
            wandb.run.summary["lora_compression_ratio"] = lora_stats["compression_ratio"]
    
    if use_wandb and wandb_run is not None:
        wandb.run.summary["adc_qat_linear_layers"] = stats["adc_qat_linear"]
        wandb.run.summary["lora_linear_layers"] = stats.get("lora_linear", 0)
        wandb.run.summary["regular_linear_layers"] = stats["regular_linear"]
        wandb.run.summary["total_params"] = stats["total_params"]

    # Data
    raw = load_dataset("squad")
    train_dataset = raw["train"].map(
        lambda x: prepare_train_features(x, tokenizer, args.max_length, args.doc_stride),
        batched=True,
        remove_columns=raw["train"].column_names,
        desc="Tokenizing train",
    )
    eval_examples = raw["validation"]
    # Use prepare_train_features for eval too, so labels (start_positions, end_positions) are included
    # This allows Trainer to compute F1/EM during training evaluation
    eval_dataset = eval_examples.map(
        lambda x: prepare_train_features(x, tokenizer, args.max_length, args.doc_stride),
        batched=True,
        remove_columns=eval_examples.column_names,
        desc="Tokenizing validation",
    )

    squad_metric = evaluate.load("squad")
    metrics_computer = MetricsComputer(eval_examples, eval_dataset, tokenizer, squad_metric)

    # TrainingArguments (keep parity with FP pipeline)
    # Normalize warmup settings so ratio is never None
    _warmup_steps = args.warmup_steps if args.warmup_steps not in (None, 0) else 0
    _warmup_ratio = args.warmup_ratio if _warmup_steps == 0 else 0.0
    
    # Configure reporting: use WandB if available and enabled, otherwise none (same as PTQ)
    _report_to = "wandb" if use_wandb else "none"

    try:
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            num_train_epochs=args.num_train_epochs,
            warmup_ratio=_warmup_ratio,
            warmup_steps=_warmup_steps,
            lr_scheduler_type=args.lr_scheduler_type,  # Paper uses linear decay
            logging_steps=100,  # Less frequent logging for speed
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            fp16=args.fp16,
            report_to=_report_to,
            # Gradient settings
            max_grad_norm=1.0,
            gradient_accumulation_steps=1,  # No accumulation for max speed (set to 2-4 if OOM)
        )
        if resume_ckpt:
            training_args.resume_from_checkpoint = resume_ckpt
    except TypeError:
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            num_train_epochs=args.num_train_epochs,
            warmup_steps=_warmup_steps,
            lr_scheduler_type=args.lr_scheduler_type,  # Paper uses linear decay
            logging_steps=100,  # Less frequent logging for speed
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            fp16=args.fp16,
            report_to=_report_to,
            # Gradient settings
            max_grad_norm=1.0,
            gradient_accumulation_steps=1,  # No accumulation for max speed
        )
        if resume_ckpt:
            training_args.resume_from_checkpoint = resume_ckpt

    # Create custom trainer with ADC loss handling
    callbacks = [EvalMetricsLogger(use_wandb=use_wandb)]

    # Preprocess logits for QA metrics (needed to extract start/end logits)
    def preprocess_logits_for_metrics(logits, labels):
        """Extract start and end logits for QA metrics computation"""
        return logits[0], logits[1]  # (start_logits, end_logits)

    # Prepare BitAug neighbors if enabled
    bitaug_neighbors = None
    if args.bitaug:
        bitaug_neighbors = get_bitaug_neighbors(
            target_ba=args.ba,
            neighbor_range=args.bitaug_neighbor_range,
            min_bits=args.bitaug_min_bits,
            max_bits=args.bitaug_max_bits
        )
        logger.info(f"BitAug enabled: target_ba={args.ba}, lambda={args.bitaug_lambda}, "
                   f"neighbors={bitaug_neighbors}")
        if use_wandb and wandb_run is not None:
            wandb.run.config.update({
                "bitaug": True,
                "bitaug_lambda": args.bitaug_lambda,
                "bitaug_neighbors": bitaug_neighbors,
            })

    trainer = ADCLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=metrics_computer.compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=callbacks,
        adc_step_monitor=adc_step_monitor,  # Add ADC monitoring
        # BitAug parameters
        use_bitaug=args.bitaug,
        bitaug_lambda=args.bitaug_lambda,
        target_ba=args.ba,
        bitaug_neighbors=bitaug_neighbors,
    )

    # LoRA MSE warmup (Paper Equation 12)
    if args.use_lora and args.lora_warmup_steps > 0 and not args.eval_only:
        logger.info(f"Running LoRA MSE warmup for {args.lora_warmup_steps} steps...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        
        # Select only columns needed for model input (avoid None values in other columns)
        model_input_columns = ["input_ids", "attention_mask", "token_type_ids", "start_positions", "end_positions"]
        warmup_columns = [col for col in model_input_columns if col in train_dataset.column_names]
        warmup_dataset = train_dataset.select_columns(warmup_columns)
        
        # Create a small dataloader for warmup
        warmup_dataloader = torch.utils.data.DataLoader(
            warmup_dataset,
            batch_size=args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=default_data_collator,
        )
        
        warmup_loss = warmup_lora_mse(
            model=model,
            dataloader=warmup_dataloader,
            num_steps=args.lora_warmup_steps,
            lr=args.lora_warmup_lr,
            device=device,
        )
        
        if use_wandb and wandb_run is not None:
            wandb.log({"lora/warmup_loss": warmup_loss})
        
        logger.info(f"LoRA warmup complete with final loss: {warmup_loss:.6f}")

    train_metrics = {}
    if args.eval_only:
        logger.info("Skipping training; running evaluation only...")
    else:
        logger.info("Starting ADC QAT fine-tuning with HF Trainer...")
        train_output = trainer.train()
        if train_output is not None:
            train_metrics = getattr(train_output, "metrics", {}) or {}
            training_loss = getattr(train_output, "training_loss", None)
            if training_loss is not None and "training_loss" not in train_metrics:
                train_metrics["training_loss"] = training_loss
        if use_wandb and wandb_run is not None:
            wandb_log = {}
            for key, value in train_metrics.items():
                if isinstance(value, (int, float)):
                    wandb_log[f"train/{key}"] = value
            if wandb_log:
                wandb.log(wandb_log, step=trainer.state.global_step)
    logger.info("ADC QAT training completed.")

    # Final eval (same as FP script)
    logger.info("Running final evaluation with F1 computation...")
    preds = trainer.predict(eval_dataset).predictions
    formatted = postprocess_qa_predictions(
        examples=eval_examples,
        features=eval_dataset,
        predictions=preds,
    )
    refs = [{"id": ex_id, "answers": ans} for ex_id, ans in zip(eval_examples["id"], eval_examples["answers"])]
    preds_for_metric = [{"id": k, "prediction_text": v} for k, v in formatted.items()]
    eval_metrics = squad_metric.compute(predictions=preds_for_metric, references=refs)

    logger.info(f"Final F1: {eval_metrics['f1']:.2f}, EM: {eval_metrics['exact_match']:.2f}")
    if use_wandb and wandb_run is not None:
        wandb.log(
            {
                "eval/f1": float(eval_metrics["f1"]),
                "eval/exact_match": float(eval_metrics["exact_match"]),
            },
            step=trainer.state.global_step,
        )
        wandb.run.summary["final_eval_f1"] = float(eval_metrics["f1"])
        wandb.run.summary["final_eval_exact_match"] = float(eval_metrics["exact_match"])
        if train_metrics:
            for key, value in train_metrics.items():
                if isinstance(value, (int, float)):
                    wandb.run.summary[f"train/{key}"] = float(value)

    # Generate final ADC distribution plots
    if ADC_MONITORING_AVAILABLE and adc_step_monitor:
        try:
            logger.info("Generating final ADC distribution evolution plots...")
            for layer_name in adc_plotter.batch_data.keys():
                adc_plotter.plot_evolution_over_batches(layer_name, max_batches=50)
            logger.info(f"ADC distribution plots saved to: {os.path.join(args.output_dir, 'adc_distributions')}")
        except Exception as e:
            logger.warning(f"Failed to generate final ADC plots: {e}")

    # Save artifacts
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    if use_wandb and wandb_run is not None:
        wandb.run.summary["output_dir"] = args.output_dir
        if args.fp_checkpoint_dir:
            wandb.run.summary["fp_checkpoint_dir"] = args.fp_checkpoint_dir

    with open(os.path.join(args.output_dir, "eval_metrics.txt"), "w") as f:
        for k, v in sorted(eval_metrics.items()):
            f.write(f"{k}: {v}\n")

    # Save ADC configuration
    with open(os.path.join(args.output_dir, "adc_config.txt"), "w") as f:
        f.write(f"bx (activation bits): {args.bx}\n")
        f.write(f"bw (weight bits): {args.bw}\n")
        f.write(f"ba (ADC bits): {args.ba}\n")
        f.write(f"k (hardware parameter): {args.k}\n")
        f.write(f"ashift: {args.ashift}\n")
        activation_strategy = (
            "per-layer (A-shift for GeLU outputs, symmetric otherwise)"
            if args.ashift
            else "per-layer symmetric (signed activations)"
        )
        f.write(f"activation_strategy: {activation_strategy}\n")
        if 'exclude_patterns' in locals():
            f.write(f"exclude_patterns: {exclude_patterns}\n")
        f.write(f"mvm_limit: {args.mvm_limit}\n")

    print("Final metrics:", eval_metrics)
    print(f"Artifacts saved to: {args.output_dir}")
    if ADC_MONITORING_AVAILABLE and adc_step_monitor:
        print(f"ADC distribution plots: {os.path.join(args.output_dir, 'adc_distributions')}")
    if use_wandb and wandb_run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
