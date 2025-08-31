import argparse
import os
import time
import collections
import logging
from typing import Dict, Any, Optional, List, Tuple
import json

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
)

from adc_layers import QATLinearADC, TiledLinearADC, LearnableQuantizer  # noqa: F401

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        signed_activations: bool = False,
        exclude_patterns: Optional[List[str]] = None,
        mvm_limit: int = 256,  # Default to 256 to match CLI default
    ) -> nn.Module:
        """
        Replace all nn.Linear layers in the model with TiledLinearADC, except excluded.

        Args:
            model: Model to convert.
            bx: Bits for activation quantization.
            bw: Bits for weight quantization.
            ba: Bits for ADC quantization.
            k: Hardware design parameter for ADC.
            ashift: Enable ashift functionality.
            signed_activations: Use signed activation quantization.
            exclude_patterns: List of substrings of module names to exclude.
            mvm_limit: Memory vector multiplication limit for tiling.
        """
        if exclude_patterns is None:
            # Default: don't quantize embeddings/pooler and QA output head unless requested
            exclude_patterns = ["embeddings", "pooler", "qa_outputs"]

        def should_exclude(name: str) -> bool:
            return any(pat in name for pat in exclude_patterns)

        def replace_recursive(module: nn.Module, name: str = ""):
            for child_name, child_module in module.named_children():
                full_name = f"{name}.{child_name}" if name else child_name

                if isinstance(child_module, nn.Linear) and not should_exclude(full_name):
                    adc_qat_layer = TiledLinearADC(
                        in_features=child_module.in_features,
                        out_features=child_module.out_features,
                        bias=(child_module.bias is not None),
                        bx=bx,
                        bw=bw,
                        ba=ba,
                        k=k,
                        ashift=ashift,
                        signed_activations=signed_activations,
                        mvm_limit=mvm_limit,
                    )
                    # Use the load_weights method instead of manual copying
                    adc_qat_layer.load_weights(child_module)
                    setattr(module, child_name, adc_qat_layer)
                    logger.info(f"Replaced {full_name} with TiledLinearADC (bx={bx}, bw={bw}, ba={ba}, k={k}, mvm_limit={mvm_limit})")
                else:
                    replace_recursive(child_module, full_name)

        replace_recursive(model)
        return model

    @staticmethod
    def count_adc_qat_layers(model: nn.Module) -> Dict[str, int]:
        counts = {"adc_qat_linear": 0, "regular_linear": 0, "total_params": 0}
        for _, module in model.named_modules():
            if isinstance(module, (QATLinearADC, TiledLinearADC)):
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
    offsets_mapping = tokenized.pop("offset_mapping")

    tokenized["start_positions"] = []
    tokenized["end_positions"] = []

    for i, offsets in enumerate(offsets_mapping):
        input_ids = tokenized["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = tokenized.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

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
            predictions, _ = eval_pred
            formatted_predictions = postprocess_qa_predictions(
                examples=self.eval_examples,
                features=self.eval_dataset,
                predictions=predictions,
            )
            references = [{"id": ex_id, "answers": ans} for ex_id, ans in zip(self.eval_examples["id"], self.eval_examples["answers"])]
            predictions_for_metric = [{"id": k, "prediction_text": v} for k, v in formatted_predictions.items()]
            result = self.squad_metric.compute(predictions=predictions_for_metric, references=references)
            return {"f1": result["f1"], "exact_match": result["exact_match"]}
        except Exception:
            import traceback
            traceback.print_exc()
            return {"f1": 0.0, "exact_match": 0.0}


def add_gradient_hooks(model):
    """Add hooks to monitor gradients and detect infinite gradients"""
    
    def grad_hook(name):
        def hook(grad):
            if grad is not None:
                grad_norm = grad.norm().item()
                has_nan = torch.isnan(grad).any()
                has_inf = torch.isinf(grad).any()
                
                # Log at a slightly higher threshold to reduce noise; still clip
                if has_nan or has_inf or grad_norm > 500:
                    print(f"GRADIENT ISSUE in {name}: norm={grad_norm:.6f}, nan={has_nan}, inf={has_inf}")
                    print(f"  Grad shape: {grad.shape}, min: {grad.min().item():.6f}, max: {grad.max().item():.6f}")
                    
                    # More aggressive gradient clipping
                    if has_nan or has_inf:
                        grad = torch.nan_to_num(grad, nan=0.0, posinf=10.0, neginf=-10.0)
                    elif grad_norm > 500:
                        grad = grad / (grad_norm / 50.0)  # Scale down to norm≈50
                        
            return grad
        return hook
    
    # Add hooks to all parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.register_hook(grad_hook(name))
    
    print("Added gradient monitoring hooks to all parameters")


def main():
    parser = argparse.ArgumentParser()
    # Where to load FP model checkpoint from (dir with checkpoint-* or the checkpoint dir itself)
    parser.add_argument("--fp_checkpoint_dir", type=str, required=True, help="Path to FP fine-tuning output dir or a specific checkpoint-* dir")
    parser.add_argument("--output_dir", type=str, default="./outputs_qa_adc_qat")
    parser.add_argument("--seed", type=int, default=42)

    # ADC QAT settings
    parser.add_argument("--bx", type=int, default=8, help="Activation bits")
    parser.add_argument("--bw", type=int, default=8, help="Weight bits")
    parser.add_argument("--ba", type=int, default=8, help="ADC bits")
    parser.add_argument("--k", type=int, default=4, help="Hardware design parameter for ADC")
    parser.add_argument("--ashift", action="store_true", help="Enable ashift functionality")
    parser.add_argument("--signed_activations", action="store_true", help="Use signed activation quantization")
    parser.add_argument("--exclude_head", action="store_true", help="Exclude qa_outputs from quantization")
    parser.add_argument("--exclude_pooler", action="store_true", help="Exclude pooler from quantization")
    parser.add_argument("--exclude_embeddings", action="store_true", help="Exclude embeddings from quantization")
    parser.add_argument("--mvm_limit", type=int, default=256, help="Memory vector multiplication limit for tiling")

    # Data/Trainer settings (same pipeline as FP)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-8)  # Reduced from 1e-6
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--doc_stride", type=int, default=128)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    # Resolve the last checkpoint directory
    last_ckpt = find_last_checkpoint_dir(args.fp_checkpoint_dir)
    logger.info(f"Loading fine-tuned FP checkpoint from: {last_ckpt}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_dir, f"squad_adc_qat_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    # Load tokenizer from the FP checkpoint to keep exact vocab/tokenization
    tokenizer = AutoTokenizer.from_pretrained(last_ckpt, use_fast=True)
    tokenizer.padding_side = "right"

    # Load fine-tuned model robustly (works for FP and QAT checkpoints)
    model = load_qa_model_robust(last_ckpt)

    exclude_patterns = []
    if args.exclude_embeddings:
        exclude_patterns.append("embeddings")
    if args.exclude_pooler:
        exclude_patterns.append("pooler")
    if args.exclude_head:
        exclude_patterns.append("qa_outputs")
    if not exclude_patterns:
        # Default exclusions to mirror typical practice
        exclude_patterns = ["embeddings", "pooler", "qa_outputs"]

    model = BertADCConverter.replace_linear_with_adc_qat(
        model,
        bx=args.bx,
        bw=args.bw,
        ba=args.ba,
        k=args.k,
        ashift=args.ashift,
        signed_activations=args.signed_activations,
        exclude_patterns=exclude_patterns,
        mvm_limit=args.mvm_limit,
    )
    # Warm-start ADC quantizer parameters from QAT checkpoint if available
    try:
        warm_start_adc_quantizers_from_qat(model, last_ckpt)
    except Exception as e:
        logger.warning(f"Quantizer warm-start failed: {e}")
    
    # Add gradient monitoring
    add_gradient_hooks(model)

    stats = BertADCConverter.count_adc_qat_layers(model)
    logger.info(f"ADC QAT conversion: {stats['adc_qat_linear']} TiledLinearADC, {stats['regular_linear']} remaining Linear, "
                f"{stats['total_params']:,} params")

    # Data
    raw = load_dataset("squad")
    train_dataset = raw["train"].map(
        lambda x: prepare_train_features(x, tokenizer, args.max_length, args.doc_stride),
        batched=True,
        remove_columns=raw["train"].column_names,
        desc="Tokenizing train",
    )
    eval_examples = raw["validation"]
    eval_dataset = eval_examples.map(
        lambda x: prepare_validation_features(x, tokenizer, args.max_length, args.doc_stride),
        batched=True,
        remove_columns=eval_examples.column_names,
        desc="Tokenizing validation",
    )

    squad_metric = evaluate.load("squad")
    metrics_computer = MetricsComputer(eval_examples, eval_dataset, tokenizer, squad_metric)

    # TrainingArguments (keep parity with FP pipeline)
    try:
        training_args = TrainingArguments(
            output_dir=out_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            num_train_epochs=args.num_train_epochs,
            warmup_ratio=args.warmup_ratio,
            logging_steps=50,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=2,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            fp16=args.fp16,
            report_to="none",
            # Add gradient clipping
            max_grad_norm=1.0,  # Tighter clipping to stabilize early training
            gradient_accumulation_steps=2,  # Accumulate gradients to reduce variance
        )
    except TypeError:
        training_args = TrainingArguments(
            output_dir=out_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            num_train_epochs=args.num_train_epochs,
            warmup_steps=0,
            logging_steps=50,
            save_steps=args.save_steps,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            fp16=args.fp16,
            # Add gradient clipping
            max_grad_norm=1.0,
            gradient_accumulation_steps=2,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=metrics_computer.compute_metrics,
    )

    logger.info("Starting ADC QAT fine-tuning with HF Trainer...")
    trainer.train()
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

    # Save artifacts
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    with open(os.path.join(out_dir, "eval_metrics.txt"), "w") as f:
        for k, v in sorted(eval_metrics.items()):
            f.write(f"{k}: {v}\n")

    # Save ADC configuration
    with open(os.path.join(out_dir, "adc_config.txt"), "w") as f:
        f.write(f"bx (activation bits): {args.bx}\n")
        f.write(f"bw (weight bits): {args.bw}\n")
        f.write(f"ba (ADC bits): {args.ba}\n")
        f.write(f"k (hardware parameter): {args.k}\n")
        f.write(f"ashift: {args.ashift}\n")
        f.write(f"signed_activations: {args.signed_activations}\n")
        f.write(f"exclude_patterns: {exclude_patterns}\n")
        f.write(f"mvm_limit: {args.mvm_limit}\n")

    print("Final metrics:", eval_metrics)
    print(f"Artifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
