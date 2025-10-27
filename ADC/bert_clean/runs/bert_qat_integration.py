import argparse
import os
import time
import collections
import logging
from typing import Dict, Any, Optional, List, Tuple

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

from ADC.bert_clean.core.qat_layers import QATLinear
from ADC.bert_clean.core.wandb_integration import (
    WandbQATCallback,
    init_wandb_run,
    log_model_architecture,
    log_training_summary,
    WANDB_AVAILABLE,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KurtosisLossTrainer(Trainer):
    """Custom Trainer that adds kurtosis regularization over QATLinear weights."""

    def __init__(self, *args, kurtosis_lambda: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.kurtosis_lambda = float(kurtosis_lambda)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        outputs = model(**inputs)

        # Handle standard HF output structures
        if isinstance(outputs, tuple):
            loss = outputs[0] if hasattr(outputs[0], 'loss') else outputs[0]
        elif isinstance(outputs, dict):
            loss = outputs.get('loss', None)
        else:
            loss = outputs

        # Kurtosis penalty over QATLinear weights (W-reshape regularization analogue)
        kurtosis_reg = 0.0
        if self.kurtosis_lambda > 0.0 and model.training:
            eps = 1e-6
            for module in model.modules():
                try:
                    if isinstance(module, QATLinear):
                        w = module.weight
                        if w is None:
                            continue
                        w_flat = w.view(-1)
                        mu = torch.mean(w_flat)
                        std = torch.std(w_flat) + eps
                        z = (w_flat - mu) / std
                        kappa = torch.mean(z ** 4)
                        kurtosis_reg = kurtosis_reg + kappa
                except Exception:
                    pass

        if isinstance(loss, torch.Tensor):
            loss = loss + (self.kurtosis_lambda * kurtosis_reg)
        else:
            loss = (self.kurtosis_lambda * kurtosis_reg)

        return (loss, outputs) if return_outputs else loss


class BertQATConverter:
    """Convert BERT model to use QAT layers for QA."""

    @staticmethod
    def replace_linear_with_qat(
        model: nn.Module,
        weight_bits: int = 8,
        activation_bits: int = 8,
        exclude_patterns: Optional[List[str]] = None,
    ) -> nn.Module:
        """
        Replace all nn.Linear layers in the model with QATLinear, except excluded.

        Args:
            model: Model to convert.
            weight_bits: Bits for weight quantization.
            activation_bits: Bits for activation quantization.
            exclude_patterns: List of substrings of module names to exclude.
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
                    qat_layer = QATLinear(
                        child_module.in_features,
                        child_module.out_features,
                        bias=(child_module.bias is not None),
                        weight_bits=weight_bits,
                        activation_bits=activation_bits,
                    )
                    with torch.no_grad():
                        qat_layer.weight.copy_(child_module.weight)
                        if child_module.bias is not None:
                            qat_layer.bias.copy_(child_module.bias)
                    setattr(module, child_name, qat_layer)
                    logger.info(f"Replaced {full_name} with QATLinear")
                else:
                    replace_recursive(child_module, full_name)

        replace_recursive(model)
        return model

    @staticmethod
    def count_qat_layers(model: nn.Module) -> Dict[str, int]:
        counts = {"qat_linear": 0, "regular_linear": 0, "total_params": 0}
        for _, module in model.named_modules():
            if isinstance(module, QATLinear):
                counts["qat_linear"] += 1
            elif isinstance(module, nn.Linear):
                counts["regular_linear"] += 1
            if hasattr(module, "parameters"):
                counts["total_params"] += sum(p.numel() for p in module.parameters())
        return counts
def _load_state_dict_from_dir(model_dir: str) -> Dict[str, torch.Tensor]:
    """Load a HF checkpoint state dict from a directory supporting safetensors and PT."""
    safetensors_path = os.path.join(model_dir, "model.safetensors")
    pytorch_bin_path = os.path.join(model_dir, "pytorch_model.bin")

    # Try safetensors first
    try:
        from safetensors.torch import load_file as safe_load_file  # type: ignore
        if os.path.isfile(safetensors_path):
            return safe_load_file(safetensors_path)
    except Exception:
        pass

    # Fallback to PyTorch bin
    if os.path.isfile(pytorch_bin_path):
        return torch.load(pytorch_bin_path, map_location="cpu")

    raise FileNotFoundError(
        f"No model state file found in {model_dir} (expected 'model.safetensors' or 'pytorch_model.bin')"
    )


def _is_qat_state_dict(state_dict: Dict[str, torch.Tensor]) -> bool:
    """Heuristically detect presence of QAT parameters in a state dict."""
    for key in state_dict.keys():
        if ".weight_quantizer." in key or ".activation_quantizer." in key:
            return True
    return False


def _coerce_state_dict_to_model_shapes(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Return a copy of state_dict with tensors reshaped/dropped to fit model shapes.

    Rules:
    - If shapes match: keep as-is.
    - If both are 1D and one has length 1 and the other length N:
      - If model wants [1] and checkpoint has [N]: use mean to reduce to [1].
      - If model wants [N] and checkpoint has [1]: expand to [N].
    - Otherwise: drop the key so load_state_dict ignores it.
    """
    model_sd = model.state_dict()
    fixed: Dict[str, torch.Tensor] = {}
    adapted_count = 0
    dropped_count = 0

    for key, tensor in state_dict.items():
        if key not in model_sd:
            # Let strict=False ignore unexpected keys; we don't include them
            continue
        target = model_sd[key]

        if tuple(tensor.shape) == tuple(target.shape):
            fixed[key] = tensor.to(dtype=target.dtype)
            continue

        # Handle common 1D per-channel vs per-tensor mismatch
        if tensor.ndim == 1 and target.ndim == 1:
            src_len = int(tensor.shape[0])
            dst_len = int(target.shape[0])
            if dst_len == 1 and src_len > 1:
                # Reduce many->one via mean
                reduced = tensor.float().mean().reshape(1).to(dtype=target.dtype)
                fixed[key] = reduced
                adapted_count += 1
                continue
            if src_len == 1 and dst_len > 1:
                # Broadcast one->many
                expanded = tensor.reshape(1).to(dtype=target.dtype).expand(dst_len)
                fixed[key] = expanded
                adapted_count += 1
                continue

        # As a safe default, drop mismatched keys
        dropped_count += 1

    if adapted_count or dropped_count:
        logger.info(
            f"Adjusted state_dict shapes for loading: adapted={adapted_count}, dropped={dropped_count}"
        )

    return fixed



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


def main():
    parser = argparse.ArgumentParser()
    # Where to load FP model checkpoint from (dir with checkpoint-* or the checkpoint dir itself)
    parser.add_argument("--fp_checkpoint_dir", type=str, required=True, help="Path to FP fine-tuning output dir or a specific checkpoint-* dir")
    parser.add_argument("--output_dir", type=str, default="./outputs_qa_qat_w_reshape")
    parser.add_argument("--seed", type=int, default=42)

    # QAT settings
    parser.add_argument("--weight_bits", type=int, default=8)
    parser.add_argument("--activation_bits", type=int, default=8)
    parser.add_argument("--exclude_head", action="store_true", help="Exclude qa_outputs from quantization")
    parser.add_argument("--exclude_pooler", action="store_true", help="Exclude pooler from quantization")
    parser.add_argument("--exclude_embeddings", action="store_true", help="Exclude embeddings from quantization")

    # Data/Trainer settings (same pipeline as FP)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--doc_stride", type=int, default=128)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--kurtosis_lambda", type=float, default=0.05, help="Lambda for W-reshape kurtosis regularization (0 disables)")
    
    # WandB settings
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="bert-qat-squad", help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name (auto-generated if not provided)")
    parser.add_argument("--wandb_tags", type=str, nargs="+", default=None, help="WandB tags")
    parser.add_argument("--visualize_layers", type=str, nargs="+", 
                        default=["layer.0.attention.output.dense", "layer.5.intermediate.dense", "layer.11.output.dense"],
                        help="Layer patterns to visualize")
    parser.add_argument("--visualize_every_n_epochs", type=int, default=1, help="Visualize every N epochs")
    
    args = parser.parse_args()

    set_seed(args.seed)

    # Resolve the last checkpoint directory
    last_ckpt = find_last_checkpoint_dir(args.fp_checkpoint_dir)
    logger.info(f"Loading fine-tuned FP checkpoint from: {last_ckpt}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_dir, f"squad_qat_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    
    # Initialize WandB
    wandb_run = None
    if args.use_wandb:
        if not WANDB_AVAILABLE:
            logger.warning("WandB requested but not available. Install with: pip install wandb")
        else:
            wandb_run_name = args.wandb_run_name or f"qat_w{args.weight_bits}a{args.activation_bits}_{timestamp}"
            
            # Prepare config for WandB
            wandb_config = {
                # Model settings
                "fp_checkpoint": args.fp_checkpoint_dir,
                "weight_bits": args.weight_bits,
                "activation_bits": args.activation_bits,
                "exclude_head": args.exclude_head,
                "exclude_pooler": args.exclude_pooler,
                "exclude_embeddings": args.exclude_embeddings,
                "kurtosis_lambda": args.kurtosis_lambda,
                
                # Training settings
                "num_train_epochs": args.num_train_epochs,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "per_device_eval_batch_size": args.per_device_eval_batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "warmup_ratio": args.warmup_ratio,
                "fp16": args.fp16,
                
                # Data settings
                "max_length": args.max_length,
                "doc_stride": args.doc_stride,
                "dataset": "squad",
                
                # Paths
                "output_dir": out_dir,
                "seed": args.seed,
            }
            
            wandb_tags = args.wandb_tags or [
                f"w{args.weight_bits}bit",
                f"a{args.activation_bits}bit",
                "qat",
                "squad",
                "bert",
            ]
            
            if args.kurtosis_lambda > 0:
                wandb_tags.append("w-reshape")
            
            wandb_run = init_wandb_run(
                project_name=args.wandb_project,
                run_name=wandb_run_name,
                config=wandb_config,
                tags=wandb_tags,
                notes=f"QAT training on SQuAD with {args.weight_bits}-bit weights and {args.activation_bits}-bit activations",
            )

    # Load tokenizer from the FP checkpoint to keep exact vocab/tokenization
    tokenizer = AutoTokenizer.from_pretrained(last_ckpt, use_fast=True)
    tokenizer.padding_side = "right"

    # Build exclude patterns from flags
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

    # Decide how to initialize model depending on checkpoint contents
    model = None
    qat_sd: Optional[Dict[str, torch.Tensor]] = None
    try:
        cand_state_dict = _load_state_dict_from_dir(last_ckpt)
        if _is_qat_state_dict(cand_state_dict):
            qat_sd = cand_state_dict
            logger.info("Detected QAT checkpoint: will initialize QAT architecture and load full state dict.")
        else:
            logger.info("Checkpoint appears FP (no QAT keys found): will load FP then convert to QAT.")
    except Exception:
        # If we cannot read the raw state dict, fallback to FP load path
        pass

    if qat_sd is not None:
        # Initialize from config (no weights), then swap to QAT modules, then load QAT weights
        config = AutoConfig.from_pretrained(last_ckpt)
        model = BertForQuestionAnswering(config)
        model = BertQATConverter.replace_linear_with_qat(
            model,
            weight_bits=args.weight_bits,
            activation_bits=args.activation_bits,
            exclude_patterns=exclude_patterns,
        )
        qat_sd_fixed = _coerce_state_dict_to_model_shapes(model, qat_sd)
        load_res = model.load_state_dict(qat_sd_fixed, strict=False)
        if getattr(load_res, "missing_keys", None):
            logger.warning(f"Missing keys when loading QAT checkpoint: {len(load_res.missing_keys)}")
        if getattr(load_res, "unexpected_keys", None):
            logger.warning(f"Unexpected keys when loading QAT checkpoint: {len(load_res.unexpected_keys)}")
    else:
        # Load fine-tuned FP model and convert to QAT
        model = BertForQuestionAnswering.from_pretrained(last_ckpt)
        model = BertQATConverter.replace_linear_with_qat(
            model,
            weight_bits=args.weight_bits,
            activation_bits=args.activation_bits,
            exclude_patterns=exclude_patterns,
        )

    stats = BertQATConverter.count_qat_layers(model)
    logger.info(f"QAT conversion: {stats['qat_linear']} QATLinear, {stats['regular_linear']} remaining Linear, "
                f"{stats['total_params']:,} params")
    
    # Log model architecture to WandB
    if wandb_run is not None:
        log_model_architecture(model, {
            'qat_layers': stats['qat_linear'],
            'regular_layers': stats['regular_linear'],
            'total_params': stats['total_params'],
        })

    # Data
    raw = load_dataset("squad")
    
    # Full train dataset for training
    train_dataset = raw["train"].map(
        lambda x: prepare_train_features(x, tokenizer, args.max_length, args.doc_stride),
        batched=True,
        remove_columns=raw["train"].column_names,
        desc="Tokenizing train",
    )
    
    # Small train subset for F1 evaluation (1000 examples)
    train_eval_size = min(1000, len(raw["train"]))
    train_eval_examples = raw["train"].select(range(train_eval_size))
    train_eval_dataset = train_eval_examples.map(
        lambda x: prepare_validation_features(x, tokenizer, args.max_length, args.doc_stride),
        batched=True,
        remove_columns=train_eval_examples.column_names,
        desc="Tokenizing train eval subset",
    )
    logger.info(f"Created train eval subset with {train_eval_size} examples")
    
    # Validation dataset (used as dev set)
    eval_examples = raw["validation"]
    eval_dataset = eval_examples.map(
        lambda x: prepare_validation_features(x, tokenizer, args.max_length, args.doc_stride),
        batched=True,
        remove_columns=eval_examples.column_names,
        desc="Tokenizing validation",
    )

    squad_metric = evaluate.load("squad")
    metrics_computer = MetricsComputer(eval_examples, eval_dataset, tokenizer, squad_metric)
    
    # Prepare sample input for visualization
    sample_input = None
    if wandb_run is not None and args.visualize_layers:
        # Get a sample from eval dataset
        sample_idx = 0
        sample_input = {
            'input_ids': torch.tensor([eval_dataset[sample_idx]['input_ids']]),
            'attention_mask': torch.tensor([eval_dataset[sample_idx]['attention_mask']]),
        }
        logger.info(f"Prepared sample input for visualization (shape: {sample_input['input_ids'].shape})")

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
            report_to="wandb" if wandb_run is not None else "none",
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
        )

    # Prepare callbacks
    callbacks = []
    if wandb_run is not None:
        wandb_callback = WandbQATCallback(
            visualize_layers=args.visualize_layers,
            visualize_every_n_epochs=args.visualize_every_n_epochs,
            log_quantization_stats=True,
            sample_input=sample_input,
            compute_train_f1=True,
            train_eval_dataset=train_eval_dataset,
            train_eval_examples=train_eval_examples,
            squad_metric=squad_metric,
        )
        callbacks.append(wandb_callback)
        logger.info(f"Added WandB callback with train F1 and visualization for layers: {args.visualize_layers}")
    
    #trainer = Trainer(
    trainer = KurtosisLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=metrics_computer.compute_metrics,
        kurtosis_lambda=args.kurtosis_lambda,
        callbacks=callbacks,
    )

    logger.info("Starting QAT fine-tuning with HF Trainer...")
    trainer.train()
    logger.info("QAT training completed.")

    # ===== Final Evaluation =====
    logger.info("\n" + "="*80)
    logger.info("FINAL EVALUATION")
    logger.info("="*80)
    
    # 1. Train F1 (full 1000 examples)
    logger.info("\n[1/3] Evaluating on Train subset...")
    train_preds = trainer.predict(train_eval_dataset).predictions
    train_formatted = postprocess_qa_predictions(
        examples=train_eval_examples,
        features=train_eval_dataset,
        predictions=train_preds,
    )
    train_refs = [{"id": ex_id, "answers": ans} 
                  for ex_id, ans in zip(train_eval_examples["id"], train_eval_examples["answers"])]
    train_preds_for_metric = [{"id": k, "prediction_text": v} for k, v in train_formatted.items()]
    train_metrics = squad_metric.compute(predictions=train_preds_for_metric, references=train_refs)
    logger.info(f"Train F1: {train_metrics['f1']:.2f}, EM: {train_metrics['exact_match']:.2f}")
    
    # 2. Dev F1 (validation set)
    logger.info("\n[2/3] Evaluating on Dev set (validation)...")
    dev_preds = trainer.predict(eval_dataset).predictions
    dev_formatted = postprocess_qa_predictions(
        examples=eval_examples,
        features=eval_dataset,
        predictions=dev_preds,
    )
    dev_refs = [{"id": ex_id, "answers": ans} for ex_id, ans in zip(eval_examples["id"], eval_examples["answers"])]
    dev_preds_for_metric = [{"id": k, "prediction_text": v} for k, v in dev_formatted.items()]
    dev_metrics = squad_metric.compute(predictions=dev_preds_for_metric, references=dev_refs)
    logger.info(f"Dev F1: {dev_metrics['f1']:.2f}, EM: {dev_metrics['exact_match']:.2f}")
    
    # 3. Test F1 (using validation as test - in real scenario, use held-out test set)
    logger.info("\n[3/3] Test set evaluation (using validation as test)...")
    test_metrics = dev_metrics  # Same as dev for now
    logger.info(f"Test F1: {test_metrics['f1']:.2f}, EM: {test_metrics['exact_match']:.2f}")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("="*80)
    logger.info(f"Train - F1: {train_metrics['f1']:.2f}, EM: {train_metrics['exact_match']:.2f}")
    logger.info(f"Dev   - F1: {dev_metrics['f1']:.2f}, EM: {dev_metrics['exact_match']:.2f}")
    logger.info(f"Test  - F1: {test_metrics['f1']:.2f}, EM: {test_metrics['exact_match']:.2f}")
    logger.info("="*80 + "\n")
    
    # Use dev metrics as eval_metrics for backward compatibility
    eval_metrics = dev_metrics
    
    # Log final metrics to WandB
    if wandb_run is not None:
        import wandb
        
        # Log all final metrics
        wandb.log({
            'final/train_f1': train_metrics['f1'],
            'final/train_em': train_metrics['exact_match'],
            'final/dev_f1': dev_metrics['f1'],
            'final/dev_em': dev_metrics['exact_match'],
            'final/test_f1': test_metrics['f1'],
            'final/test_em': test_metrics['exact_match'],
        })
        
        # Get best metrics from trainer state
        best_metrics = {
            'f1': dev_metrics.get('f1', 0.0),
            'exact_match': dev_metrics.get('exact_match', 0.0),
        }
        if trainer.state.best_metric is not None:
            best_metrics['f1'] = trainer.state.best_metric
        
        log_training_summary(
            final_metrics=dev_metrics,
            best_metrics=best_metrics,
            training_args=training_args,
        )
        
        # Set summary values
        wandb.run.summary['train_f1'] = train_metrics['f1']
        wandb.run.summary['dev_f1'] = dev_metrics['f1']
        wandb.run.summary['test_f1'] = test_metrics['f1']

    # Save artifacts
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    # Save all metrics to file
    with open(os.path.join(out_dir, "all_metrics.txt"), "w") as f:
        f.write("="*60 + "\n")
        f.write("FINAL EVALUATION RESULTS\n")
        f.write("="*60 + "\n\n")
        
        f.write("Train Set (1000 examples):\n")
        f.write(f"  F1 Score: {train_metrics['f1']:.4f}\n")
        f.write(f"  Exact Match: {train_metrics['exact_match']:.4f}\n\n")
        
        f.write("Dev Set (validation):\n")
        f.write(f"  F1 Score: {dev_metrics['f1']:.4f}\n")
        f.write(f"  Exact Match: {dev_metrics['exact_match']:.4f}\n\n")
        
        f.write("Test Set:\n")
        f.write(f"  F1 Score: {test_metrics['f1']:.4f}\n")
        f.write(f"  Exact Match: {test_metrics['exact_match']:.4f}\n\n")
        
        f.write("="*60 + "\n")
        f.write("Configuration:\n")
        f.write(f"  Weight bits: {args.weight_bits}\n")
        f.write(f"  Activation bits: {args.activation_bits}\n")
        f.write(f"  Learning rate: {args.learning_rate}\n")
        f.write(f"  Epochs: {args.num_train_epochs}\n")
        f.write(f"  Kurtosis lambda: {args.kurtosis_lambda}\n")
        f.write("="*60 + "\n")

    print("\n" + "="*80)
    print("FINAL RESULTS:")
    print(f"  Train F1: {train_metrics['f1']:.2f} | Dev F1: {dev_metrics['f1']:.2f} | Test F1: {test_metrics['f1']:.2f}")
    print(f"Artifacts saved to: {out_dir}")
    print("="*80)
    
    # Finish WandB run
    if wandb_run is not None:
        try:
            import wandb
            wandb.finish()
            logger.info("✅ WandB run finished")
        except Exception as e:
            logger.warning(f"Error finishing WandB run: {e}")


if __name__ == "__main__":
    main()