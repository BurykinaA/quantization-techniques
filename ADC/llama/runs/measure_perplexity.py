#!/usr/bin/env python3
"""
Measure perplexity for LLaMA models on WikiText-2 dataset.

This script can be used to:
1. Measure baseline (full precision) perplexity
2. Compare different models or configurations
3. Log results to WandB for tracking

Usage:
    # Basic usage
    python measure_perplexity.py --model_name "meta-llama/Llama-3.2-1B"
    
    # With WandB logging
    python measure_perplexity.py --model_name "meta-llama/Llama-3.2-1B" --wandb_project "llama-perplexity"
    
    # Quick test with fewer batches
    python measure_perplexity.py --model_name "meta-llama/Llama-3.2-1B" --max_eval_batches 50
"""

import argparse
import math
import logging
from datetime import datetime

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from torch.utils.data import DataLoader

# WandB import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, logging will be disabled")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_perplexity(model, dataloader, device, max_batches: int = None, desc: str = "Evaluating"):
    """
    Compute perplexity for a causal language model.
    
    Args:
        model: The language model
        dataloader: DataLoader with tokenized data
        device: Device to run on
        max_batches: Maximum number of batches to evaluate (None = all)
        desc: Description for progress bar
    
    Returns:
        dict with perplexity, avg_loss, total_tokens, num_batches
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0
    
    total_batches = len(dataloader) if max_batches is None else min(max_batches, len(dataloader))
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, total=total_batches):
            if max_batches is not None and num_batches >= max_batches:
                break
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # For causal LM, labels = input_ids (shifted internally by the model)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            
            # Count non-padding tokens
            num_tokens = attention_mask.sum().item()
            total_loss += outputs.loss.item() * num_tokens
            total_tokens += num_tokens
            num_batches += 1
    
    # Compute perplexity
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    
    return {
        "perplexity": perplexity,
        "avg_loss": avg_loss,
        "total_tokens": total_tokens,
        "num_batches": num_batches,
    }


def main():
    parser = argparse.ArgumentParser(description="Measure LLaMA perplexity on WikiText-2")
    
    # Model settings
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B",
                       help="HuggingFace model name or path")
    parser.add_argument("--torch_dtype", type=str, default="float16",
                       choices=["float16", "bfloat16", "float32"],
                       help="Model dtype")
    parser.add_argument("--seed", type=int, default=42)
    
    # Evaluation settings
    parser.add_argument("--eval_batch_size", type=int, default=4,
                       help="Batch size for evaluation")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--max_eval_batches", type=int, default=None,
                       help="Maximum batches for evaluation (None = all)")
    parser.add_argument("--dataset_split", type=str, default="test",
                       choices=["validation", "test"],
                       help="Which split to evaluate on")
    
    # WandB settings
    parser.add_argument("--wandb_project", type=str, default=None,
                       help="WandB project name (None = disable WandB)")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="WandB run name (auto-generated if not provided)")
    parser.add_argument("--tags", type=str, nargs="+", default=None,
                       help="Tags for WandB run")
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    # Determine dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.torch_dtype]
    
    # Initialize WandB
    use_wandb = WANDB_AVAILABLE and args.wandb_project is not None
    if use_wandb:
        model_short_name = args.model_name.split("/")[-1]
        run_name = args.wandb_run_name or f"ppl_{model_short_name}_{args.torch_dtype}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            tags=args.tags,
            config={
                "model_name": args.model_name,
                "torch_dtype": args.torch_dtype,
                "eval_batch_size": args.eval_batch_size,
                "max_length": args.max_length,
                "max_eval_batches": args.max_eval_batches,
                "dataset_split": args.dataset_split,
                "seed": args.seed,
            }
        )
        logger.info(f"WandB initialized: project={args.wandb_project}, run={run_name}")
    
    # =========================================================================
    # Load Model
    # =========================================================================
    logger.info("="*80)
    logger.info(f"Loading model: {args.model_name}")
    logger.info("="*80)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # LLaMA uses left padding for generation
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    
    # Get the actual device
    if hasattr(model, 'device'):
        device = model.device
    elif hasattr(model, 'hf_device_map'):
        device = next(iter(model.hf_device_map.values())) if model.hf_device_map else device
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded: {total_params:,} parameters, dtype={torch_dtype}, device={device}")
    
    # =========================================================================
    # Load Dataset
    # =========================================================================
    logger.info("="*80)
    logger.info("Loading WikiText-2 dataset")
    logger.info("="*80)
    
    raw = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    def tokenize_function(examples):
        texts = [t for t in examples["text"] if t.strip()]
        if not texts:
            return {"input_ids": [], "attention_mask": []}
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
            return_tensors=None,
        )
        return tokenized
    
    # Prepare evaluation dataset
    eval_split = raw[args.dataset_split]
    eval_dataset = eval_split.filter(lambda x: len(x["text"].strip()) > 50)
    eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc=f"Tokenizing {args.dataset_split} data",
    )
    eval_dataset = eval_dataset.filter(lambda x: len(x["input_ids"]) > 0)
    
    logger.info(f"Evaluation dataset: {len(eval_dataset)} samples from '{args.dataset_split}' split")
    
    def eval_collator(features):
        batch = {
            "input_ids": torch.tensor([f["input_ids"] for f in features]),
            "attention_mask": torch.tensor([f["attention_mask"] for f in features]),
        }
        return batch
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=eval_collator,
    )
    
    # =========================================================================
    # Compute Perplexity
    # =========================================================================
    logger.info("="*80)
    logger.info("Computing Perplexity")
    logger.info("="*80)
    
    metrics = compute_perplexity(
        model, eval_loader, device,
        max_batches=args.max_eval_batches,
        desc=f"Perplexity ({args.model_name.split('/')[-1]})"
    )
    
    # =========================================================================
    # Results
    # =========================================================================
    logger.info("="*80)
    logger.info("RESULTS")
    logger.info("="*80)
    logger.info(f"Model:         {args.model_name}")
    logger.info(f"Dtype:         {args.torch_dtype}")
    logger.info(f"Dataset:       WikiText-2 ({args.dataset_split})")
    logger.info(f"Perplexity:    {metrics['perplexity']:.4f}")
    logger.info(f"Avg Loss:      {metrics['avg_loss']:.4f}")
    logger.info(f"Total Tokens:  {metrics['total_tokens']:,}")
    logger.info(f"Num Batches:   {metrics['num_batches']}")
    logger.info("="*80)
    
    # Log to WandB
    if use_wandb:
        wandb.log({
            "perplexity": metrics['perplexity'],
            "avg_loss": metrics['avg_loss'],
            "total_tokens": metrics['total_tokens'],
            "num_batches": metrics['num_batches'],
        })
        
        # Summary
        wandb.run.summary["perplexity"] = metrics['perplexity']
        wandb.run.summary["avg_loss"] = metrics['avg_loss']
        wandb.run.summary["total_params"] = total_params
        
        wandb.finish()
        logger.info("Results logged to WandB")
    
    # Return metrics for programmatic use
    return metrics


if __name__ == "__main__":
    main()
