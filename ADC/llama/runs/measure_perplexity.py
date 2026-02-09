#!/usr/bin/env python3
"""
Measure perplexity for LLaMA models on WikiText-2 / C4 datasets.

Uses the STANDARD sliding window approach for proper perplexity evaluation,
matching methodology used in papers like GPTQ, AWQ, FlatQuant, etc.

Key features:
- Concatenates all text into one long sequence (no per-sample truncation)
- Uses sliding window with configurable stride
- No padding - pure continuous text evaluation
- Supports both WikiText-2 and C4

Usage:
    # Basic usage (WikiText-2)
    python measure_perplexity.py --model_name "meta-llama/Llama-3.2-1B"
    
    # With longer context (recommended for accuracy)
    python measure_perplexity.py --model_name "meta-llama/Llama-3.1-8B" --max_length 2048
    
    # Evaluate on C4
    python measure_perplexity.py --model_name "meta-llama/Llama-3.2-1B" --dataset c4
    
    # With WandB logging
    python measure_perplexity.py --model_name "meta-llama/Llama-3.2-1B" --wandb_project "llama-ppl"
"""

import argparse
import math
import logging

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# WandB import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, logging will be disabled")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_perplexity_sliding_window(
    model, 
    encodings, 
    device,
    max_length: int = 2048,
    stride: int = None,
    desc: str = "Evaluating"
):
    """
    Compute perplexity using sliding window approach (standard for papers).
    
    This is the proper way to evaluate perplexity on language models:
    1. Concatenate all text into one long sequence
    2. Use sliding window with stride
    3. Only count loss on the "new" tokens (stride portion) to avoid double counting
    
    Args:
        model: The language model
        encodings: Tokenized text (dict with 'input_ids' tensor of shape [1, seq_len])
        device: Device to run on
        max_length: Context window size (should match model's context length)
        stride: How many tokens to advance each step (default: max_length // 2)
        desc: Description for progress bar
    
    Returns:
        dict with perplexity, avg_loss, total_tokens, num_windows
    """
    if stride is None:
        stride = max_length // 2  # 50% overlap is common
    
    model.eval()
    
    input_ids = encodings["input_ids"]
    seq_len = input_ids.size(1)
    
    logger.info(f"Total tokens in corpus: {seq_len:,}")
    logger.info(f"Context window: {max_length}, Stride: {stride}")
    
    nlls = []  # Negative log likelihoods
    total_tokens = 0
    
    # Calculate number of windows
    num_windows = max(1, (seq_len - max_length) // stride + 1)
    
    prev_end_loc = 0
    with torch.no_grad():
        for begin_loc in tqdm(range(0, seq_len, stride), desc=desc, total=num_windows):
            end_loc = min(begin_loc + max_length, seq_len)
            
            # Get the window
            input_ids_window = input_ids[:, begin_loc:end_loc].to(device)
            
            # Target length: only count loss on the "new" tokens
            # This avoids double-counting when using overlapping windows
            target_len = end_loc - prev_end_loc
            
            # Create labels: -100 for tokens we've already counted
            labels = input_ids_window.clone()
            labels[:, :-target_len] = -100  # Mask already-counted tokens
            
            outputs = model(
                input_ids=input_ids_window,
                labels=labels,
            )
            
            # Accumulate the loss weighted by number of target tokens
            neg_log_likelihood = outputs.loss * target_len
            nlls.append(neg_log_likelihood.item())
            total_tokens += target_len
            
            prev_end_loc = end_loc
            
            # Stop if we've processed the whole sequence
            if end_loc >= seq_len:
                break
    
    # Compute average loss and perplexity
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


def load_and_tokenize_dataset(dataset_name: str, split: str, tokenizer, max_samples: int = None):
    """
    Load and tokenize a dataset, concatenating all text.
    
    Args:
        dataset_name: "wikitext2" or "c4"
        split: "train", "validation", or "test"
        tokenizer: Tokenizer to use
        max_samples: Maximum samples to use (for C4 which is huge)
    
    Returns:
        dict with 'input_ids' tensor of shape [1, total_tokens]
    """
    logger.info(f"Loading {dataset_name} ({split} split)...")
    
    if dataset_name == "wikitext2":
        raw = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        # Concatenate all text
        text = "\n\n".join([t for t in raw["text"] if t.strip()])
    elif dataset_name == "c4":
        # C4 is huge, use streaming and limit samples
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
    
    # Tokenize the entire text as one sequence
    encodings = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,  # Don't add BOS/EOS between concatenated texts
    )
    
    logger.info(f"Total tokens: {encodings['input_ids'].size(1):,}")
    
    return encodings


def main():
    parser = argparse.ArgumentParser(
        description="Measure LLaMA perplexity using standard sliding window approach"
    )
    
    # Model settings
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B",
                       help="HuggingFace model name or path")
    parser.add_argument("--torch_dtype", type=str, default="float16",
                       choices=["float16", "bfloat16", "float32"],
                       help="Model dtype")
    parser.add_argument("--seed", type=int, default=42)
    
    # Dataset settings
    parser.add_argument("--dataset", type=str, default="wikitext2",
                       choices=["wikitext2", "c4"],
                       help="Dataset to evaluate on")
    parser.add_argument("--dataset_split", type=str, default="test",
                       choices=["train", "validation", "test"],
                       help="Which split to evaluate on")
    parser.add_argument("--max_samples", type=int, default=1000,
                       help="Max samples for C4 (ignored for WikiText-2)")
    
    # Evaluation settings
    parser.add_argument("--max_length", type=int, default=2048,
                       help="Context window size (should match model's context length)")
    parser.add_argument("--stride", type=int, default=None,
                       help="Stride for sliding window (default: max_length // 2)")
    
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
        run_name = args.wandb_run_name or f"ppl_{model_short_name}_{args.dataset}_{args.torch_dtype}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            tags=args.tags,
            config={
                "model_name": args.model_name,
                "torch_dtype": args.torch_dtype,
                "dataset": args.dataset,
                "dataset_split": args.dataset_split,
                "max_length": args.max_length,
                "stride": args.stride,
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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    
    # Get the actual device (for multi-GPU, get the first one)
    if hasattr(model, 'device'):
        device = model.device
    elif hasattr(model, 'hf_device_map'):
        devices = list(model.hf_device_map.values())
        device = devices[0] if devices else device
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded: {total_params:,} parameters, dtype={torch_dtype}")
    
    # Get model's max context length
    model_max_length = getattr(model.config, 'max_position_embeddings', 4096)
    if args.max_length > model_max_length:
        logger.warning(f"max_length ({args.max_length}) > model's max ({model_max_length}), using {model_max_length}")
        args.max_length = model_max_length
    
    # =========================================================================
    # Load and Tokenize Dataset
    # =========================================================================
    logger.info("="*80)
    logger.info(f"Loading {args.dataset.upper()} dataset")
    logger.info("="*80)
    
    encodings = load_and_tokenize_dataset(
        args.dataset,
        args.dataset_split,
        tokenizer,
        max_samples=args.max_samples if args.dataset == "c4" else None
    )
    
    # =========================================================================
    # Compute Perplexity
    # =========================================================================
    logger.info("="*80)
    logger.info("Computing Perplexity (Sliding Window)")
    logger.info("="*80)
    
    metrics = compute_perplexity_sliding_window(
        model, 
        encodings, 
        device,
        max_length=args.max_length,
        stride=args.stride,
        desc=f"Perplexity ({args.model_name.split('/')[-1]})"
    )
    
    # =========================================================================
    # Results
    # =========================================================================
    logger.info("="*80)
    logger.info("RESULTS")
    logger.info("="*80)
    logger.info(f"Model:           {args.model_name}")
    logger.info(f"Dtype:           {args.torch_dtype}")
    logger.info(f"Dataset:         {args.dataset.upper()} ({args.dataset_split})")
    logger.info(f"Context window:  {metrics['max_length']}")
    logger.info(f"Stride:          {metrics['stride']}")
    logger.info(f"")
    logger.info(f"Perplexity:      {metrics['perplexity']:.4f}")
    logger.info(f"Avg Loss:        {metrics['avg_loss']:.4f}")
    logger.info(f"Total Tokens:    {metrics['total_tokens']:,}")
    logger.info(f"Num Windows:     {metrics['num_windows']}")
    logger.info("="*80)
    
    # Log to WandB
    if use_wandb:
        wandb.log({
            "perplexity": metrics['perplexity'],
            "avg_loss": metrics['avg_loss'],
            "total_tokens": metrics['total_tokens'],
            "num_windows": metrics['num_windows'],
        })
        
        # Summary
        wandb.run.summary["perplexity"] = metrics['perplexity']
        wandb.run.summary["avg_loss"] = metrics['avg_loss']
        wandb.run.summary["total_params"] = total_params
        wandb.run.summary["max_length"] = metrics['max_length']
        wandb.run.summary["stride"] = metrics['stride']
        
        wandb.finish()
        logger.info("Results logged to WandB")
    
    # Return metrics for programmatic use
    return metrics


if __name__ == "__main__":
    main()
