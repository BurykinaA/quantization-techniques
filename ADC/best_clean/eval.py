"""
Evaluation utilities: perplexity (sliding window) and latency measurement.

Both functions are self-contained and work with any HuggingFace-compatible model.
"""

import math
import time
import logging

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Perplexity — standard sliding-window approach
# ─────────────────────────────────────────────────────────────────────────────

def load_eval_encodings(dataset_name: str, split: str, tokenizer, max_samples: int = 1000):
    """
    Load a dataset and tokenize it as one long concatenated sequence.

    This is the standard approach for paper-quality perplexity:
    no per-sample truncation or padding, pure continuous text.

    Args:
        dataset_name: "wikitext2" or "c4"
        split:        "test" for WikiText2, "validation" for C4
        tokenizer:    HuggingFace tokenizer
        max_samples:  max docs to load (C4 only; WikiText2 uses full test set)

    Returns:
        dict with 'input_ids' tensor of shape [1, total_tokens], or None on failure
    """
    logger.info(f"Loading {dataset_name} ({split}) ...")

    try:
        if dataset_name == "wikitext2":
            raw = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
            text = "\n\n".join(t for t in raw["text"] if t.strip())

        elif dataset_name == "c4":
            # C4 has no "test" split; map it to "validation"
            actual_split = "validation" if split == "test" else split
            raw = load_dataset("allenai/c4", "en", split=actual_split, streaming=True)
            texts = []
            for i, example in enumerate(raw):
                if i >= max_samples:
                    break
                if example["text"].strip():
                    texts.append(example["text"])
            text = "\n\n".join(texts)
            logger.info(f"  Loaded {len(texts)} C4 documents")

        else:
            raise ValueError(f"Unknown dataset '{dataset_name}'. Use 'wikitext2' or 'c4'.")

    except Exception as exc:
        logger.warning(f"Could not load {dataset_name}: {exc}")
        return None

    encodings = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    logger.info(f"  {dataset_name}: {encodings['input_ids'].size(1):,} tokens")
    return encodings


def compute_perplexity(
    model,
    encodings,
    device,
    max_length: int = 2048,
    stride: int = 1024,
    desc: str = "PPL",
) -> dict:
    """
    Compute perplexity with the standard sliding-window method.

    The corpus is one long sequence. A window of `max_length` tokens slides
    forward by `stride` tokens at each step. Only the `stride` new tokens at
    the end of each window contribute to the loss (the earlier tokens provide
    context but are masked out), so no token is counted twice.

    Formula:
        PPL = exp( Σ_i  loss_i · target_len_i  /  Σ_i  target_len_i )

    Args:
        model:      any PyTorch model that accepts input_ids + labels
        encodings:  output of load_eval_encodings()
        device:     torch.device
        max_length: context window size (should match model's max_position_embeddings)
        stride:     tokens to advance per step (default 1024 = 50% overlap)
        desc:       progress bar label

    Returns:
        dict with perplexity, avg_loss, total_tokens, num_windows
    """
    model.eval()

    input_ids = encodings["input_ids"]
    seq_len = input_ids.size(1)
    num_windows = max(1, (seq_len - max_length) // stride + 1)

    logger.info(f"  {desc}: {seq_len:,} tokens, window={max_length}, stride={stride}")

    nlls = []
    total_tokens = 0
    prev_end = 0

    with torch.no_grad():
        for begin in tqdm(range(0, seq_len, stride), desc=desc, total=num_windows):
            end = min(begin + max_length, seq_len)

            ids = input_ids[:, begin:end].to(device)
            target_len = end - prev_end  # only the "new" tokens count

            labels = ids.clone()
            labels[:, :-target_len] = -100  # mask already-counted context

            loss = model(input_ids=ids, labels=labels).loss
            nlls.append(loss.item() * target_len)
            total_tokens += target_len
            prev_end = end

            if end >= seq_len:
                break

    avg_loss = sum(nlls) / total_tokens
    ppl = math.exp(avg_loss)

    return {
        "perplexity": ppl,
        "avg_loss": avg_loss,
        "total_tokens": total_tokens,
        "num_windows": len(nlls),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Latency measurement
# ─────────────────────────────────────────────────────────────────────────────

def measure_latency(
    model,
    device,
    seq_len: int = 512,
    batch_size: int = 1,
    n_warmup: int = 5,
    n_runs: int = 20,
) -> dict:
    """
    Measure forward-pass latency (batch_size × seq_len tokens per call).

    Uses CUDA Events on GPU for microsecond-accurate timing that accounts for
    asynchronous kernel launch. Falls back to time.perf_counter() on CPU.

    Args:
        model:      quantized model (after ADC conversion / LoRA)
        device:     torch.device
        seq_len:    sequence length to benchmark (512 is typical for comparison)
        batch_size: batch size (1 for single-query latency)
        n_warmup:   warm-up calls (discarded)
        n_runs:     timed calls

    Returns:
        dict with latency_mean_ms, latency_std_ms, latency_p50_ms,
        latency_p95_ms, throughput_tok_s
    """
    model.eval()
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
    n_tokens = batch_size * seq_len
    use_cuda = device.type == "cuda"

    with torch.no_grad():
        # Warm-up
        for _ in range(n_warmup):
            model(input_ids=input_ids)
        if use_cuda:
            torch.cuda.synchronize(device)

        # Timed runs
        latencies_ms = []
        for _ in range(n_runs):
            if use_cuda:
                t_start = torch.cuda.Event(enable_timing=True)
                t_end = torch.cuda.Event(enable_timing=True)
                t_start.record()
                model(input_ids=input_ids)
                t_end.record()
                torch.cuda.synchronize(device)
                latencies_ms.append(t_start.elapsed_time(t_end))
            else:
                t0 = time.perf_counter()
                model(input_ids=input_ids)
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(latencies_ms)
    mean_ms = float(arr.mean())
    return {
        "latency_mean_ms":  mean_ms,
        "latency_std_ms":   float(arr.std()),
        "latency_p50_ms":   float(np.percentile(arr, 50)),
        "latency_p95_ms":   float(np.percentile(arr, 95)),
        "throughput_tok_s": float(n_tokens / (mean_ms / 1000.0)),
        "batch_size":       batch_size,
        "seq_len":          seq_len,
    }
