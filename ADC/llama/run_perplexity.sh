#!/bin/bash
# ==============================================================================
# Measure LLaMA Perplexity (Standard Sliding Window Method)
# ==============================================================================
# Uses the same methodology as papers like GPTQ, AWQ, FlatQuant:
# - Concatenate all text into one long sequence
# - Use sliding window with overlap
# - No padding
#
# Supports: meta-llama/Llama-3.1-8B, meta-llama/Llama-3.2-3B, meta-llama/Llama-3.2-1B

# ============================================================
# MODEL CONFIGURATION
# ============================================================
MODEL_NAME="meta-llama/Llama-3.2-1B"  # Options:
                                       # - meta-llama/Llama-3.2-1B (smallest, fastest)
                                       # - meta-llama/Llama-3.2-3B (medium)
                                       # - meta-llama/Llama-3.1-8B (largest)

# ============================================================
# Model Loading Settings
# ============================================================
TORCH_DTYPE="float16"            # Options: float16, bfloat16, float32

# ============================================================
# Dataset Settings
# ============================================================
DATASET="wikitext2"              # Options: wikitext2, c4
DATASET_SPLIT="test"             # Options: train, validation, test
                                 # Standard papers use "test" for final evaluation
MAX_SAMPLES=1000                 # Only for C4 (which is huge), ignored for WikiText-2

# ============================================================
# Evaluation Settings (Sliding Window)
# ============================================================
MAX_LENGTH=2048                  # Context window size (2048 is standard for papers)
                                 # LLaMA-3 supports up to 8192, but 2048 is common
STRIDE=""                        # Sliding window stride (empty = max_length // 2)
                                 # Non-overlapping: set STRIDE=$MAX_LENGTH

# ============================================================
# WandB Settings (leave empty to disable)
# ============================================================
WANDB_PROJECT="llama-fp"         # Set to enable WandB logging
WANDB_RUN_NAME=""                # Auto-generated if empty

# Seed for reproducibility
SEED=42

# ============================================================
# Display Configuration
# ============================================================
MODEL_SHORT_NAME=$(echo $MODEL_NAME | sed 's/.*\///')

echo "========================================"
echo "LLaMA Perplexity (Sliding Window)"
echo "========================================"
echo "Model:             $MODEL_NAME"
echo "Dtype:             $TORCH_DTYPE"
echo ""
echo "Evaluation:"
echo "  Dataset:         $DATASET ($DATASET_SPLIT)"
echo "  Context window:  $MAX_LENGTH"
if [ -n "$STRIDE" ]; then
    echo "  Stride:          $STRIDE"
else
    echo "  Stride:          $((MAX_LENGTH / 2)) (default: half context)"
fi
if [ "$DATASET" = "c4" ]; then
    echo "  Max samples:     $MAX_SAMPLES"
fi
echo ""
if [ -n "$WANDB_PROJECT" ]; then
    echo "WandB:             $WANDB_PROJECT"
else
    echo "WandB:             disabled"
fi
echo "========================================"
echo ""

# ============================================================
# Build Command
# ============================================================
CMD="python ADC/llama/runs/measure_perplexity.py \
    --model_name \"$MODEL_NAME\" \
    --torch_dtype $TORCH_DTYPE \
    --dataset $DATASET \
    --dataset_split $DATASET_SPLIT \
    --max_length $MAX_LENGTH \
    --seed $SEED"

# Add optional arguments
if [ -n "$STRIDE" ]; then
    CMD="$CMD --stride $STRIDE"
fi

if [ "$DATASET" = "c4" ]; then
    CMD="$CMD --max_samples $MAX_SAMPLES"
fi

if [ -n "$WANDB_PROJECT" ]; then
    CMD="$CMD --wandb_project \"$WANDB_PROJECT\""
fi

if [ -n "$WANDB_RUN_NAME" ]; then
    CMD="$CMD --wandb_run_name \"$WANDB_RUN_NAME\""
fi

# ============================================================
# Run
# ============================================================
echo "Running perplexity measurement..."
echo ""
eval $CMD

EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Perplexity measurement complete!"
    echo "========================================"
    if [ -n "$WANDB_PROJECT" ]; then
        echo ""
        echo "WandB: https://wandb.ai/your-username/$WANDB_PROJECT"
    fi
else
    echo "Failed with exit code $EXIT_CODE"
    echo "========================================"
    echo ""
    echo "Common issues:"
    echo "  - OOM: Reduce MAX_LENGTH (try 1024 or 512)"
    echo "  - Auth: Run 'huggingface-cli login' for gated models"
    echo "  - CUDA: Check GPU availability with 'nvidia-smi'"
fi
echo "========================================"
