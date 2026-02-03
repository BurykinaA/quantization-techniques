#!/bin/bash
# ==============================================================================
# Measure LLaMA Perplexity on WikiText-2 / C4
# ==============================================================================
# Use this script to measure baseline (full precision) perplexity before PTQ.
# Supports: meta-llama/Llama-3.1-8B, meta-llama/Llama-3.2-3B, meta-llama/Llama-3.2-1B

# ============================================================
# MODEL CONFIGURATION - Change this to use different models
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
# Evaluation Settings
# ============================================================
EVAL_BATCH_SIZE=4                # Batch size for perplexity evaluation
MAX_LENGTH=512                   # Maximum sequence length
MAX_EVAL_BATCHES=100              # Maximum batches for evaluation (empty = all)
DATASET_SPLIT="validation"       # Dataset split: validation or test (PTQ uses validation)

# ============================================================
# WandB Settings (leave empty to disable)
# ============================================================
WANDB_PROJECT="llama-fp"                 # Set to enable WandB logging, e.g., "llama-perplexity"
WANDB_RUN_NAME="fp_${MODEL_NAME}"                # Auto-generated if empty

# Seed for reproducibility
SEED=42

# ============================================================
# Display Configuration
# ============================================================
MODEL_SHORT_NAME=$(echo $MODEL_NAME | sed 's/.*\///')

echo "========================================"
echo "LLaMA Perplexity Measurement"
echo "========================================"
echo "Model:             $MODEL_NAME"
echo "Dtype:             $TORCH_DTYPE"
echo ""
echo "Evaluation:"
echo "  Dataset:         WikiText-2 ($DATASET_SPLIT)"
echo "  Batch size:      $EVAL_BATCH_SIZE"
echo "  Max length:      $MAX_LENGTH"
if [ -n "$MAX_EVAL_BATCHES" ]; then
    echo "  Max batches:     $MAX_EVAL_BATCHES"
else
    echo "  Max batches:     all"
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
    --eval_batch_size $EVAL_BATCH_SIZE \
    --max_length $MAX_LENGTH \
    --dataset_split $DATASET_SPLIT \
    --seed $SEED"

# Add optional arguments
if [ -n "$MAX_EVAL_BATCHES" ]; then
    CMD="$CMD --max_eval_batches $MAX_EVAL_BATCHES"
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
    echo "✓ Perplexity measurement complete!"
    echo "========================================"
    if [ -n "$WANDB_PROJECT" ]; then
        echo ""
        echo "🔗 WandB: https://wandb.ai/your-username/$WANDB_PROJECT"
    fi
else
    echo "✗ Failed with exit code $EXIT_CODE"
    echo "========================================"
    echo ""
    echo "Common issues:"
    echo "  - OOM: Reduce EVAL_BATCH_SIZE"
    echo "  - Auth: Run 'huggingface-cli login' for gated models"
    echo "  - CUDA: Check GPU availability with 'nvidia-smi'"
fi
echo "========================================"
