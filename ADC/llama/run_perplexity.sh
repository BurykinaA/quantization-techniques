#!/bin/bash
# ==============================================================================
# Measure LLaMA Perplexity on WikiText-2
# ==============================================================================
#
# This script measures baseline (full precision) perplexity for LLaMA models.
# Use this to establish baseline metrics before quantization.
#
# Usage:
#   ./ADC/llama/run_perplexity.sh
#
# ==============================================================================

set -e  # Exit on error

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Model to evaluate (can be changed via command line: MODEL_NAME=... ./run_perplexity.sh)
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-1B}"

# Precision
TORCH_DTYPE="${TORCH_DTYPE:-float16}"

# Evaluation settings
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
MAX_LENGTH="${MAX_LENGTH:-512}"
MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-}"  # Empty = evaluate all
DATASET_SPLIT="${DATASET_SPLIT:-test}"

# WandB settings (set WANDB_PROJECT to enable)
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"

# ==============================================================================
# BUILD COMMAND
# ==============================================================================

echo "========================================"
echo "LLaMA Perplexity Measurement"
echo "========================================"
echo "Model:        $MODEL_NAME"
echo "Dtype:        $TORCH_DTYPE"
echo "Batch size:   $EVAL_BATCH_SIZE"
echo "Max length:   $MAX_LENGTH"
echo "Dataset:      WikiText-2 ($DATASET_SPLIT)"
if [ -n "$MAX_EVAL_BATCHES" ]; then
    echo "Max batches:  $MAX_EVAL_BATCHES"
else
    echo "Max batches:  all"
fi
if [ -n "$WANDB_PROJECT" ]; then
    echo "WandB:        $WANDB_PROJECT"
fi
echo "========================================"

CMD="python ADC/llama/runs/measure_perplexity.py"
CMD="$CMD --model_name \"$MODEL_NAME\""
CMD="$CMD --torch_dtype $TORCH_DTYPE"
CMD="$CMD --eval_batch_size $EVAL_BATCH_SIZE"
CMD="$CMD --max_length $MAX_LENGTH"
CMD="$CMD --dataset_split $DATASET_SPLIT"

if [ -n "$MAX_EVAL_BATCHES" ]; then
    CMD="$CMD --max_eval_batches $MAX_EVAL_BATCHES"
fi

if [ -n "$WANDB_PROJECT" ]; then
    CMD="$CMD --wandb_project \"$WANDB_PROJECT\""
fi

if [ -n "$WANDB_RUN_NAME" ]; then
    CMD="$CMD --wandb_run_name \"$WANDB_RUN_NAME\""
fi

# ==============================================================================
# RUN
# ==============================================================================

echo ""
echo "Running: $CMD"
echo ""

eval $CMD
EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "========================================"
    echo "✓ Perplexity measurement complete"
    echo "========================================"
else
    echo "========================================"
    echo "✗ Failed with exit code $EXIT_CODE"
    echo "========================================"
fi

exit $EXIT_CODE
