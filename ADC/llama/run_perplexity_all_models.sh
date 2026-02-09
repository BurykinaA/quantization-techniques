#!/bin/bash
# ==============================================================================
# Measure LLaMA Perplexity for ALL Models on ALL Datasets
# ==============================================================================
# Runs perplexity evaluation for:
#   - meta-llama/Llama-3.2-1B
#   - meta-llama/Llama-3.2-3B
#   - meta-llama/Llama-3.1-8B
# 
# On both datasets:
#   - WikiText-2 (test split)
#   - C4 (validation split)
#
# All results are logged to a single WandB project for easy comparison.

# ============================================================
# CONFIGURATION
# ============================================================
TORCH_DTYPE="float16"            # Options: float16, bfloat16, float32

# Evaluation Settings (Sliding Window - Standard for Papers)
MAX_LENGTH=2048                  # Context window size (2048 is standard)
STRIDE=""                        # Empty = max_length // 2 (50% overlap)

# Dataset settings
WIKITEXT_SPLIT="test"            # Standard for papers
C4_SPLIT="test"                  # Use test split (same as papers)
C4_MAX_SAMPLES=10000                # No limit - use full validation split

# Visualization Settings
VISUALIZE=false                  # Set to true to enable visualization
VIZ_NUM_SAMPLES=10               # Number of samples for visualization
VIZ_SEQ_LENGTH=2048              # Sequence length for visualization

# WandB Settings
WANDB_PROJECT="llama-fp-benchmark"  # All runs go to this project

# Seed for reproducibility
SEED=42

# ============================================================
# MODELS TO EVALUATE
# ============================================================
MODELS=(
    "meta-llama/Llama-3.2-1B"
    "meta-llama/Llama-3.2-3B"
    "meta-llama/Llama-3.1-8B"
)

# ============================================================
# DATASETS TO EVALUATE
# ============================================================
DATASETS=(
    "wikitext2"
    "c4"
)

# ============================================================
# RUN ALL COMBINATIONS
# ============================================================
echo "========================================================"
echo "LLaMA Full Precision Perplexity Benchmark"
echo "========================================================"
echo "Models:    ${MODELS[*]}"
echo "Datasets:  ${DATASETS[*]}"
echo "WandB:     $WANDB_PROJECT"
echo "Context:   $MAX_LENGTH tokens"
echo "Dtype:     $TORCH_DTYPE"
if [ "$VISUALIZE" = true ]; then
    echo "Visualize: enabled"
else
    echo "Visualize: disabled"
fi
echo "========================================================"
echo ""

# Track results
TOTAL_RUNS=$((${#MODELS[@]} * ${#DATASETS[@]}))
CURRENT_RUN=0
FAILED_RUNS=0

# Results summary
declare -A RESULTS

for MODEL_NAME in "${MODELS[@]}"; do
    MODEL_SHORT=$(echo $MODEL_NAME | sed 's/.*\///')
    
    for DATASET in "${DATASETS[@]}"; do
        CURRENT_RUN=$((CURRENT_RUN + 1))
        
        echo ""
        echo "========================================================"
        echo "[$CURRENT_RUN/$TOTAL_RUNS] $MODEL_SHORT on $DATASET"
        echo "========================================================"
        
        # Determine split based on dataset
        if [ "$DATASET" = "wikitext2" ]; then
            SPLIT=$WIKITEXT_SPLIT
        else
            SPLIT=$C4_SPLIT
        fi
        
        # Build run name for WandB
        WANDB_RUN_NAME="fp_${MODEL_SHORT}_${DATASET}"
        
        # Build command
        CMD="python ADC/llama/runs/measure_perplexity.py \
            --model_name \"$MODEL_NAME\" \
            --torch_dtype $TORCH_DTYPE \
            --dataset $DATASET \
            --dataset_split $SPLIT \
            --max_length $MAX_LENGTH \
            --seed $SEED \
            --wandb_project \"$WANDB_PROJECT\" \
            --wandb_run_name \"$WANDB_RUN_NAME\" \
            --tags \"full_precision\" \"$MODEL_SHORT\" \"$DATASET\""
        
        # Add stride if specified
        if [ -n "$STRIDE" ]; then
            CMD="$CMD --stride $STRIDE"
        fi
        
        # Add max_samples for C4 (only if limit is set)
        if [ "$DATASET" = "c4" ] && [ -n "$C4_MAX_SAMPLES" ]; then
            CMD="$CMD --max_samples $C4_MAX_SAMPLES"
        fi
        
        # Add visualization options
        if [ "$VISUALIZE" = true ]; then
            VIZ_PATH="viz_${MODEL_SHORT}_${DATASET}"
            CMD="$CMD --visualize --viz_save_path \"$VIZ_PATH\" --viz_num_samples $VIZ_NUM_SAMPLES --viz_seq_length $VIZ_SEQ_LENGTH"
        fi
        
        # Run
        echo "Running: $MODEL_SHORT on $DATASET ($SPLIT)..."
        echo ""
        eval $CMD
        
        EXIT_CODE=$?
        
        if [ $EXIT_CODE -eq 0 ]; then
            echo ""
            echo "[OK] $MODEL_SHORT on $DATASET completed successfully"
            RESULTS["${MODEL_SHORT}_${DATASET}"]="SUCCESS"
        else
            echo ""
            echo "[FAILED] $MODEL_SHORT on $DATASET failed with exit code $EXIT_CODE"
            RESULTS["${MODEL_SHORT}_${DATASET}"]="FAILED"
            FAILED_RUNS=$((FAILED_RUNS + 1))
        fi
        
        echo ""
    done
done

# ============================================================
# SUMMARY
# ============================================================
echo ""
echo "========================================================"
echo "BENCHMARK COMPLETE"
echo "========================================================"
echo ""
echo "Results Summary:"
echo "----------------"

for MODEL_NAME in "${MODELS[@]}"; do
    MODEL_SHORT=$(echo $MODEL_NAME | sed 's/.*\///')
    echo ""
    echo "$MODEL_SHORT:"
    for DATASET in "${DATASETS[@]}"; do
        STATUS="${RESULTS[${MODEL_SHORT}_${DATASET}]}"
        if [ "$STATUS" = "SUCCESS" ]; then
            echo "  $DATASET: OK"
        else
            echo "  $DATASET: FAILED"
        fi
    done
done

echo ""
echo "========================================================"
echo "Total runs: $TOTAL_RUNS"
echo "Successful: $((TOTAL_RUNS - FAILED_RUNS))"
echo "Failed:     $FAILED_RUNS"
echo "========================================================"
echo ""
echo "View all results at:"
echo "  https://wandb.ai/your-username/$WANDB_PROJECT"
echo ""
echo "Compare models in WandB:"
echo "  1. Go to the project page"
echo "  2. Select all runs"
echo "  3. Use the comparison view to see perplexity across models/datasets"
echo "========================================================"

# Exit with error if any run failed
if [ $FAILED_RUNS -gt 0 ]; then
    exit 1
fi
