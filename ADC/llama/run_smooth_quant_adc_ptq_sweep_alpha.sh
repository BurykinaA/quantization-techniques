#!/bin/bash
# SmoothQuant + ADC PTQ: Alpha Sweep
#
# Runs the full SmoothQuant + ADC pipeline for alpha = 0.0, 0.1, 0.2, ..., 1.0
# All results are logged to the same WandB project for easy comparison.
# Visualizations are disabled to speed up the sweep.

# ============================================================
# MODEL CONFIGURATION
# ============================================================
MODEL_NAME="meta-llama/Llama-3.2-1B"
OUTPUT_DIR_BASE="./ADC/llama/checkpoints/sweep_alpha"

# ============================================================
# ADC Hardware Configuration
# ============================================================
BX=8
BW=8
BA=8
K=4
ASHIFT=false
MVM_LIMIT=256

# ============================================================
# Calibration Settings
# ============================================================
CALIBRATION_METHOD="percentile"
NUM_CALIBRATION_BATCHES=128
CALIBRATION_BATCH_SIZE=4
CALIBRATION_MAX_LENGTH=512
SMOOTH_QUANT_BATCHES=64

# ============================================================
# Dataset & Evaluation Settings
# ============================================================
CALIBRATION_DATASET="wikitext2"
EVAL_DATASETS="wikitext2"
MAX_LENGTH=2048
STRIDE=""
EVAL_SPLIT="test"
MAX_EVAL_SAMPLES=1000

# ============================================================
# Other Settings
# ============================================================
TORCH_DTYPE="float16"
WANDB_PROJECT="llama-sq-adc-sweep-alpha"
MODEL_SHORT_NAME=$(echo $MODEL_NAME | sed 's/.*\///')
SEED=42

# ============================================================
# Alpha values to sweep
# ============================================================
ALPHAS="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"

echo "========================================================"
echo "  SmoothQuant + ADC PTQ — Alpha Sweep"
echo "========================================================"
echo "Model:   $MODEL_NAME"
echo "Alphas:  $ALPHAS"
echo "ADC:     BX=$BX  BW=$BW  BA=$BA  K=$K  MVM=$MVM_LIMIT"
echo "Calib:   $CALIBRATION_METHOD ($NUM_CALIBRATION_BATCHES batches)"
echo "Eval:    $EVAL_DATASETS (sliding window, ctx=$MAX_LENGTH)"
echo "WandB:   $WANDB_PROJECT"
echo "========================================================"
echo ""

PASSED=0
FAILED=0
RESULTS=""

for ALPHA in $ALPHAS; do
    echo "========================================================"
    echo "  Running alpha=$ALPHA"
    echo "========================================================"

    OUTPUT_DIR="${OUTPUT_DIR_BASE}/alpha_${ALPHA}"
    WANDB_RUN_NAME="sweep_${MODEL_SHORT_NAME}_a${ALPHA}_bx${BX}_bw${BW}_ba${BA}_k${K}"

    CMD="python ADC/llama/runs/llama_smooth_quant_adc_ptq.py \
        --model_name \"$MODEL_NAME\" \
        --output_dir \"$OUTPUT_DIR\" \
        --alpha $ALPHA \
        --smooth_quant_batches $SMOOTH_QUANT_BATCHES \
        --bx $BX \
        --bw $BW \
        --ba $BA \
        --k $K \
        --mvm_limit $MVM_LIMIT \
        --calibration_method $CALIBRATION_METHOD \
        --calibration_dataset $CALIBRATION_DATASET \
        --eval_datasets $EVAL_DATASETS \
        --num_calibration_batches $NUM_CALIBRATION_BATCHES \
        --calibration_batch_size $CALIBRATION_BATCH_SIZE \
        --calibration_max_length $CALIBRATION_MAX_LENGTH \
        --max_length $MAX_LENGTH \
        --eval_split $EVAL_SPLIT \
        --max_eval_samples $MAX_EVAL_SAMPLES \
        --torch_dtype $TORCH_DTYPE \
        --seed $SEED \
        --wandb_project \"$WANDB_PROJECT\" \
        --wandb_run_name \"$WANDB_RUN_NAME\" \
        --disable_visualizations"

    if [ -n "$STRIDE" ]; then
        CMD="$CMD --stride $STRIDE"
    fi

    if [ "$ASHIFT" = true ]; then
        CMD="$CMD --ashift"
    fi

    eval $CMD
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        PASSED=$((PASSED + 1))
        # Extract perplexity from eval_metrics.txt
        METRICS_FILE="${OUTPUT_DIR}_$(date +%Y%m%d)/eval_metrics.txt"
        if [ -f "$METRICS_FILE" ]; then
            PPL=$(grep "perplexity" "$METRICS_FILE" | head -1 | awk '{print $2}')
            RESULTS="${RESULTS}\n  alpha=${ALPHA}  perplexity=${PPL}  [OK]"
        else
            RESULTS="${RESULTS}\n  alpha=${ALPHA}  [OK, metrics file not found]"
        fi
    else
        FAILED=$((FAILED + 1))
        RESULTS="${RESULTS}\n  alpha=${ALPHA}  [FAILED, exit code $EXIT_CODE]"
    fi

    echo ""
done

echo "========================================================"
echo "  Alpha Sweep Complete"
echo "========================================================"
echo "  Passed: $PASSED / $((PASSED + FAILED))"
echo "  Failed: $FAILED / $((PASSED + FAILED))"
echo ""
echo "Results:"
echo -e "$RESULTS"
echo ""
echo "Compare all runs in WandB:"
echo "  https://wandb.ai/your-username/$WANDB_PROJECT"
echo "========================================================"
