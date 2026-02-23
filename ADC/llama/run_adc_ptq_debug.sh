#!/bin/bash
# ADC PTQ Debug Script for LLaMA
#
# Runs extra diagnostics on top of the normal PTQ pipeline:
#   1. FP16 baseline perplexity (before any quantization)
#   2. Quantization + Tiling WITHOUT ADC (isolates quantization error)
#   3. Full pipeline WITH ADC
#
# Use this to diagnose whether perplexity issues come from
# quantization, tiling, or the ADC step.

# ============================================================
# MODEL CONFIGURATION
# ============================================================
MODEL_NAME="meta-llama/Llama-3.2-1B"
OUTPUT_DIR="./ADC/llama/checkpoints/outputs_llama_adc_ptq"

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

# ============================================================
# Dataset Settings
# ============================================================
CALIBRATION_DATASET="wikitext2"
EVAL_DATASETS="wikitext2"

# ============================================================
# Evaluation Settings
# ============================================================
MAX_LENGTH=2048
STRIDE=""
EVAL_SPLIT="test"
MAX_EVAL_SAMPLES=1000

# ============================================================
# Other Settings
# ============================================================
TORCH_DTYPE="float16"
WANDB_PROJECT="llama-adc-ptq"
MODEL_SHORT_NAME=$(echo $MODEL_NAME | sed 's/.*\///')
WANDB_RUN_NAME="debug_${MODEL_SHORT_NAME}_bx${BX}_bw${BW}_ba${BA}_k${K}_${CALIBRATION_METHOD}"
SEED=42

echo "========================================"
echo "LLaMA ADC PTQ — DEBUG MODE"
echo "========================================"
echo "Model:        $MODEL_NAME"
echo "Config:       BX=$BX  BW=$BW  BA=$BA  K=$K  MVM=$MVM_LIMIT"
echo ""
echo "Extra diagnostics enabled:"
echo "  [1] FP16 baseline perplexity    (--check_baseline)"
echo "  [2] Quant+Tiling WITHOUT ADC    (--run_no_adc_eval)"
echo "  [3] Full pipeline WITH ADC      (always)"
echo "========================================"
echo ""

CMD="python ADC/llama/runs/llama_adc_ptq.py \
    --model_name \"$MODEL_NAME\" \
    --output_dir \"$OUTPUT_DIR\" \
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
    --disable_visualizations \
    --check_baseline \
    --run_no_adc_eval"

if [ -n "$STRIDE" ]; then
    CMD="$CMD --stride $STRIDE"
fi

if [ "$ASHIFT" = true ]; then
    CMD="$CMD --ashift"
fi

echo "Running PTQ with diagnostics..."
echo ""
eval $CMD

EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Debug PTQ Complete!"
    echo "========================================"
    echo ""
    echo "Check WandB for metrics comparison:"
    echo "  baseline/perplexity       — FP16 (no quantization)"
    echo "  diagnostic/no_adc_perplexity — Quant + Tiling only"
    echo "  final perplexity          — Full ADC pipeline"
else
    echo "Debug PTQ Failed (exit code $EXIT_CODE)"
    echo "========================================"
fi
echo "========================================"
