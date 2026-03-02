#!/bin/bash
# FlatQuant + ADC PTQ Debug Script for LLaMA
#
# Runs extra diagnostics on top of the normal FlatQuant + ADC PTQ pipeline:
#   1. FP16 baseline perplexity (before any changes)
#   2. FlatQuant preprocessing
#   3. Quantization + Tiling WITHOUT ADC (isolates quantization error)
#   4. Full pipeline WITH ADC

# ============================================================
# MODEL CONFIGURATION
# ============================================================
MODEL_NAME="meta-llama/Llama-3.2-1B"
OUTPUT_DIR="./ADC/llama/checkpoints/outputs_llama_flat_quant_adc_ptq"

# ============================================================
# FlatQuant Configuration
# ============================================================
FLAT_QUANT_BATCHES=64
FLAT_QUANT_BETA=0.5
FLAT_QUANT_FLATTEN_STRENGTH=0.25
FLAT_QUANT_SAVE_TRANSFORMS=true
FLAT_QUANT_RELOAD_PATH=""

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
WANDB_PROJECT="llama-flat-quant-adc-ptq"
MODEL_SHORT_NAME=$(echo $MODEL_NAME | sed 's/.*\///')
WANDB_RUN_NAME="debug_fq_${MODEL_SHORT_NAME}_b${FLAT_QUANT_BETA}_fs${FLAT_QUANT_FLATTEN_STRENGTH}_bx${BX}_bw${BW}_ba${BA}_k${K}_${CALIBRATION_METHOD}"
SEED=42

echo "========================================"
echo "LLaMA FlatQuant + ADC PTQ — DEBUG MODE"
echo "========================================"
echo "Model:        $MODEL_NAME"
echo "FlatQuant:    Batches=$FLAT_QUANT_BATCHES  Beta=$FLAT_QUANT_BETA  Flatten=$FLAT_QUANT_FLATTEN_STRENGTH"
echo "ADC Config:   BX=$BX  BW=$BW  BA=$BA  K=$K  MVM=$MVM_LIMIT"
echo ""
echo "Extra diagnostics enabled:"
echo "  [1] FP16 baseline perplexity          (--check_baseline)"
echo "  [2] FlatQuant preprocessing            (--preprocess_method flat_quant)"
echo "  [3] Quant+Tiling WITHOUT ADC          (--run_no_adc_eval)"
echo "  [4] Full FlatQuant+ADC pipeline       (always)"
echo "========================================"
echo ""

CMD="python ADC/llama/runs/llama_smooth_quant_adc_ptq.py \
    --model_name \"$MODEL_NAME\" \
    --output_dir \"$OUTPUT_DIR\" \
    --preprocess_method flat_quant \
    --flat_quant_batches $FLAT_QUANT_BATCHES \
    --flat_quant_beta $FLAT_QUANT_BETA \
    --flat_quant_flatten_strength $FLAT_QUANT_FLATTEN_STRENGTH \
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

if [ "$FLAT_QUANT_SAVE_TRANSFORMS" = true ]; then
    CMD="$CMD --flat_quant_save_transforms"
fi

if [ -n "$FLAT_QUANT_RELOAD_PATH" ]; then
    CMD="$CMD --flat_quant_reload_path \"$FLAT_QUANT_RELOAD_PATH\""
fi

echo "Running FlatQuant + ADC PTQ with diagnostics..."
echo ""
eval $CMD

EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Debug FlatQuant + ADC PTQ Complete!"
    echo "========================================"
    echo ""
    echo "Check WandB for metrics comparison:"
    echo "  baseline/perplexity             — FP16 (no quantization, no FlatQuant)"
    echo "  diagnostic/no_adc_perplexity    — FlatQuant + Quant + Tiling (no ADC)"
    echo "  final perplexity                — Full FlatQuant + ADC pipeline"
else
    echo "Debug FlatQuant + ADC PTQ Failed (exit code $EXIT_CODE)"
    echo "========================================"
fi
echo "========================================"
