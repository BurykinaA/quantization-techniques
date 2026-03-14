#!/bin/bash
# FlatQuant + ADC PTQ Debug Script for LLaMA
#
# Runs extra diagnostics on top of the normal FlatQuant + ADC PTQ pipeline:
#   1. FP16 baseline perplexity (before any changes)
#   2. FlatQuant preprocessing (learnable transforms, layer-by-layer MSE training)
#   3. Quantization + Tiling WITHOUT ADC (isolates quantization error)
#   4. Full pipeline WITH ADC
#
# Reference: Sun et al., "FlatQuant: Flatness Matters for LLM Quantization", ICML 2025
# Official: https://github.com/ruikangliu/FlatQuant

# ============================================================
# MODEL CONFIGURATION
# ============================================================
MODEL_NAME="meta-llama/Llama-3.2-1B"
OUTPUT_DIR="./ADC/llama/checkpoints/outputs_llama_flat_quant_adc_ptq"

# ============================================================
# FlatQuant Configuration
# ============================================================
FQ_W_BITS=8
FQ_A_BITS=8
FQ_NSAMPLES=64
FQ_CALI_BSZ=16
FQ_EPOCHS=15
FQ_LR=0.005
FQ_DIAG_ALPHA=0.5
FQ_ADD_DIAG=true
FQ_LWC=true
FQ_LAC=true
FQ_SAVE_TRANSFORMS=true
FQ_RELOAD_PATH=""

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
WANDB_RUN_NAME="debug_fq_${MODEL_SHORT_NAME}_w${FQ_W_BITS}a${FQ_A_BITS}_e${FQ_EPOCHS}_bx${BX}_bw${BW}_ba${BA}_k${K}_${CALIBRATION_METHOD}"
SEED=42

echo "========================================"
echo "LLaMA FlatQuant + ADC PTQ — DEBUG MODE"
echo "========================================"
echo "Model:        $MODEL_NAME"
echo "FlatQuant:    W${FQ_W_BITS}A${FQ_A_BITS}  epochs=${FQ_EPOCHS}  lr=${FQ_LR}  diag=${FQ_ADD_DIAG}  lwc=${FQ_LWC}  lac=${FQ_LAC}"
echo "ADC Config:   BX=$BX  BW=$BW  BA=$BA  K=$K  MVM=$MVM_LIMIT"
echo ""
echo "Extra diagnostics enabled:"
echo "  [1] FP16 baseline perplexity          (--check_baseline)"
echo "  [2] FlatQuant preprocessing           (learnable transforms, MSE training)"
echo "  [3] Quant+Tiling WITHOUT ADC          (--run_no_adc_eval)"
echo "  [4] Full FlatQuant+ADC pipeline       (always)"
echo "  [E3] Calib vs inference mismatch      (--run_e3_check)"
echo "========================================"
echo ""

CMD="python ADC/llama/runs/llama_smooth_quant_adc_ptq.py \
    --model_name \"$MODEL_NAME\" \
    --output_dir \"$OUTPUT_DIR\" \
    --preprocess_method flat_quant \
    --fq_w_bits $FQ_W_BITS \
    --fq_a_bits $FQ_A_BITS \
    --fq_nsamples $FQ_NSAMPLES \
    --fq_cali_bsz $FQ_CALI_BSZ \
    --fq_epochs $FQ_EPOCHS \
    --fq_lr $FQ_LR \
    --fq_diag_alpha $FQ_DIAG_ALPHA \
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
    --run_no_adc_eval \
    --run_e3_check"

if [ -n "$STRIDE" ]; then
    CMD="$CMD --stride $STRIDE"
fi

if [ "$ASHIFT" = true ]; then
    CMD="$CMD --ashift"
fi

if [ "$FQ_ADD_DIAG" = true ]; then
    CMD="$CMD --fq_add_diag"
else
    CMD="$CMD --fq_no_diag"
fi

if [ "$FQ_LWC" = true ]; then
    CMD="$CMD --fq_lwc"
else
    CMD="$CMD --fq_no_lwc"
fi

if [ "$FQ_LAC" = true ]; then
    CMD="$CMD --fq_lac"
else
    CMD="$CMD --fq_no_lac"
fi

if [ "$FQ_SAVE_TRANSFORMS" = true ]; then
    CMD="$CMD --fq_save_transforms"
fi

if [ -n "$FQ_RELOAD_PATH" ]; then
    CMD="$CMD --fq_reload_path \"$FQ_RELOAD_PATH\""
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
    echo "  e3/mean_mse, e3/max_rel_error   — Calib vs inference per-layer mismatch"
else
    echo "Debug FlatQuant + ADC PTQ Failed (exit code $EXIT_CODE)"
    echo "========================================"
fi
echo "========================================"
