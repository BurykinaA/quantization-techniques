#!/bin/bash
# FlatQuant + ADC PTQ — Stage-by-Stage Debug Script
#
# Runs perplexity at every key pipeline stage to pinpoint where quality degrades:
#
#   [A] FP baseline                           (--check_baseline)
#   [B] After FlatQuant reparameterize        (--stage_eval, stage B)
#       FP model with trained transforms baked in, no ADC
#       → if bad: FlatQuant training / reparameterization is broken
#   [C] After replace_linear_with_adc         (--stage_eval, stage C)
#       bypass_all=True → FP forward through ADC layer structure
#       → should match B; divergence means ADC replacement breaks something
#   [D] After ADC calibration                 (always, main eval)
#       → final quantized result
#   [+] Quant+Tiling WITHOUT ADC floor        (--run_no_adc_eval)
#       → isolates ADC non-linearity from tiling/quant error
#
# Usage:
#   bash ADC/llama/run_flat_quant_adc_ptq_debug.sh
#   bash ADC/llama/run_flat_quant_adc_ptq_debug.sh --reload /path/to/transforms.pt
#
# Pass --reload <path> as first argument to skip FlatQuant training and reuse
# a previously saved flat_quant_transforms.pt (much faster for ADC debugging).

FQ_RELOAD_PATH="${2}"  # optional: pass transforms path as 2nd arg
if [ "$1" = "--reload" ]; then
    FQ_RELOAD_PATH="$2"
fi

# ============================================================
# MODEL CONFIGURATION
# ============================================================
MODEL_NAME="meta-llama/Llama-3.2-1B"
OUTPUT_DIR="./ADC/llama/checkpoints/outputs_llama_flat_quant_adc_ptq_debug"

# ============================================================
# FlatQuant Configuration
# (match production settings from e7e8e9 script)
# ============================================================
FQ_W_BITS=8
FQ_A_BITS=8
FQ_NSAMPLES=128
FQ_CALI_BSZ=16
FQ_EPOCHS=30
FQ_LR=0.005
FQ_DIAG_ALPHA=0.5
FQ_ADD_DIAG=false     # --fq_no_diag (best setting found)
FQ_LWC=true
FQ_LAC=true
FQ_SAVE_TRANSFORMS=true

# ============================================================
# ADC Hardware Configuration
# ============================================================
BX=8
BW=8
BA=8
K=16
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
# Dataset / Evaluation Settings
# ============================================================
CALIBRATION_DATASET="wikitext2"
EVAL_DATASETS="wikitext2"
MAX_LENGTH=2048
STRIDE=""
EVAL_SPLIT="test"
MAX_EVAL_SAMPLES=1000

# ============================================================
# Stage eval: max windows per stage (smaller = faster)
# 50 windows ≈ 50 * 1024 = 51k tokens, ~30 sec per stage
# ============================================================
STAGE_EVAL_MAX_WINDOWS=50

# ============================================================
# Layer ablation: max windows per layer (16 layers × windows)
# 20 windows × 16 layers ≈ 10-15 min extra
# ============================================================
LAYER_ABLATION_MAX_WINDOWS=20

# ============================================================
# Other
# ============================================================
TORCH_DTYPE="float32"
WANDB_PROJECT="llama-flat-quant-adc-ptq-debug"
MODEL_SHORT_NAME=$(echo $MODEL_NAME | sed 's/.*\///')
WANDB_RUN_NAME="debug_stage_${MODEL_SHORT_NAME}_w${FQ_W_BITS}a${FQ_A_BITS}_e${FQ_EPOCHS}_bx${BX}_bw${BW}_ba${BA}_k${K}_${CALIBRATION_METHOD}"
SEED=42

echo "========================================"
echo "LLaMA FlatQuant + ADC PTQ — STAGE DEBUG"
echo "========================================"
echo "Model:        $MODEL_NAME"
echo "FlatQuant:    W${FQ_W_BITS}A${FQ_A_BITS}  epochs=${FQ_EPOCHS}  lr=${FQ_LR}"
echo "              diag=${FQ_ADD_DIAG}  lwc=${FQ_LWC}  lac=${FQ_LAC}"
echo "ADC Config:   BX=$BX  BW=$BW  BA=$BA  K=$K  MVM=$MVM_LIMIT"
echo ""
echo "Stage evals enabled (max $STAGE_EVAL_MAX_WINDOWS windows each):"
echo "  [A] FP baseline perplexity            (--check_baseline)"
echo "  [B] Post-FQ-reparameterize (no ADC)   (--stage_eval)"
echo "  [C] Post-ADC-replace, bypass=FP       (--stage_eval)"
echo "  [D] Post-ADC-calibration              (main eval)"
echo "  [+] Tiling+Quant WITHOUT ADC floor    (--run_no_adc_eval)"
echo "  [L] Layer-by-layer ablation           (--layer_ablation, $LAYER_ABLATION_MAX_WINDOWS windows/layer)"
echo ""
echo "WandB metrics to compare:"
echo "  baseline/perplexity"
echo "  stage_eval/B_post_fq_reparameterize/perplexity"
echo "  stage_eval/C_post_adc_replace_bypass/perplexity"
echo "  diagnostic/no_adc_perplexity"
echo "  eval/wikitext2/perplexity"
echo "========================================"
echo ""
if [ -n "$FQ_RELOAD_PATH" ]; then
    echo "Reloading transforms from: $FQ_RELOAD_PATH"
    echo "(FlatQuant training will be skipped)"
    echo ""
fi

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
    --stage_eval \
    --stage_eval_max_windows $STAGE_EVAL_MAX_WINDOWS \
    --run_no_adc_eval \
    --layer_ablation \
    --layer_ablation_max_windows $LAYER_ABLATION_MAX_WINDOWS \
    --proj_ablation"

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

echo "Running stage-by-stage debug..."
echo ""
eval $CMD

EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Stage Debug Complete!"
    echo "========================================"
    echo ""
    echo "Check WandB run: $WANDB_RUN_NAME"
    echo ""
    echo "Stage perplexity breakdown:"
    echo "  [A] baseline/perplexity                              — FP (target)"
    echo "  [B] stage_eval/B_post_fq_reparameterize/perplexity  — after FlatQuant"
    echo "  [C] stage_eval/C_post_adc_replace_bypass/perplexity — ADC structure (FP)"
    echo "  [+] diagnostic/no_adc_perplexity                    — tiling+quant, no ADC"
    echo "  [D] eval/wikitext2/perplexity                        — full pipeline"
    echo "  [L] layer_ablation/layer_XX/perplexity               — per-layer bottleneck"
    echo ""
    echo "Diagnosis guide:"
    echo "  A≈B          → FlatQuant OK"
    echo "  B≈C          → ADC layer replacement transparent"
    echo "  B>>C (B bad) → FlatQuant training or reparameterize broken"
    echo "  C>>D (C bad) → ADC structure introduces overhead even in bypass"
    echo "  D>>+         → ADC quantization (floor) is the main source of error"
    echo "  +≈D          → tiling or quantization range wrong"
    if [ -n "$OUTPUT_DIR" ]; then
        echo ""
        echo "Transforms saved to: $OUTPUT_DIR/flat_quant_transforms.pt"
        echo "  Rerun faster:  bash run_flat_quant_adc_ptq_debug.sh --reload $OUTPUT_DIR/flat_quant_transforms.pt"
    fi
else
    echo "Stage Debug Failed (exit code $EXIT_CODE)"
    echo "========================================"
    echo ""
    echo "Common issues:"
    echo "  - OOM: Reduce FQ_NSAMPLES, FQ_CALI_BSZ, or STAGE_EVAL_MAX_WINDOWS"
    echo "  - Auth: Run 'huggingface-cli login' for gated models"
fi
echo "========================================"
