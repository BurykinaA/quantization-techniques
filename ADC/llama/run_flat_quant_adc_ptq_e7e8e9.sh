#!/bin/bash
# FlatQuant + ADC PTQ — E7 / E8 / E9 output-aware penalty experiments
#
# Variants (first argument):
#   baseline  — no penalty (reproduces existing run_flat_quant_adc_ptq_example.sh)
#   e7        — clip penalty only  (lambda_clip=0.01)
#   e8        — dead-zone penalty only
#   e9        — combined clip + dead
#
# Intensity (second argument, applies to e8/e9 dead-zone penalty):
#   weak    lambda_dead=0.01  dead_threshold=1.0
#   mid     lambda_dead=0.1   dead_threshold=2.0   [default]
#   strong  lambda_dead=1.0   dead_threshold=2.0
#
# Usage:
#   bash run_flat_quant_adc_ptq_e7e8e9.sh [baseline|e7|e8|e9] [weak|mid|strong]
#   bash run_flat_quant_adc_ptq_e7e8e9.sh e8 weak
#   bash run_flat_quant_adc_ptq_e7e8e9.sh e8          # mid by default

EXPERIMENT="${1:-baseline}"
# Optional intensity suffix for e8/e9: weak | mid | strong (default: mid)
# Controls lambda_dead + dead_threshold; ignored for baseline/e7.
#   weak:   lambda_dead=0.01, dead_threshold=1.0
#   mid:    lambda_dead=0.1,  dead_threshold=2.0
#   strong: lambda_dead=1.0,  dead_threshold=2.0
INTENSITY="${2:-mid}"

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
FQ_NSAMPLES=128
FQ_CALI_BSZ=16
FQ_EPOCHS=30
FQ_LR=0.005
FQ_DIAG_ALPHA=0.5
FQ_ADD_DIAG=false
FQ_LWC=true
FQ_LAC=true
FQ_SAVE_TRANSFORMS=true
FQ_RELOAD_PATH=""          # set to a .pt path to skip FQ training

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
# ADC Visualization
# ============================================================
VISUALIZE_LAYERS="layers.0.self_attn.q_proj layers.0.mlp.down_proj layers.15.mlp.gate_proj"

# ============================================================
# Other
# ============================================================
TORCH_DTYPE="float32"
WANDB_PROJECT="llama-flat-quant-adc-ptq-blocks"
SEED=42

# ============================================================
# Experiment-specific penalty parameters
# ============================================================
FQ_LAMBDA_CLIP=0.0
FQ_LAMBDA_DEAD=0.0
FQ_CLIP_MARGIN=1.0
FQ_DEAD_THRESHOLD=1.0
# FQ_PENALTY_PROJECTIONS="o_proj down_proj"
FQ_PENALTY_PROJECTIONS="o_proj down_proj q_proj k_proj v_proj gate_proj up_proj"

# Resolve intensity → lambda_dead + dead_threshold (used by e8/e9)
case "$INTENSITY" in
    weak)
        _LAMBDA_DEAD_INTENSITY=0.01
        _DEAD_THRESHOLD_INTENSITY=1.0
        ;;
    mid)
        _LAMBDA_DEAD_INTENSITY=0.1
        _DEAD_THRESHOLD_INTENSITY=2.0
        ;;
    strong)
        _LAMBDA_DEAD_INTENSITY=1.0
        _DEAD_THRESHOLD_INTENSITY=2.0
        ;;
    *)
        echo "Unknown intensity: '$INTENSITY'"
        echo "Available: weak | mid | strong"
        exit 1
        ;;
esac

case "$EXPERIMENT" in
    baseline)
        FQ_LAMBDA_CLIP=0.0
        FQ_LAMBDA_DEAD=0.0
        ;;
    e7)
        # clip penalty only — intensity not applicable (clip_rate=0 in baseline)
        FQ_LAMBDA_CLIP=0.01
        FQ_LAMBDA_DEAD=0.0
        ;;
    e8)
        FQ_LAMBDA_CLIP=0.0
        FQ_LAMBDA_DEAD=$_LAMBDA_DEAD_INTENSITY
        FQ_DEAD_THRESHOLD=$_DEAD_THRESHOLD_INTENSITY
        ;;
    e9)
        FQ_LAMBDA_CLIP=0.01
        FQ_LAMBDA_DEAD=$_LAMBDA_DEAD_INTENSITY
        FQ_DEAD_THRESHOLD=$_DEAD_THRESHOLD_INTENSITY
        ;;
    *)
        echo "Unknown experiment: '$EXPERIMENT'"
        echo "Available: baseline | e7 | e8 | e9"
        exit 1
        ;;
esac

# Build run name (mirrors Python default_run_name logic)
MODEL_SHORT_NAME=$(echo $MODEL_NAME | sed 's/.*\///')
if [[ "$EXPERIMENT" == "e8" || "$EXPERIMENT" == "e9" ]]; then
    WANDB_RUN_NAME="${EXPERIMENT}_fq_ptq_${MODEL_SHORT_NAME}_w${FQ_W_BITS}a${FQ_A_BITS}_e${FQ_EPOCHS}_bx${BX}_bw${BW}_ba${BA}_k${K}_${CALIBRATION_METHOD}_lc${FQ_LAMBDA_CLIP}_ld${FQ_LAMBDA_DEAD}_tau${FQ_DEAD_THRESHOLD}_${INTENSITY}"
else
    WANDB_RUN_NAME="${EXPERIMENT}_fq_ptq_${MODEL_SHORT_NAME}_w${FQ_W_BITS}a${FQ_A_BITS}_e${FQ_EPOCHS}_bx${BX}_bw${BW}_ba${BA}_k${K}_${CALIBRATION_METHOD}_lc${FQ_LAMBDA_CLIP}_ld${FQ_LAMBDA_DEAD}"
fi

echo "========================================"
if [[ "$EXPERIMENT" == "e8" || "$EXPERIMENT" == "e9" ]]; then
    EXP_LABEL="${EXPERIMENT^^}/${INTENSITY}"
else
    EXP_LABEL="${EXPERIMENT^^}"
fi
echo "LLaMA FlatQuant + ADC PTQ  [${EXP_LABEL}]"
echo "========================================"
echo "Model:             $MODEL_NAME"
echo "Output:            $OUTPUT_DIR"
echo ""
echo "FlatQuant Configuration:"
echo "  Internal quant:  W${FQ_W_BITS}A${FQ_A_BITS}"
echo "  Training:        epochs=${FQ_EPOCHS}  lr=${FQ_LR}  samples=${FQ_NSAMPLES}  bsz=${FQ_CALI_BSZ}"
echo "  Features:        diag=${FQ_ADD_DIAG}  lwc=${FQ_LWC}  lac=${FQ_LAC}"
if [ -n "$FQ_RELOAD_PATH" ]; then
    echo "  Reload:          $FQ_RELOAD_PATH"
fi
echo ""
echo "ADC Configuration:"
echo "  BX=$BX  BW=$BW  BA=$BA  K=$K  MVM=$MVM_LIMIT"
echo ""
echo "Penalty Configuration:"
echo "  lambda_clip=${FQ_LAMBDA_CLIP}  lambda_dead=${FQ_LAMBDA_DEAD}"
echo "  clip_margin=${FQ_CLIP_MARGIN}  dead_threshold=${FQ_DEAD_THRESHOLD}"
echo "  projections: ${FQ_PENALTY_PROJECTIONS}"
echo ""
echo "Calibration: $CALIBRATION_METHOD ($NUM_CALIBRATION_BATCHES batches)"
echo "Evaluation:  $EVAL_DATASETS (sliding window, ctx=$MAX_LENGTH)"
echo "WandB run:   $WANDB_RUN_NAME"
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
    --fq_lambda_clip $FQ_LAMBDA_CLIP \
    --fq_lambda_dead $FQ_LAMBDA_DEAD \
    --fq_clip_margin $FQ_CLIP_MARGIN \
    --fq_dead_threshold $FQ_DEAD_THRESHOLD \
    --fq_penalty_projections $FQ_PENALTY_PROJECTIONS \
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
    --visualize_layers $VISUALIZE_LAYERS"

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

echo "Running FlatQuant + ADC PTQ [${EXP_LABEL}]..."
echo ""
eval $CMD

EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "FlatQuant + ADC PTQ [${EXP_LABEL}] Complete!"
    echo "========================================"
    echo ""
    echo "Results:  $OUTPUT_DIR"
    echo "Plots:"
    echo "  ADC Before:     $OUTPUT_DIR/viz_before/"
    echo "  ADC After:      $OUTPUT_DIR/viz_after/"
    if [ "$FQ_SAVE_TRANSFORMS" = true ]; then
        echo "  Transforms:     $OUTPUT_DIR/flat_quant_transforms.pt"
    fi
    echo "WandB:    https://wandb.ai/your-username/$WANDB_PROJECT"
    echo ""
    echo "Key WandB metrics to check:"
    echo "  adc_diag/mean_dead_rate      — should decrease for e8/e9"
    echo "  adc_diag/mean_clip_rate      — should not spike for e8/e9"
    echo "  adc_diag/mean_bin_usage      — should increase"
    echo "  adc_diag/mean_reconstruction_mse"
    echo "  perplexity/wikitext2"
else
    echo "FlatQuant + ADC PTQ [${EXP_LABEL}] Failed (exit code $EXIT_CODE)"
    echo "========================================"
    echo ""
    echo "Common issues:"
    echo "  - OOM: Reduce FQ_NSAMPLES, FQ_CALI_BSZ, MAX_LENGTH, or CALIBRATION_BATCH_SIZE"
    echo "  - Auth: Run 'huggingface-cli login' for gated models"
    echo "  - CUDA: Check GPU availability with 'nvidia-smi'"
fi
echo "========================================"
