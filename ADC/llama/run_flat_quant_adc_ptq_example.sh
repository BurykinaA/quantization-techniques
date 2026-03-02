#!/bin/bash
# FlatQuant + ADC Post-Training Quantization (PTQ) for LLaMA models
# Pipeline: FlatQuant -> ADC Convert -> Calibrate -> Evaluate -> Visualize
#
# Supports: meta-llama/Llama-3.1-8B, meta-llama/Llama-3.2-3B, meta-llama/Llama-3.2-1B

# ============================================================
# MODEL CONFIGURATION
# ============================================================
MODEL_NAME="meta-llama/Llama-3.2-1B"

# Output directory (timestamp appended automatically)
OUTPUT_DIR="./ADC/llama/checkpoints/outputs_llama_flat_quant_adc_ptq"

# ============================================================
# FlatQuant Configuration (W4A4-first)
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
# Evaluation Settings (Sliding Window)
# ============================================================
MAX_LENGTH=2048
STRIDE=""
EVAL_SPLIT="test"
MAX_EVAL_SAMPLES=1000

# ============================================================
# ADC Visualization Settings
# ============================================================
VISUALIZE_LAYERS="layers.0.self_attn.q_proj layers.0.mlp.down_proj layers.15.mlp.gate_proj"

# ============================================================
# Other Settings
# ============================================================
TORCH_DTYPE="float16"
WANDB_PROJECT="llama-flat-quant-adc-ptq"
MODEL_SHORT_NAME=$(echo $MODEL_NAME | sed 's/.*\///')
WANDB_RUN_NAME="fq_ptq_${MODEL_SHORT_NAME}_b${FLAT_QUANT_BETA}_fs${FLAT_QUANT_FLATTEN_STRENGTH}_bx${BX}_bw${BW}_ba${BA}_k${K}_${CALIBRATION_METHOD}"
SEED=42

echo "========================================"
echo "LLaMA FlatQuant + ADC PTQ"
echo "========================================"
echo "Model:             $MODEL_NAME"
echo "Output:            $OUTPUT_DIR"
echo ""
echo "FlatQuant Configuration:"
echo "  Batches=$FLAT_QUANT_BATCHES  Beta=$FLAT_QUANT_BETA  Flatten=$FLAT_QUANT_FLATTEN_STRENGTH"
if [ "$FLAT_QUANT_SAVE_TRANSFORMS" = true ]; then
    echo "  Save transforms: enabled"
else
    echo "  Save transforms: disabled"
fi
if [ -n "$FLAT_QUANT_RELOAD_PATH" ]; then
    echo "  Reload transforms: $FLAT_QUANT_RELOAD_PATH"
fi
echo ""
echo "ADC Configuration:"
echo "  BX=$BX  BW=$BW  BA=$BA  K=$K  MVM=$MVM_LIMIT"
echo ""
echo "Quantization Strategy:"
if [ "$ASHIFT" = true ]; then
    echo "  Asymmetric (unsigned) + A-shift for SiLU outputs"
else
    echo "  Symmetric (signed) for all activations"
fi
echo ""
echo "Calibration: $CALIBRATION_METHOD ($NUM_CALIBRATION_BATCHES batches)"
echo "Evaluation:  $EVAL_DATASETS (sliding window, ctx=$MAX_LENGTH)"
echo "ADC Viz:     $VISUALIZE_LAYERS"
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
    --visualize_layers $VISUALIZE_LAYERS"

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

echo "Running FlatQuant + ADC PTQ..."
echo ""
eval $CMD

EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "FlatQuant + ADC PTQ Complete!"
    echo "========================================"
    echo ""
    echo "Results:  $OUTPUT_DIR"
    echo "Plots:"
    echo "  ADC Before:     $OUTPUT_DIR/viz_before/"
    echo "  ADC After:      $OUTPUT_DIR/viz_after/"
    if [ "$FLAT_QUANT_SAVE_TRANSFORMS" = true ]; then
        echo "  Transforms:     $OUTPUT_DIR/flat_quant_transforms.pt"
    fi
    echo "WandB:    https://wandb.ai/your-username/$WANDB_PROJECT"
else
    echo "FlatQuant + ADC PTQ Failed (exit code $EXIT_CODE)"
    echo "========================================"
    echo ""
    echo "Common issues:"
    echo "  - OOM: Reduce MAX_LENGTH or CALIBRATION_BATCH_SIZE"
    echo "  - Auth: Run 'huggingface-cli login' for gated models"
    echo "  - CUDA: Check GPU availability with 'nvidia-smi'"
fi
echo "========================================"
