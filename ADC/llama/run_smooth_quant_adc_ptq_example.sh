#!/bin/bash
# SmoothQuant + ADC Post-Training Quantization (PTQ) for LLaMA models
# Pipeline: SmoothQuant → ADC Convert → Calibrate → Evaluate → Visualize
#
# Supports: meta-llama/Llama-3.1-8B, meta-llama/Llama-3.2-3B, meta-llama/Llama-3.2-1B

# ============================================================
# MODEL CONFIGURATION
# ============================================================
MODEL_NAME="meta-llama/Llama-3.2-1B"

# Output directory (timestamp appended automatically)
OUTPUT_DIR="./ADC/llama/checkpoints/outputs_llama_smooth_quant_adc_ptq"

# ============================================================
# SmoothQuant Configuration
# ============================================================
ALPHA=0.5                 # Migration strength (0=all on weights, 1=all on activations)
SMOOTH_QUANT_BATCHES=64   # Calibration batches for activation statistics

# Layer patterns for SmoothQuant 3D visualization
SMOOTH_QUANT_VIZ_LAYERS="layers.0.self_attn.q_proj layers.0.mlp.gate_proj"

# ============================================================
# ADC Hardware Configuration
# ============================================================
BX=8              # Activation bits
BW=8              # Weight bits
BA=8              # ADC bits
K=4               # Hardware design parameter (sub-ADCs per column)
ASHIFT=false      # A-shift: false=symmetric, true=asymmetric+A-shift
MVM_LIMIT=256     # Max crossbar size (tiling splits layers > this)

# ============================================================
# Calibration Settings
# ============================================================
CALIBRATION_METHOD="percentile"  # Options: minmax, percentile, mse
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
# Layer patterns to generate 3x4 diagnostic plots for.
VISUALIZE_LAYERS="layers.0.self_attn.q_proj layers.0.mlp.down_proj layers.15.mlp.gate_proj"

# ============================================================
# Other Settings
# ============================================================
TORCH_DTYPE="float16"
WANDB_PROJECT="llama-smooth-quant-adc-ptq"
MODEL_SHORT_NAME=$(echo $MODEL_NAME | sed 's/.*\///')
WANDB_RUN_NAME="sq_ptq_${MODEL_SHORT_NAME}_a${ALPHA}_bx${BX}_bw${BW}_ba${BA}_k${K}_${CALIBRATION_METHOD}"
SEED=42

# ============================================================
# Print configuration
# ============================================================
echo "========================================"
echo "LLaMA SmoothQuant + ADC PTQ"
echo "========================================"
echo "Model:             $MODEL_NAME"
echo "Output:            $OUTPUT_DIR"
echo ""
echo "SmoothQuant Configuration:"
echo "  Alpha=$ALPHA  Batches=$SMOOTH_QUANT_BATCHES"
echo "  3D Viz layers: $SMOOTH_QUANT_VIZ_LAYERS"
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

# ============================================================
# Build and run command
# ============================================================
CMD="python ADC/llama/runs/llama_smooth_quant_adc_ptq.py \
    --model_name \"$MODEL_NAME\" \
    --output_dir \"$OUTPUT_DIR\" \
    --alpha $ALPHA \
    --smooth_quant_batches $SMOOTH_QUANT_BATCHES \
    --smooth_quant_layers $SMOOTH_QUANT_VIZ_LAYERS \
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

echo "Running SmoothQuant + ADC PTQ..."
echo ""
eval $CMD

EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "SmoothQuant + ADC PTQ Complete!"
    echo "========================================"
    echo ""
    echo "Results:  $OUTPUT_DIR"
    echo "Plots:"
    echo "  SmoothQuant 3D: $OUTPUT_DIR/viz_smooth_quant/"
    echo "  ADC Before:     $OUTPUT_DIR/viz_before/"
    echo "  ADC After:      $OUTPUT_DIR/viz_after/"
    echo "WandB:    https://wandb.ai/your-username/$WANDB_PROJECT"
else
    echo "SmoothQuant + ADC PTQ Failed (exit code $EXIT_CODE)"
    echo "========================================"
    echo ""
    echo "Common issues:"
    echo "  - OOM: Reduce MAX_LENGTH or CALIBRATION_BATCH_SIZE"
    echo "  - Auth: Run 'huggingface-cli login' for gated models"
    echo "  - CUDA: Check GPU availability with 'nvidia-smi'"
fi
echo "========================================"
