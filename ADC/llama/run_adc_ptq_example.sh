#!/bin/bash
# ADC Post-Training Quantization (PTQ) for LLaMA models
# Supports: meta-llama/Llama-3.1-8B, meta-llama/Llama-3.2-3B, meta-llama/Llama-3.2-1B

# ============================================================
# MODEL CONFIGURATION - Change this to use different models
# ============================================================
MODEL_NAME="meta-llama/Llama-3.2-1B"  # Options:
                                       # - meta-llama/Llama-3.2-1B (smallest, fastest)
                                       # - meta-llama/Llama-3.2-3B (medium)
                                       # - meta-llama/Llama-3.1-8B (largest)

# Output directory (will have timestamp appended)
OUTPUT_DIR="./ADC/llama/checkpoints/outputs_llama_adc_ptq"

# ============================================================
# ADC Hardware Configuration
# ============================================================
BX=4              # Activation bits (4-bit for aggressive quantization)
BW=4              # Weight bits (4-bit for aggressive quantization)
BA=8              # ADC bits
K=4               # Hardware design parameter
ASHIFT=false      # A-shift quantization strategy:
                  #   false = symmetric/signed quantization (standard)
                  #   true  = asymmetric/unsigned + A-shift (optimal for SiLU outputs)
MVM_LIMIT=256     # Memory vector multiplication limit for tiling

# ============================================================
# Calibration Settings
# ============================================================
CALIBRATION_METHOD="percentile"  # Options: minmax, percentile, mse
NUM_CALIBRATION_BATCHES=128      # Number of batches for calibration
CALIBRATION_BATCH_SIZE=4         # Batch size during calibration (reduce if OOM)

# ============================================================
# Evaluation Settings
# ============================================================
EVAL_BATCH_SIZE=4                # Batch size for perplexity evaluation
MAX_LENGTH=512                   # Maximum sequence length
MAX_EVAL_BATCHES=100             # Maximum batches for evaluation

# ============================================================
# Model Loading Settings
# ============================================================
TORCH_DTYPE="float16"            # Options: float16, bfloat16, float32

# ============================================================
# WandB Settings
# ============================================================
WANDB_PROJECT="llama-adc-ptq"
MODEL_SHORT_NAME=$(echo $MODEL_NAME | sed 's/.*\///')
WANDB_RUN_NAME="ptq_${MODEL_SHORT_NAME}_bx${BX}_bw${BW}_ba${BA}_k${K}_${CALIBRATION_METHOD}"

# Seed for reproducibility
SEED=42

echo "========================================"
echo "LLaMA ADC Post-Training Quantization"
echo "========================================"
echo "Model:             $MODEL_NAME"
echo "Output Directory:  $OUTPUT_DIR"
echo ""
echo "ADC Configuration:"
echo "  Activation bits: $BX"
echo "  Weight bits:     $BW"
echo "  ADC bits:        $BA"
echo "  Hardware param:  $K"
echo "  MVM limit:       $MVM_LIMIT"
echo ""
echo "Quantization Strategy:"
echo "  A-shift:         $ASHIFT"
if [ "$ASHIFT" = true ]; then
    echo "  → Asymmetric (unsigned) + A-shift for SiLU outputs (down_proj)"
else
    echo "  → Symmetric (signed) for all activations"
fi
echo ""
echo "Calibration:"
echo "  Method:          $CALIBRATION_METHOD"
echo "  Batches:         $NUM_CALIBRATION_BATCHES"
echo "  Batch size:      $CALIBRATION_BATCH_SIZE"
echo ""
echo "Evaluation:"
echo "  Max batches:     $MAX_EVAL_BATCHES"
echo "  Batch size:      $EVAL_BATCH_SIZE"
echo "  Max length:      $MAX_LENGTH"
echo "========================================"
echo ""

# Build the command
CMD="python ADC/llama/runs/llama_adc_ptq.py \
    --model_name \"$MODEL_NAME\" \
    --output_dir \"$OUTPUT_DIR\" \
    --bx $BX \
    --bw $BW \
    --ba $BA \
    --k $K \
    --mvm_limit $MVM_LIMIT \
    --calibration_method $CALIBRATION_METHOD \
    --num_calibration_batches $NUM_CALIBRATION_BATCHES \
    --calibration_batch_size $CALIBRATION_BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --max_length $MAX_LENGTH \
    --max_eval_batches $MAX_EVAL_BATCHES \
    --torch_dtype $TORCH_DTYPE \
    --seed $SEED \
    --wandb_project \"$WANDB_PROJECT\" \
    --wandb_run_name \"$WANDB_RUN_NAME\""

# Add optional flags
if [ "$ASHIFT" = true ]; then
    CMD="$CMD --ashift"
fi

# Run the command
echo "Running PTQ calibration..."
echo ""
eval $CMD

EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ PTQ Complete!"
    echo "========================================"
    echo ""
    echo "📁 Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Files created:"
    echo "  - model.safetensors    (Calibrated model)"
    echo "  - config.json"
    echo "  - calibration_info.txt (Calibration details & perplexity)"
    echo ""
    echo "🔗 WandB: https://wandb.ai/your-username/$WANDB_PROJECT"
    echo ""
    echo "To use the calibrated model:"
    echo "  from transformers import AutoModelForCausalLM"
    echo "  model = AutoModelForCausalLM.from_pretrained(\"$OUTPUT_DIR\")"
else
    echo "✗ PTQ Failed with exit code $EXIT_CODE"
    echo "========================================"
    echo ""
    echo "Common issues:"
    echo "  - OOM: Reduce CALIBRATION_BATCH_SIZE or EVAL_BATCH_SIZE"
    echo "  - Auth: Run 'huggingface-cli login' for gated models"
    echo "  - CUDA: Check GPU availability with 'nvidia-smi'"
fi

echo "========================================"
