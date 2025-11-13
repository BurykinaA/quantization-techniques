#!/bin/bash
# Example script to run ADC QAT training with WandB logging and optional monitoring

# ==============================================================================
# Configuration
# ==============================================================================

# Fine-tuned floating-point checkpoint to start from
FP_CHECKPOINT="./ADC/bert_clean/checkpoints/outputs_adc_ptq_asymmetric_20251106"

# Where to store QAT outputs (checkpoints, logs, metrics)
OUTPUT_DIR="./ADC/bert_clean/checkpoints/outputs_adc_qat_fixed_delta"

# ADC hardware configuration
BX=8                 # Activation bits
BW=8                 # Weight bits
BA=8                 # ADC bits
K=4                  # Hardware design parameter
ASHIFT=true          # Enable A-shift (unsigned activations after GeLU)
MVM_LIMIT=256        # Tile size limit for MVM units

# Training hyper-parameters
NUM_EPOCHS=2
TRAIN_BATCH_SIZE=32
EVAL_BATCH_SIZE=64
LEARNING_RATE=3e-5
WARMUP_RATIO=0.0
WARMUP_STEPS=0
EVAL_STEPS=5
SAVE_STEPS=500
SAVE_TOTAL_LIMIT=3
KURTOSIS_LAMBDA=0.05  # Start stable; re-enable later (e.g., 0.01) after warmup
USE_FP16=false       # Start in fp32; turn on later when stable
FIXED_DELTA=true     # Set true to disable dynamic delta / annealing
EVAL_ONLY=false      # Set true to skip training and only run evaluation

# Monitoring
ENABLE_ADC_MONITORING=true
ADC_RESUME_DIR=""     # Provide path to resume from ADC checkpoint, or leave empty

# Reproducibility
SEED=42

# WandB settings
WANDB_ENABLE=true
WANDB_PROJECT="bert-adc-qat"
WANDB_RUN_NAME="qat_bx${BX}_bw${BW}_ba${BA}_k${K}"
WANDB_TAGS=("qat" "adc" "${BX}bx" "${BW}bw" "${BA}ba")
WANDB_NOTES="ADC QAT demo run on SQuAD v1.1"

if [ "$ASHIFT" = true ]; then
    WANDB_TAGS+=("ashift")
else
    WANDB_TAGS+=("symmetric")
fi

# ==============================================================================
# Display configuration
# ==============================================================================

echo "========================================"
echo "BERT ADC QAT Training"
echo "========================================"
echo "FP Checkpoint:       $FP_CHECKPOINT"
echo "Output Directory:    $OUTPUT_DIR"
echo ""
echo "ADC Configuration:"
echo "  Activation bits:   $BX"
echo "  Weight bits:       $BW"
echo "  ADC bits:          $BA"
echo "  Hardware param:    $K"
echo "  MVM limit:         $MVM_LIMIT"
echo "  A-shift:           $ASHIFT"
echo ""
echo "Training:"
echo "  Epochs:            $NUM_EPOCHS"
echo "  Train batch size:  $TRAIN_BATCH_SIZE"
echo "  Eval batch size:   $EVAL_BATCH_SIZE"
echo "  Learning rate:     $LEARNING_RATE"
echo "  Warmup ratio:      $WARMUP_RATIO"
echo "  Warmup steps:      $WARMUP_STEPS"
echo "  Eval steps:        $EVAL_STEPS"
echo "  Save steps:        $SAVE_STEPS"
echo "  Save limit:        $SAVE_TOTAL_LIMIT"
echo "  Kurtosis λ:        $KURTOSIS_LAMBDA"
echo "  FP16:              $USE_FP16"
echo "  Fixed delta:       $FIXED_DELTA"
echo "  Eval only:         $EVAL_ONLY"
echo ""
echo "Monitoring:"
echo "  ADC monitoring:    $ENABLE_ADC_MONITORING"
if [ -n "$ADC_RESUME_DIR" ]; then
    echo "  Resume from:       $ADC_RESUME_DIR"
fi
echo ""
echo "WandB:"
echo "  Enabled:           $WANDB_ENABLE"
echo "  Project:           $WANDB_PROJECT"
echo "  Run name:          $WANDB_RUN_NAME"
echo "  Tags:              ${WANDB_TAGS[*]}"
echo "========================================"
echo ""

# ==============================================================================
# Build command
# ==============================================================================

CMD=(
    python ADC/bert_clean/runs/bert_adc_integration.py
    --fp_checkpoint_dir "$FP_CHECKPOINT"
    --output_dir "$OUTPUT_DIR"
    --bx $BX
    --bw $BW
    --ba $BA
    --k $K
    --mvm_limit $MVM_LIMIT
    --num_train_epochs $NUM_EPOCHS
    --per_device_train_batch_size $TRAIN_BATCH_SIZE
    --per_device_eval_batch_size $EVAL_BATCH_SIZE
    --learning_rate $LEARNING_RATE
    --warmup_ratio $WARMUP_RATIO
    --warmup_steps $WARMUP_STEPS
    --eval_steps $EVAL_STEPS
    --save_steps $SAVE_STEPS
    --save_total_limit $SAVE_TOTAL_LIMIT
    --kurtosis_lambda $KURTOSIS_LAMBDA
    --seed $SEED
)

if [ "$ASHIFT" = true ]; then
    CMD+=(--ashift)
fi

if [ "$USE_FP16" = true ]; then
    CMD+=(--fp16)
fi

if [ "$ENABLE_ADC_MONITORING" = false ]; then
    CMD+=(--disable_adc_monitoring)
fi

if [ "$FIXED_DELTA" = true ]; then
    CMD+=(--fixed_delta)
fi

if [ "$EVAL_ONLY" = true ]; then
    CMD+=(--eval_only)
fi

if [ -n "$ADC_RESUME_DIR" ]; then
    CMD+=(--adc_resume_dir "$ADC_RESUME_DIR")
fi

if [ "$WANDB_ENABLE" = true ]; then
    CMD+=(--wandb_project "$WANDB_PROJECT")
    CMD+=(--wandb_run_name "$WANDB_RUN_NAME")
    if [ ${#WANDB_TAGS[@]} -gt 0 ]; then
        CMD+=(--wandb_tags "${WANDB_TAGS[@]}")
    fi
    if [ -n "$WANDB_NOTES" ]; then
        CMD+=(--wandb_notes "$WANDB_NOTES")
    fi
else
    CMD+=(--disable_wandb)
fi

echo "Running ADC QAT..."
echo ""
echo "${CMD[@]}"
echo ""

"${CMD[@]}"
EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ QAT run complete!"
    echo "========================================"
    echo ""
    echo "📁 Outputs saved to: $OUTPUT_DIR"
    echo ""
    echo "Files to expect:"
    echo "  - pytorch_model.bin / model.safetensors (trained model)"
    echo "  - config.json"
    echo "  - eval_metrics.txt (final eval metrics)"
    echo "  - adc_config.txt (quantization + hardware settings)"
    echo "  - trainer_state.json (HF Trainer state)"
    echo ""
    if [ "$WANDB_ENABLE" = true ]; then
        echo "🔗 WandB dashboard: https://wandb.ai/your-username/$WANDB_PROJECT"
    fi
else
    echo "✗ QAT run failed with exit code $EXIT_CODE"
    echo "========================================"
    echo ""
    echo "Check the logs above for details."
fi

echo "========================================"