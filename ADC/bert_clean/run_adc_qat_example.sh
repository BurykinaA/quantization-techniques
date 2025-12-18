#!/bin/bash
# Example script to run ADC QAT training with WandB logging and optional monitoring

# ==============================================================================
# Configuration
# ==============================================================================

# IMPORTANT: Since we're starting from PTQ (already ADC), use adc_resume_dir instead of fp_checkpoint_dir!
# Fine-tuned floating-point checkpoint to start from
FP_CHECKPOINT=""  # Leave empty when resuming from PTQ

# ADC PTQ checkpoint to resume from (already has calibrated ADC layers)
# This should point to the output of run_adc_ptq_example.sh
ADC_RESUME_DIR_DEFAULT="./ADC/bert_clean/checkpoints/outputs_adc_ptq_k16_fix_20251125"

# Where to store QAT outputs (checkpoints, logs, metrics)
OUTPUT_DIR="./ADC/bert_clean/checkpoints/outputs_adc_qat_k16_conservative"

# ADC hardware configuration - MUST MATCH PTQ CHECKPOINT!
BX=8                # Activation bits
BW=8                 # Weight bits
BA=8                 # ADC bits
K=16                 # Hardware design parameter (k=16 gave F1=65 vs k=4 gave F1=17)
ASHIFT=false         # MUST MATCH PTQ checkpoint! (PTQ was created with ashift=false)
MVM_LIMIT=256        # Tile size limit for MVM units

# Training hyper-parameters
# ============================================================================
# PAPER SETTINGS (for training from scratch or from QAT checkpoint):
#   NUM_EPOCHS=4, TRAIN_BATCH_SIZE=16, LEARNING_RATE=3e-5, KURTOSIS_LAMBDA=0.0006
# CONSERVATIVE SETTINGS (for resuming from PTQ - if training destabilizes):
#   NUM_EPOCHS=3, TRAIN_BATCH_SIZE=8, LEARNING_RATE=1e-6, KURTOSIS_LAMBDA=0.0
# ============================================================================
# NOTE: With round_ste fix for proper gradient flow, the autograd graph is larger.
# Reduced batch size and enabled FP16 to fit in GPU memory.
NUM_EPOCHS=1         # Paper: 4 epochs
TRAIN_BATCH_SIZE=8   # Reduced from 16 due to larger gradient graph with STE fix
EVAL_BATCH_SIZE=32
LEARNING_RATE=1e-6   # Paper: 0.00003 initial LR
WARMUP_RATIO=0.0     # Paper: linear decay (no warmup mentioned)
WARMUP_STEPS=0
EVAL_STEPS=50
SAVE_STEPS=500
SAVE_TOTAL_LIMIT=3
KURTOSIS_LAMBDA=0.0006  # Paper: W-reshape regularization
DROPOUT=0.2          # Paper: 0.2 for BERT-base
LR_SCHEDULER="linear" # Paper: linear decay
USE_FP16=false        # Enable FP16 to reduce memory with larger gradient graph
FIXED_DELTA=true     # Keep delta fixed (already calibrated in PTQ)
EVAL_ONLY=false      # Set true to skip training and only run evaluation

# Monitoring
ENABLE_ADC_MONITORING=true
ADC_RESUME_DIR="$ADC_RESUME_DIR_DEFAULT"  # Resume from PTQ checkpoint

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
if [ -n "$FP_CHECKPOINT" ]; then
    echo "FP Checkpoint:       $FP_CHECKPOINT"
fi
if [ -n "$ADC_RESUME_DIR" ]; then
    echo "Resume from PTQ:     $ADC_RESUME_DIR"
fi
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
echo "  Dropout:           $DROPOUT"
echo "  LR scheduler:      $LR_SCHEDULER"
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
    --dropout $DROPOUT
    --lr_scheduler_type $LR_SCHEDULER
    --seed $SEED
)

# Add fp_checkpoint_dir only if not empty
if [ -n "$FP_CHECKPOINT" ]; then
    CMD+=(--fp_checkpoint_dir "$FP_CHECKPOINT")
fi

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