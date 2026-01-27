#!/bin/bash
# Example script to run ADC-LoRA QAT training (Paper Section 3.4: Training Overhead Reduction)
#
# ADC-LoRA reduces trainable parameters by keeping base weights frozen and only
# training low-rank adaptation matrices A and B.
# Formula: Y = QA(Qx(X)Qw(W + AB))

# ==============================================================================
# Configuration
# ==============================================================================

# IMPORTANT: Start from PTQ checkpoint (already calibrated ADC layers)
# This should point to the output of run_adc_ptq_example.sh
ADC_RESUME_DIR="./ADC/bert_clean_old/checkpoints/outputs_adc_ptq_k16_fix_20251125"

# Alternative: Start from FP checkpoint (will calibrate quantizers from scratch)
# FP_CHECKPOINT="./path/to/bert_squad_fp"

# Where to store ADC-LoRA outputs (checkpoints, logs, metrics)
OUTPUT_DIR="./ADC/bert_clean/checkpoints/outputs_adc_lora"

# ADC hardware configuration - MUST MATCH PTQ CHECKPOINT!
BX=8                # Activation bits
BW=8                # Weight bits
BA=8                # ADC bits
K=16                # Hardware design parameter
ASHIFT=false        # MUST MATCH PTQ checkpoint!
MVM_LIMIT=256       # Tile size limit for MVM units

# ==============================================================================
# LoRA Configuration (Paper Section 3.4)
# ==============================================================================
# ADC-LoRA keeps base weights W frozen and only trains low-rank matrices A, B
# where effective weight = W + (alpha/r) * A @ B
# This dramatically reduces trainable parameters (~48x for r=8)
# ==============================================================================

USE_LORA=true
LORA_R=8                          # LoRA rank (smaller = fewer params, less capacity)
LORA_ALPHA=16.0                   # LoRA scaling factor (scaling = alpha/r)
LORA_DROPOUT=0.0                  # LoRA dropout (0 = no dropout)
LORA_TARGET_MODULES="query value" # Which modules to apply LoRA (space-separated)

# LoRA MSE Warmup (Paper Equation 12)
# Initializes A, B to minimize ||Qx(X)Qw(W) - QA(Qx(X)Qw(W + AB))||^2
# This helps LoRA compensate for ADC quantization error from the start
LORA_WARMUP_STEPS=100             # Number of MSE warmup steps (0 = disabled)
LORA_WARMUP_LR=0.001              # Learning rate for warmup

# ==============================================================================
# Training hyper-parameters
# ==============================================================================
# With LoRA, we can use larger learning rates since we're only training
# a small number of parameters. The base model weights are frozen.
# ==============================================================================

NUM_EPOCHS=3
TRAIN_BATCH_SIZE=8   # Can use larger batch with LoRA (less memory for gradients)
EVAL_BATCH_SIZE=16
LEARNING_RATE=1e-4    # Higher LR is OK for LoRA (fewer params to update)
WARMUP_RATIO=0.1
WARMUP_STEPS=0
EVAL_STEPS=10
SAVE_STEPS=500
SAVE_TOTAL_LIMIT=3
KURTOSIS_LAMBDA=0.0006  # Paper: W-reshape regularization (applied to effective weight)
DROPOUT=0.2             # Paper: 0.2 for BERT-base
LR_SCHEDULER="linear"
USE_FP16=false
FIXED_DELTA=true        # Keep delta fixed (already calibrated in PTQ)
EVAL_ONLY=false

# BitAug configuration (Paper: Bit Augmentation)
# With LoRA, BitAug has lower memory overhead since gradients are smaller
USE_BITAUG=false
BITAUG_LAMBDA=0.5
BITAUG_NEIGHBOR_RANGE=1
BITAUG_MIN_BITS=4
BITAUG_MAX_BITS=12

# Monitoring
ENABLE_ADC_MONITORING=true

# Reproducibility
SEED=42

# WandB settings
WANDB_ENABLE=true
WANDB_PROJECT="bert-adc-lora"
WANDB_RUN_NAME="lora_r${LORA_R}_bx${BX}_bw${BW}_ba${BA}_k${K}"
WANDB_TAGS=("lora" "adc" "r${LORA_R}" "${BX}bx" "${BW}bw" "${BA}ba")
WANDB_NOTES="ADC-LoRA QAT on SQuAD v1.1 (Paper Section 3.4)"

if [ "$ASHIFT" = true ]; then
    WANDB_TAGS+=("ashift")
else
    WANDB_TAGS+=("symmetric")
fi

if [ "$USE_BITAUG" = true ]; then
    WANDB_TAGS+=("bitaug")
fi

if [ "$LORA_WARMUP_STEPS" -gt 0 ]; then
    WANDB_TAGS+=("mse_warmup")
fi

# ==============================================================================
# Display configuration
# ==============================================================================

echo "========================================"
echo "BERT ADC-LoRA QAT Training"
echo "(Paper Section 3.4: Training Overhead Reduction)"
echo "========================================"
if [ -n "$ADC_RESUME_DIR" ]; then
    echo "Resume from PTQ:     $ADC_RESUME_DIR"
elif [ -n "$FP_CHECKPOINT" ]; then
    echo "FP Checkpoint:       $FP_CHECKPOINT"
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
echo "LoRA Configuration:"
echo "  Enabled:           $USE_LORA"
echo "  Rank (r):          $LORA_R"
echo "  Alpha:             $LORA_ALPHA"
echo "  Scaling:           $(python3 -c "print(${LORA_ALPHA} / ${LORA_R})" 2>/dev/null || echo "${LORA_ALPHA}/${LORA_R}")"
echo "  Dropout:           $LORA_DROPOUT"
echo "  Target modules:    $LORA_TARGET_MODULES"
echo "  MSE warmup steps:  $LORA_WARMUP_STEPS"
echo "  MSE warmup LR:     $LORA_WARMUP_LR"
echo ""
echo "Training:"
echo "  Epochs:            $NUM_EPOCHS"
echo "  Train batch size:  $TRAIN_BATCH_SIZE"
echo "  Eval batch size:   $EVAL_BATCH_SIZE"
echo "  Learning rate:     $LEARNING_RATE"
echo "  Warmup ratio:      $WARMUP_RATIO"
echo "  Eval steps:        $EVAL_STEPS"
echo "  Kurtosis λ:        $KURTOSIS_LAMBDA"
echo "  Dropout:           $DROPOUT"
echo "  LR scheduler:      $LR_SCHEDULER"
echo ""
echo "BitAug:"
echo "  Enabled:           $USE_BITAUG"
if [ "$USE_BITAUG" = true ]; then
    echo "  Lambda:            $BITAUG_LAMBDA"
    echo "  Neighbor range:    ±$BITAUG_NEIGHBOR_RANGE bits"
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
    python -m ADC.bert_clean.runs.bert_adc_integration
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

# LoRA flags
if [ "$USE_LORA" = true ]; then
    CMD+=(--use_lora)
    CMD+=(--lora_r $LORA_R)
    CMD+=(--lora_alpha $LORA_ALPHA)
    CMD+=(--lora_dropout $LORA_DROPOUT)
    CMD+=(--lora_target_modules $LORA_TARGET_MODULES)
    CMD+=(--lora_warmup_steps $LORA_WARMUP_STEPS)
    CMD+=(--lora_warmup_lr $LORA_WARMUP_LR)
fi

# Checkpoint source
if [ -n "$ADC_RESUME_DIR" ]; then
    CMD+=(--adc_resume_dir "$ADC_RESUME_DIR")
elif [ -n "$FP_CHECKPOINT" ]; then
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

# BitAug flags
if [ "$USE_BITAUG" = true ]; then
    CMD+=(--bitaug)
    CMD+=(--bitaug_lambda $BITAUG_LAMBDA)
    CMD+=(--bitaug_neighbor_range $BITAUG_NEIGHBOR_RANGE)
    CMD+=(--bitaug_min_bits $BITAUG_MIN_BITS)
    CMD+=(--bitaug_max_bits $BITAUG_MAX_BITS)
fi

# WandB flags
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

echo "Running ADC-LoRA QAT..."
echo ""
echo "${CMD[@]}"
echo ""

"${CMD[@]}"
EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "ADC-LoRA QAT complete!"
    echo "========================================"
    echo ""
    echo "Outputs saved to: $OUTPUT_DIR"
    echo ""
    echo "Expected files:"
    echo "  - pytorch_model.bin / model.safetensors (model with LoRA weights)"
    echo "  - config.json"
    echo "  - trainer_state.json"
    echo ""
    echo "LoRA benefits:"
    echo "  - ~${LORA_R}x fewer trainable params than full fine-tuning"
    echo "  - Lower GPU memory usage (especially with BitAug)"
    echo "  - LoRA weights can be merged into base for inference"
    echo ""
    if [ "$WANDB_ENABLE" = true ]; then
        echo "WandB dashboard: https://wandb.ai/$WANDB_PROJECT"
    fi
else
    echo "ADC-LoRA QAT failed with exit code $EXIT_CODE"
    echo "========================================"
    echo ""
    echo "Check the logs above for details."
fi

echo "========================================"
