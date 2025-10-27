#!/bin/bash
# Example script to run QAT training with WandB logging

# Configuration
FP_CHECKPOINT="./bert-squad-fp-checkpoint"  # Change this to your FP checkpoint
OUTPUT_DIR="./ADC/bert_clean/checkpoints/outputs_qat"
WEIGHT_BITS=8
ACTIVATION_BITS=8
NUM_EPOCHS=3
LEARNING_RATE=3e-5
BATCH_SIZE=16
KURTOSIS_LAMBDA=0.05

# WandB settings
WANDB_PROJECT="bert-qat-squad"
WANDB_RUN_NAME="qat_w${WEIGHT_BITS}a${ACTIVATION_BITS}_lr${LEARNING_RATE}"

# Layers to visualize (first, middle, last)
VISUALIZE_LAYERS=(
    "bert.encoder.layer.0.attention.output.dense"
    "bert.encoder.layer.5.intermediate.dense"
    "bert.encoder.layer.11.output.dense"
)

echo "================================"
echo "BERT QAT Training on SQuAD"
echo "================================"
echo "FP Checkpoint: $FP_CHECKPOINT"
echo "Weight bits: $WEIGHT_BITS"
echo "Activation bits: $ACTIVATION_BITS"
echo "Epochs: $NUM_EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "W-reshape lambda: $KURTOSIS_LAMBDA"
echo "================================"
echo ""

python ADC/bert_clean/runs/bert_qat_integration.py \
    --fp_checkpoint_dir "$FP_CHECKPOINT" \
    --output_dir "$OUTPUT_DIR" \
    --weight_bits $WEIGHT_BITS \
    --activation_bits $ACTIVATION_BITS \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --kurtosis_lambda $KURTOSIS_LAMBDA \
    --use_wandb \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "$WANDB_RUN_NAME" \
    --wandb_tags "${WEIGHT_BITS}bit" "w-reshape" "squad" \
    --visualize_layers "${VISUALIZE_LAYERS[@]}" \
    --visualize_every_n_epochs 1 \
    --fp16 \
    --eval_steps 500 \
    --save_steps 500

echo ""
echo "================================"
echo "Training complete!"
echo "Check WandB: https://wandb.ai/your-username/$WANDB_PROJECT"
echo "Local outputs: $OUTPUT_DIR"
echo "================================"

