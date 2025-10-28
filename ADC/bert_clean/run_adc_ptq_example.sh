#!/bin/bash
# Example script to run ADC Post-Training Quantization (PTQ) with WandB logging and visualizations

# Configuration
QAT_CHECKPOINT="./ADC/bert_clean/checkpoints/outputs_qat/checkpoint-1000"  # Change this to your QAT checkpoint
OUTPUT_DIR="./ADC/bert_clean/outputs_adc_ptq"

# ADC Hardware Configuration
BX=8              # Activation bits
BW=8              # Weight bits
BA=8              # ADC bits
K=4               # Hardware design parameter
ASHIFT=false      # Enable A-shift (set to true if needed)
SIGNED_ACT=true   # Use signed activation quantization (RECOMMENDED)
MVM_LIMIT=256     # Memory vector multiplication limit for tiling

# Calibration Settings
CALIBRATION_METHOD="percentile"  # Options: minmax, percentile, mse
NUM_CALIBRATION_BATCHES=100     # Number of batches for calibration
CALIBRATION_BATCH_SIZE=8        # Batch size during calibration

# Evaluation Settings
EVAL_BATCH_SIZE=32
MAX_LENGTH=384
DOC_STRIDE=128

# WandB Settings
WANDB_PROJECT="bert-adc-ptq"
WANDB_RUN_NAME="ptq_bx${BX}_bw${BW}_ba${BA}_k${K}_${CALIBRATION_METHOD}"

# Monitoring Settings
DISABLE_MONITORING=false  # Set to true to disable ADC visualizations
MONITORED_LAYERS=3        # Number of layers to monitor (affects visualization count)

# Seed for reproducibility
SEED=42

echo "========================================"
echo "BERT ADC Post-Training Quantization"
echo "========================================"
echo "QAT Checkpoint:    $QAT_CHECKPOINT"
echo "Output Directory:  $OUTPUT_DIR"
echo ""
echo "ADC Configuration:"
echo "  Activation bits: $BX"
echo "  Weight bits:     $BW"
echo "  ADC bits:        $BA"
echo "  Hardware param:  $K"
echo "  A-shift:         $ASHIFT"
echo "  Signed acts:     $SIGNED_ACT"
echo "  MVM limit:       $MVM_LIMIT"
echo ""
echo "Calibration:"
echo "  Method:          $CALIBRATION_METHOD"
echo "  Batches:         $NUM_CALIBRATION_BATCHES"
echo "  Batch size:      $CALIBRATION_BATCH_SIZE"
echo ""
echo "Monitoring:"
echo "  Visualizations:  $([ "$DISABLE_MONITORING" = true ] && echo "Disabled" || echo "Enabled")"
echo "  Monitored layers: $MONITORED_LAYERS"
echo "========================================"
echo ""

# Build the command
CMD="python ADC/bert_clean/runs/bert_adc_ptq.py \
    --qat_checkpoint_dir \"$QAT_CHECKPOINT\" \
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
    --doc_stride $DOC_STRIDE \
    --seed $SEED \
    --wandb_project \"$WANDB_PROJECT\" \
    --wandb_run_name \"$WANDB_RUN_NAME\" \
    --monitored_layers $MONITORED_LAYERS"

# Add optional flags
if [ "$ASHIFT" = true ]; then
    CMD="$CMD --ashift"
fi

if [ "$SIGNED_ACT" = true ]; then
    CMD="$CMD --signed_activations"
fi

if [ "$DISABLE_MONITORING" = true ]; then
    CMD="$CMD --disable_adc_monitoring"
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
    echo "  - pytorch_model.bin       (Calibrated model)"
    echo "  - config.json"
    echo "  - calibration_info.txt    (Calibration details)"
    echo "  - eval_metrics.txt        (F1, EM scores)"
    
    if [ "$DISABLE_MONITORING" != true ]; then
        echo "  - adc_visualizations/     (📊 ADC pipeline plots)"
        echo ""
        echo "📊 Visualizations include:"
        echo "  - Full pipeline plots (before/after ADC)"
        echo "  - Distribution histograms"
        echo "  - Evolution over calibration"
    fi
    
    echo ""
    echo "🔗 WandB: https://wandb.ai/your-username/$WANDB_PROJECT"
    echo ""
    echo "To use the calibrated model:"
    echo "  python your_inference_script.py --model_path \"$OUTPUT_DIR\""
else
    echo "✗ PTQ Failed with exit code $EXIT_CODE"
    echo "========================================"
    echo ""
    echo "Check the error messages above for details."
fi

echo "========================================"

