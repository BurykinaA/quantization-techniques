#!/bin/bash
# FlatQuant + ADC Post-Training Quantization (PTQ) for LLaMA models
# Pipeline: FlatQuant (learnable transforms) -> ADC Convert -> Calibrate -> Evaluate
#
# FlatQuant trains Kronecker-decomposed orthogonal transforms + diagonal scaling
# + learnable weight/activation clipping layer-by-layer using MSE loss, then
# folds transforms into weights before ADC conversion.
#
# Reference: Sun et al., "FlatQuant: Flatness Matters for LLM Quantization", ICML 2025
# Official: https://github.com/ruikangliu/FlatQuant
#
# Supports: meta-llama/Llama-3.1-8B, meta-llama/Llama-3.2-3B, meta-llama/Llama-3.2-1B

# ============================================================
# MODEL CONFIGURATION
# ============================================================
MODEL_NAME="meta-llama/Llama-3.2-1B"

# Output directory (timestamp appended automatically)
OUTPUT_DIR="./ADC/llama/checkpoints/outputs_llama_flat_quant_adc_ptq"

# ============================================================
# FlatQuant Configuration
# ============================================================
FQ_W_BITS=8               # Weight quantizer bits during FQ calibration
FQ_A_BITS=8               # Activation quantizer bits during FQ calibration
FQ_NSAMPLES=128            # Calibration samples for FQ training
FQ_CALI_BSZ=4              # Batch size for layer-by-layer calibration
FQ_EPOCHS=15               # Training epochs per layer
FQ_LR=0.005                # AdamW learning rate for transforms
FQ_DIAG_ALPHA=0.5          # Diagonal scale init (SQ-style)
FQ_ADD_DIAG=true           # Per-channel diagonal scaling
FQ_LWC=true                # Learnable weight clipping
FQ_LAC=true                # Learnable activation clipping
FQ_SAVE_TRANSFORMS=true    # Save trained transforms
FQ_RELOAD_PATH=""          # Load pre-trained transforms (skip training)

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
WANDB_RUN_NAME="fq_ptq_${MODEL_SHORT_NAME}_w${FQ_W_BITS}a${FQ_A_BITS}_e${FQ_EPOCHS}_bx${BX}_bw${BW}_ba${BA}_k${K}_${CALIBRATION_METHOD}"
SEED=42

echo "========================================"
echo "LLaMA FlatQuant + ADC PTQ"
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
    if [ "$FQ_SAVE_TRANSFORMS" = true ]; then
        echo "  Transforms:     $OUTPUT_DIR/flat_quant_transforms.pt"
    fi
    echo "WandB:    https://wandb.ai/your-username/$WANDB_PROJECT"
else
    echo "FlatQuant + ADC PTQ Failed (exit code $EXIT_CODE)"
    echo "========================================"
    echo ""
    echo "Common issues:"
    echo "  - OOM: Reduce FQ_NSAMPLES, FQ_CALI_BSZ, MAX_LENGTH, or CALIBRATION_BATCH_SIZE"
    echo "  - Auth: Run 'huggingface-cli login' for gated models"
    echo "  - CUDA: Check GPU availability with 'nvidia-smi'"
fi
echo "========================================"
