#!/usr/bin/env bash
# run_int_only_baseline_1b.sh
#
# Train two ADC-naive INT PTQ baselines for Llama-3.2-1B:
#
#   llama-3_2-1b_int8_ptq_noadc   — bw=bx=ba=8, k=4   (pure INT8 PTQ baseline)
#   llama-3_2-1b_int4_ptq_noadc   — bw=bx=4, ba=8, k=16   (pure INT4 PTQ baseline)
#
# Key differences from run_paper_comparison.sh:
#   --fq_no_adc_loss        FlatQuant trained on plain INT fake-quant loss (no
#                           floor_ste / tiling / ADC penalties in the forward).
#   --fq_stage_b_epochs 0   Stage B skipped — no ADC propagation in any stage.
#
# These are the "true INT4/INT8 PTQ baselines" for the README table:
#   INT8 PTQ        row     ← int8_ptq_noadc bypass-mode numbers
#   INT4 PTQ        row     ← int4_ptq_noadc bypass-mode numbers
#
# Time: ~1 h calibration + ~1 h eval (8 tasks × 2 modes) per config → ~4 h total.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNS="$SCRIPT_DIR/runs/llama_smooth_quant_adc_ptq.py"
OUT="${OUTPUT_DIR:-$SCRIPT_DIR/outputs/paper_comparison}"
RESULTS_JSON="$OUT/results.json"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$OUT"

run_config() {
    local SUFFIX=$1; shift
    local LOG="$OUT/log_${SUFFIX}.txt"
    echo ""
    echo ">>> [$SUFFIX] starting" | tee -a "$OUT/int_only_baseline.log"
    if python "$RUNS" \
           --wandb_run_name "$SUFFIX" \
           --output_dir "$OUT/$SUFFIX" \
           --results_json_path "$RESULTS_JSON" \
           --disable_wandb \
           "$@" 2>&1 | tee "$LOG"; then
        echo "    [$SUFFIX] OK" | tee -a "$OUT/int_only_baseline.log"
    else
        echo "    [$SUFFIX] FAILED (exit $?) — skipping" | tee -a "$OUT/int_only_baseline.log"
    fi
}

# Common FlatQuant args: ADC-naive, single-stage, MLP-only diag (same as Stage A
# of run_paper_comparison.sh but with ADC simulation OFF in the loss).
FQ_NOADC=(
    --preprocess_method flat_quant
    --fq_no_adc_loss                       # <-- the key flag
    --fq_nsamples 1024
    --fq_cali_bsz 16 --fq_grad_accum_steps 1
    --fq_epochs 30
    --fq_stage_b_epochs 0                  # <-- no Stage B
    --fq_lwc --fq_lac --fq_add_diag
    --fq_diag_mlp
    --mvm_limit 256
)

# All 8 lm-eval tasks; record both bypass and ADC PPL + lm-eval in both modes.
COMMON_EVAL=(
    --eval_datasets wikitext2 c4
    --run_no_adc_eval                      # bypass + ADC PPL
    --run_lm_eval
    --run_lm_eval_bypass                   # bypass + ADC accuracy
    --lm_eval_tasks hellaswag winogrande mmlu arc_easy arc_challenge piqa openbookqa boolq
    --disable_visualizations
)

# 1. INT8 PTQ (ADC-naive baseline)
run_config "llama-3_2-1b_int8_ptq_noadc" \
    --model_name "meta-llama/Llama-3.2-1B" \
    --bx 8 --bw 8 --ba 8 --k 4 \
    --lora_rank 0 \
    --fq_w_bits 8 --fq_a_bits 8 \
    "${FQ_NOADC[@]}" \
    "${COMMON_EVAL[@]}"

# 2. INT4 PTQ (ADC-naive baseline)
run_config "llama-3_2-1b_int4_ptq_noadc" \
    --model_name "meta-llama/Llama-3.2-1B" \
    --bx 4 --bw 4 --ba 8 --k 16 \
    --lora_rank 0 \
    --fq_w_bits 4 --fq_a_bits 4 \
    "${FQ_NOADC[@]}" \
    "${COMMON_EVAL[@]}"

echo ""
echo "=== int_only_baseline done.  Updated: $RESULTS_JSON ==="
echo "=== Log: $OUT/int_only_baseline.log ==="
