#!/usr/bin/env bash
# run_3b_debug.sh — diagnostic sweep for Llama-3.2-3B INT4+ADC catastrophe.
#
# The paper_comparison FQ_INT4 setup gives bypass=11.24 but ADC=889 PPL on 3B
# (vs 17.83/26.66 on 1B). This script runs 3 single-knob variants to isolate
# the cause:
#
#   A. 3b_int4_stageB_alpha0  — Stage B kept, but propagation α=0 (no ADC-noise
#                               injection into the calibration loss).
#   B. 3b_int4_no_stageB      — Stage B skipped entirely. Stage A + MLP-diag
#                               only. Tests if Stage B itself is the failure
#                               mode on 3B.
#   C. 3b_int4_bsz8           — cali_bsz=8, grad_accum=2 (same effective 16).
#                               Tests if small per-step batch + Stage B
#                               propagation produces noisy enough gradients
#                               to destabilise transforms on 28-layer 3B.
#
# Reference baseline: outputs/paper_comparison/results.json,
#                     run_name=llama-3_2-3b_int4_ptq → adc_wiki=889.
#
# Usage:
#   bash ADC/llama/run_3b_debug.sh                              # all three
#   ONLY=3b_int4_stageB_alpha0 bash ADC/llama/run_3b_debug.sh   # one
#   SKIP_COMPLETED=1 bash ADC/llama/run_3b_debug.sh             # resume

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNS="$SCRIPT_DIR/runs/llama_smooth_quant_adc_ptq.py"
OUT="${OUTPUT_DIR:-$SCRIPT_DIR/outputs/3b_debug}"
mkdir -p "$OUT"

# Reduce CUDA fragmentation — same as paper_comparison; matters for 3B Stage B.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ONLY="${ONLY:-}"
SKIP_COMPLETED="${SKIP_COMPLETED:-0}"

# ─── Run-filter helpers (same pattern as run_paper_comparison.sh) ───────────

is_in_only() {
    [[ -z "$ONLY" ]] && return 0
    local s="$1"
    IFS=',' read -ra _arr <<< "$ONLY"
    for x in "${_arr[@]}"; do [[ "$x" == "$s" ]] && return 0; done
    return 1
}

is_completed() {
    [[ "$SKIP_COMPLETED" != "1" ]] && return 1
    [[ ! -f "$OUT/results.json" ]] && return 1
    python -c "import json,sys; d=json.load(open('$OUT/results.json')); \
sys.exit(0 if any(e.get('run_name')=='$1' and e.get('status')=='success' for e in d) else 1)" 2>/dev/null
}

run_config() {
    local SUFFIX=$1; shift
    local RESULTS_JSON="$OUT/results.json"
    local LOG="$OUT/log_${SUFFIX}.txt"

    if ! is_in_only "$SUFFIX"; then
        echo ">>> [$SUFFIX] skipped (not in ONLY=$ONLY)" | tee -a "$OUT/run.log"
        return 0
    fi
    if is_completed "$SUFFIX"; then
        echo ">>> [$SUFFIX] skipped (already in results.json)" | tee -a "$OUT/run.log"
        return 0
    fi

    echo ""
    echo ">>> [$SUFFIX] starting" | tee -a "$OUT/run.log"
    if python "$RUNS" \
           --wandb_run_name "$SUFFIX" \
           --output_dir "$OUT/$SUFFIX" \
           --results_json_path "$RESULTS_JSON" \
           --disable_wandb \
           "$@" 2>&1 | tee "$LOG"; then
        echo "    [$SUFFIX] OK" | tee -a "$OUT/run.log"
    else
        echo "    [$SUFFIX] FAILED (exit $?) — skipping" | tee -a "$OUT/run.log"
    fi
}

# ─── Shared FQ_INT4 base (mirrors paper_comparison FQ_INT4 + 3B sizing) ─────

BASE_INT4=(
    --model_name "meta-llama/Llama-3.2-3B"
    --bx 4 --bw 4 --ba 8 --k 16
    --mvm_limit 256
    --preprocess_method flat_quant
    --fq_nsamples 1024
    --fq_epochs 30
    --fq_w_bits 4 --fq_a_bits 4
    --fq_lwc --fq_lac --fq_add_diag
    --fq_diag_mlp
    --lora_rank 0
)

COMMON_EVAL=(
    --eval_datasets wikitext2 c4
    --run_no_adc_eval
    --run_lm_eval
    --run_lm_eval_bypass
    --disable_visualizations
)

# Default 3B sizing — matches fq_cali_bsz_for / fq_grad_accum_for in
# run_paper_comparison.sh for Llama-3.2-3B.
DEFAULT_SIZING=(
    --fq_cali_bsz 4
    --fq_grad_accum_steps 4
)

# ─── Experiments ────────────────────────────────────────────────────────────

# A. Stage B kept (attn-diag still trains), but propagation α=0 → loss is
#    FP-only. Tests if propagation × depth is the cause.
run_config "3b_int4_stageB_alpha0" \
    "${BASE_INT4[@]}" \
    "${DEFAULT_SIZING[@]}" \
    --fq_stage_b_epochs 10 \
    --fq_stage_b_prop_alpha 0.0 \
    --fq_stage_b_diag_attn \
    "${COMMON_EVAL[@]}"

# B. Stage B entirely skipped. No attention diagonals, no propagation. Tests
#    if Stage B itself (independent of α) is the failure mode on 3B.
run_config "3b_int4_no_stageB" \
    "${BASE_INT4[@]}" \
    "${DEFAULT_SIZING[@]}" \
    --fq_stage_b_epochs 0 \
    "${COMMON_EVAL[@]}"

# C. Larger per-step batch with same effective 16. Tests if Stage B is too
#    fragile under the noisier 4×4 gradients used on 3B.
run_config "3b_int4_bsz8" \
    "${BASE_INT4[@]}" \
    --fq_cali_bsz 8 \
    --fq_grad_accum_steps 2 \
    --fq_stage_b_epochs 10 \
    --fq_stage_b_prop_alpha 0.5 \
    --fq_stage_b_diag_attn \
    "${COMMON_EVAL[@]}"

echo ""
echo "=== All done. Results: $OUT/results.json ==="
echo "=== Run log:           $OUT/run.log ==="
