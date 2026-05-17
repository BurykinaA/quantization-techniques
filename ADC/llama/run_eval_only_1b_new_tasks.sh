#!/usr/bin/env bash
# run_eval_only_1b_new_tasks.sh
#
# Add 5 new lm-eval tasks (arc_easy, arc_challenge, piqa, openbookqa, boolq)
# to the existing 1B entries in results.json — WITHOUT recalibration.
#
# Phase 2 mapping:
#   FP16                     ← --fp_only (load plain Llama-3.2-1B, lm-eval only)
#   INT8+ADC PTQ             ← eval_only.py on saved llama-3_2-1b_int8_ptq    (ADC mode)
#   INT4+ADC PTQ             ← eval_only.py on saved llama-3_2-1b_int4_ptq    (ADC mode)
#   INT4+ADC+LoRA            ← eval_only.py on saved llama-3_2-1b_int4_lora   (ADC mode)
#
# The "INT8 PTQ" and "INT4 PTQ" rows of README come from a SEPARATE run
# (`run_int_only_baseline_1b.sh`), which calibrates ADC-naive baselines from
# scratch — not from this script.
#
# Smoke test (no eval):
#   DRY_RUN=1 bash ADC/llama/run_eval_only_1b_new_tasks.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNS="$SCRIPT_DIR/runs/eval_only.py"
OUT="${OUTPUT_DIR:-$SCRIPT_DIR/outputs/paper_comparison}"
RESULTS_JSON="$OUT/results.json"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DRY_RUN="${DRY_RUN:-0}"
DRY_RUN_FLAG=()
[[ "$DRY_RUN" == "1" ]] && DRY_RUN_FLAG=(--dry_run_state_dict_check)

NEW_TASKS=(arc_easy arc_challenge piqa openbookqa boolq)
COMMON=(
    --model_name meta-llama/Llama-3.2-1B
    --results_json_path "$RESULTS_JSON"
    --lm_eval_tasks "${NEW_TASKS[@]}"
    --lm_eval_batch_size 4
    --run_lm_eval
    --mvm_limit 256
    --fq_lwc --fq_lac --fq_add_diag
)

eval_config() {
    local SUFFIX=$1; shift
    local CKPT="$OUT/$SUFFIX"
    if [[ ! -d "$CKPT" ]]; then
        echo ">>> [$SUFFIX] MISSING $CKPT — skipping" | tee -a "$OUT/eval_only.log"
        return 0
    fi
    echo "" | tee -a "$OUT/eval_only.log"
    echo ">>> [$SUFFIX] starting eval_only" | tee -a "$OUT/eval_only.log"
    if python "$RUNS" \
            --checkpoint_dir "$CKPT" \
            --run_name "$SUFFIX" \
            "${COMMON[@]}" \
            "${DRY_RUN_FLAG[@]}" \
            "$@" 2>&1 | tee "$OUT/log_eval_only_${SUFFIX}.txt"; then
        echo "    [$SUFFIX] OK" | tee -a "$OUT/eval_only.log"
    else
        echo "    [$SUFFIX] FAILED (exit $?)" | tee -a "$OUT/eval_only.log"
    fi
}

eval_fp() {
    # FP16: no checkpoint to load — call eval_only.py with --fp_only.
    # --dry_run_state_dict_check is a no-op here (no state_dict to check).
    local SUFFIX="llama-3_2-1b_fp"
    echo "" | tee -a "$OUT/eval_only.log"
    echo ">>> [$SUFFIX] starting eval_only (--fp_only)" | tee -a "$OUT/eval_only.log"
    if python "$RUNS" \
            --fp_only \
            --run_name "$SUFFIX" \
            "${COMMON[@]}" 2>&1 | tee "$OUT/log_eval_only_${SUFFIX}.txt"; then
        echo "    [$SUFFIX] OK" | tee -a "$OUT/eval_only.log"
    else
        echo "    [$SUFFIX] FAILED (exit $?)" | tee -a "$OUT/eval_only.log"
    fi
}

# 1. FP16 — base Llama-3.2-1B, no quantization, no checkpoint
[[ "$DRY_RUN" == "1" ]] || eval_fp

# 2. INT8 + ADC PTQ — saved ADC-aware checkpoint, ADC-mode lm-eval only
eval_config llama-3_2-1b_int8_ptq \
    --fq_w_bits 8 --fq_a_bits 8 \
    --bx 8 --bw 8 --ba 8 --k 4

# 3. INT4 + ADC PTQ — saved ADC-aware checkpoint, ADC-mode lm-eval only
eval_config llama-3_2-1b_int4_ptq \
    --fq_w_bits 4 --fq_a_bits 4 \
    --bx 4 --bw 4 --ba 8 --k 16

# 4. INT4 + ADC + LoRA — saved LoRA checkpoint, ADC-mode lm-eval only
eval_config llama-3_2-1b_int4_lora \
    --fq_w_bits 4 --fq_a_bits 4 \
    --bx 4 --bw 4 --ba 8 --k 16 \
    --lora_rank 4 --lora_alpha 8 \
    --lora_target_modules down_proj up_proj gate_proj q_proj k_proj v_proj o_proj

echo ""
echo "=== eval_only done.  Updated: $RESULTS_JSON  ==="
echo "=== Log: $OUT/eval_only.log  ==="
