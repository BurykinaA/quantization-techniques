#!/usr/bin/env bash
# run_paper_comparison.sh — multi-model paper comparison table
# 3 models × 4 configs: FP16, INT8 PTQ, INT4+ADC PTQ, INT4+ADC+LoRA
# Metrics: WikiText2 PPL, C4 PPL, HellaSwag (0-shot), MMLU (5-shot), WinoGrande (5-shot)
#
# Usage:
#   bash run_paper_comparison.sh
#   OUTPUT_DIR=/path/to/out  bash run_paper_comparison.sh
#
#   # Re-run only specific configs (comma-separated suffixes):
#   ONLY="llama-3_2-3b_int8_ptq,llama-3_2-3b_int4_ptq" bash run_paper_comparison.sh
#
#   # Skip configs already present in results.json (resume after partial run):
#   SKIP_COMPLETED=1 bash run_paper_comparison.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNS="$SCRIPT_DIR/runs/llama_smooth_quant_adc_ptq.py"
OUT="${OUTPUT_DIR:-$SCRIPT_DIR/outputs/paper_comparison}"
mkdir -p "$OUT"

# Reduce CUDA fragmentation — required for 3B/8B Stage B propagation.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ─── Run-filter helpers ──────────────────────────────────────────────────────

ONLY="${ONLY:-}"           # comma-separated suffix list; empty = run everything
SKIP_COMPLETED="${SKIP_COMPLETED:-0}"

# is_in_only "$SUFFIX": returns 0 if SUFFIX is in $ONLY (or $ONLY is empty).
is_in_only() {
    [[ -z "$ONLY" ]] && return 0
    local s="$1"
    IFS=',' read -ra _arr <<< "$ONLY"
    for x in "${_arr[@]}"; do [[ "$x" == "$s" ]] && return 0; done
    return 1
}

# is_completed "$SUFFIX": returns 0 if results.json contains an entry for SUFFIX.
is_completed() {
    [[ "$SKIP_COMPLETED" != "1" ]] && return 1
    [[ ! -f "$OUT/results.json" ]] && return 1
    python -c "import json,sys; d=json.load(open('$OUT/results.json')); \
sys.exit(0 if any(e.get('run_name')=='$1' and e.get('status')=='success' for e in d) else 1)" 2>/dev/null
}

# ─── Helper ──────────────────────────────────────────────────────────────────

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

# ─── Per-model FlatQuant batch size (Stage B propagation needs ~2× memory) ───

fq_cali_bsz_for() {
    case "$1" in
        *Llama-3.2-1B*) echo 16 ;;
        *Llama-3.2-3B*) echo  4 ;;   # was 16, OOM in Stage B
        *Llama-3.1-8B*) echo  2 ;;   # was 16, OOM even in Stage A
        *)              echo  4 ;;
    esac
}

# ─── Common eval args ────────────────────────────────────────────────────────

COMMON_EVAL=(
    --eval_datasets wikitext2 c4
    --run_no_adc_eval                # also measure bypass (INT-only, no ADC) PPL
    --run_lm_eval
    --run_lm_eval_bypass             # second lm-eval pass in bypass mode for INT-vs-ADC comparison
    --disable_visualizations
)

# ─── Best FlatQuant settings (from v8/v9) ────────────────────────────────────

FQ_INT4=(
    --preprocess_method flat_quant
    --fq_nsamples 1024
    --fq_epochs 30 --fq_stage_b_epochs 10
    --fq_w_bits 4 --fq_a_bits 4
    --fq_lwc --fq_lac --fq_add_diag
    --fq_diag_mlp
    --fq_stage_b_prop_alpha 0.5 --fq_stage_b_diag_attn
    --mvm_limit 256
)

FQ_INT8=(
    --preprocess_method flat_quant
    --fq_nsamples 1024
    --fq_epochs 30 --fq_stage_b_epochs 10
    --fq_w_bits 8 --fq_a_bits 8
    --fq_lwc --fq_lac --fq_add_diag
    --fq_diag_mlp
    --fq_stage_b_prop_alpha 0.5 --fq_stage_b_diag_attn
    --mvm_limit 256
)

# ─── Best LoRA settings (from v8/v9) ─────────────────────────────────────────

LORA_ARGS=(
    --lora_rank 4 --lora_alpha 8
    --lora_epochs 5 --lora_lr 1e-4
    --lora_loss ce_kl --lora_kl_weight 0.5 --lora_kl_temperature 2.0
    --lora_nsamples 1024
    --lora_target_modules down_proj up_proj gate_proj q_proj k_proj v_proj o_proj
)

# ─── Per-model loop ───────────────────────────────────────────────────────────

MODELS=(
    "meta-llama/Llama-3.2-1B"
    "meta-llama/Llama-3.2-3B"
    "meta-llama/Llama-3.1-8B"
)

for MODEL_ID in "${MODELS[@]}"; do
    SHORT=$(echo "$MODEL_ID" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]' | tr '.' '_')
    BSZ=$(fq_cali_bsz_for "$MODEL_ID")
    BSZ_ARG=(--fq_cali_bsz "$BSZ")

    # 1. FP16
    run_config "${SHORT}_fp" \
        --model_name "$MODEL_ID" \
        --fp_only_eval \
        "${COMMON_EVAL[@]}"

    # 2. INT8 PTQ (no LoRA)
    run_config "${SHORT}_int8_ptq" \
        --model_name "$MODEL_ID" \
        --bx 8 --bw 8 --ba 8 --k 4 \
        --lora_rank 0 \
        "${FQ_INT8[@]}" "${BSZ_ARG[@]}" \
        "${COMMON_EVAL[@]}"

    # 3. INT4 + ADC PTQ (no LoRA)
    run_config "${SHORT}_int4_ptq" \
        --model_name "$MODEL_ID" \
        --bx 4 --bw 4 --ba 8 --k 16 \
        --lora_rank 0 \
        "${FQ_INT4[@]}" "${BSZ_ARG[@]}" \
        "${COMMON_EVAL[@]}"

    # 4. INT4 + ADC + LoRA
    run_config "${SHORT}_int4_lora" \
        --model_name "$MODEL_ID" \
        --bx 4 --bw 4 --ba 8 --k 16 \
        "${FQ_INT4[@]}" "${BSZ_ARG[@]}" \
        "${LORA_ARGS[@]}" \
        "${COMMON_EVAL[@]}"
done

echo ""
echo "=== All done. Results: $OUT/results.json ==="
echo "=== Run log:           $OUT/run.log ==="
