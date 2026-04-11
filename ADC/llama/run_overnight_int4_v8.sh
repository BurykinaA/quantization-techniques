#!/bin/bash
# run_overnight_int4_v8.sh — INT4 v8: best-3 LoRA configs, wikitext2 + C4 eval
#
# v7 identified the top configs by wikitext2 ADC PPL:
#   1. r4_all_ce_kl     — all 7 projs, CE+KL, rank=4 → ADC=14.03
#   2. r4_down_o_ce_kl  — down+o,      CE+KL, rank=4 → ADC=14.33
#   3. rank8_down_o     — down+o,      CE,    rank=8  → ADC=15.24
#
# v8 re-runs these 3 + a PTQ control with BOTH wikitext2 and C4 evaluation
# to check whether LoRA correction generalises to out-of-domain web text.
#
# Each run: single seed=42, PTQ + LoRA trained from scratch.
#
# Usage: bash ADC/llama/run_overnight_int4_v8.sh
# Results: ADC/llama/results/overnight_int4_v8_YYYYMMDD.json

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
DATE_TAG="$(date +%Y%m%d)"
RESULTS_JSON="$RESULTS_DIR/overnight_int4_v8_${DATE_TAG}.json"
LOG_DIR="$RESULTS_DIR/logs_v8_${DATE_TAG}"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

echo "============================================================"
echo "INT4 v8 Overnight Experiments (best-3 LoRA + C4 eval)"
echo "Started: $(date)"
echo "Results JSON: $RESULTS_JSON"
echo "Logs:         $LOG_DIR"
echo "============================================================"

MODEL_NAME="meta-llama/Llama-3.2-1B"
OUTPUT_BASE="$SCRIPT_DIR/checkpoints/overnight_int4_v8_${DATE_TAG}"

BASE_ARGS=(
    --model_name "$MODEL_NAME"
    --preprocess_method flat_quant
    --bx 4 --bw 4 --ba 8 --k 16
    --mvm_limit 256
    --fq_w_bits 4 --fq_a_bits 4
    --fq_epochs 30
    --fq_nsamples 1024
    --fq_cali_bsz 16
    --fq_lr 0.005
    --fq_lwc --fq_lac
    --fq_add_diag
    --fq_diag_mlp
    --fq_stage_b_epochs 10
    --fq_stage_b_prop_alpha 0.5
    --fq_stage_b_diag_attn
    --fq_save_transforms
    --calibration_method percentile
    --calibration_max_length 512
    --eval_datasets wikitext2 c4
    --max_eval_samples 1000
    --run_no_adc_eval
    --results_json_path "$RESULTS_JSON"
    --disable_visualizations
    --wandb_project llama-flat-quant-adc-ptq-blocks
)

LORA_BASE=(
    --lora_mode residual
    --lora_epochs 5
    --lora_lr 1e-4
)

run_experiment() {
    local name="$1"
    shift
    local logfile="$LOG_DIR/${name}.log"
    local out_dir="${OUTPUT_BASE}/${name}"

    echo ""
    echo "========================================"
    echo "START: $name"
    echo "Time:  $(date)"
    echo "Log:   $logfile"
    echo "========================================"

    python "$SCRIPT_DIR/runs/llama_smooth_quant_adc_ptq.py" \
        "${BASE_ARGS[@]}" \
        --output_dir "$out_dir" \
        --wandb_run_name "${name}_w4a4_int4_v8" \
        "$@" \
        > "$logfile" 2>&1

    local exit_code=$?
    if [ "$exit_code" -eq 0 ]; then
        echo "DONE:  $name  (exit 0)  $(date)"
        local wiki_adc wiki_bypass c4_adc c4_bypass
        wiki_adc=$(grep -oP "(?<=WIKITEXT2 Perplexity: )\d+\.\d+" "$logfile" | tail -1)
        wiki_bypass=$(grep -oP "(?<=WITHOUT ADC -> WIKITEXT2 Perplexity: )\d+\.\d+" "$logfile" | head -1)
        c4_adc=$(grep -oP "(?<=C4 Perplexity: )\d+\.\d+" "$logfile" | tail -1)
        c4_bypass=$(grep -oP "(?<=WITHOUT ADC -> C4 Perplexity: )\d+\.\d+" "$logfile" | head -1)
        echo "       wiki: bypass=${wiki_bypass:-n/a}  ADC=${wiki_adc:-n/a}"
        echo "       c4:   bypass=${c4_bypass:-n/a}    ADC=${c4_adc:-n/a}"
    else
        echo "FAIL:  $name  (exit $exit_code)  $(date)"
        echo "       Check log: $logfile"
        python3 -c "
import json, os, datetime
p = '$RESULTS_JSON'
data = []
try:
    data = json.load(open(p))
except (FileNotFoundError, json.JSONDecodeError):
    pass
data.append({'run_name': '$name', 'status': 'failed', 'exit_code': $exit_code,
             'timestamp': datetime.datetime.now().isoformat()})
os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
json.dump(data, open(p, 'w'), indent=2)
" 2>/dev/null || true
    fi

    sleep 10
}

# ============================================================
# CONTROL: PTQ baseline without LoRA
# ============================================================
run_experiment "base_no_lora" \
    --seed 42

# ============================================================
# 1. Best (v7): all 7 projections, CE+KL, rank=4
# ============================================================
run_experiment "r4_all_ce_kl" \
    --seed 42 \
    "${LORA_BASE[@]}" \
    --lora_rank 4 --lora_alpha 8.0 \
    --lora_target_modules down_proj up_proj gate_proj q_proj k_proj v_proj o_proj \
    --lora_loss ce_kl \
    --lora_kl_weight 0.5 \
    --lora_kl_temperature 2.0

# ============================================================
# 2. Second best (v7): down+o, CE+KL, rank=4
# ============================================================
run_experiment "r4_down_o_ce_kl" \
    --seed 42 \
    "${LORA_BASE[@]}" \
    --lora_rank 4 --lora_alpha 8.0 \
    --lora_target_modules down_proj o_proj \
    --lora_loss ce_kl \
    --lora_kl_weight 0.5 \
    --lora_kl_temperature 2.0

# ============================================================
# 3. Third best (v7): down+o, CE, rank=8
# ============================================================
run_experiment "rank8_down_o" \
    --seed 42 \
    "${LORA_BASE[@]}" \
    --lora_rank 8 --lora_alpha 16.0 \
    --lora_target_modules down_proj o_proj \
    --lora_loss ce

# ============================================================
echo ""
echo "============================================================"
echo "ALL EXPERIMENTS DONE"
echo "Finished: $(date)"
echo "============================================================"
echo ""
echo "Results JSON: $RESULTS_JSON"
echo ""

python3 -c "
import json, sys
try:
    data = json.load(open('$RESULTS_JSON'))
except Exception as e:
    print(f'Could not read results: {e}')
    sys.exit(0)
print(f\"{'Run':<25}  {'wiki-bypass':>11}  {'wiki-adc':>8}  {'c4-bypass':>9}  {'c4-adc':>6}  status\")
print('-' * 75)
for r in data:
    s = r.get('results', {})
    wb = f\"{s['ppl_bypass_wikitext2']:.2f}\" if s.get('ppl_bypass_wikitext2') else 'n/a'
    wa = f\"{s['ppl_adc_wikitext2']:.2f}\"    if s.get('ppl_adc_wikitext2')    else 'n/a'
    cb = f\"{s['ppl_bypass_c4']:.2f}\"         if s.get('ppl_bypass_c4')         else 'n/a'
    ca = f\"{s['ppl_adc_c4']:.2f}\"            if s.get('ppl_adc_c4')            else 'n/a'
    print(f\"{r['run_name']:<25}  {wb:>11}  {wa:>8}  {cb:>9}  {ca:>6}  {r['status']}\")
" 2>/dev/null || echo "(python3 not found — check \$RESULTS_JSON manually)"

echo ""
echo "WandB: https://wandb.ai/odu1/llama-flat-quant-adc-ptq-blocks"
echo "Logs:  $LOG_DIR/"
