#!/bin/bash
# run_overnight_int4_v9.sh — INT4 v9: data efficiency sweep + KL loss sweep
#
# All experiments use the best config from v8: r4_all_ce_kl (rank=4, all 7 projs, CE+KL).
# PTQ is fixed at fq_nsamples=1024, mvm_limit=256, staged FQ.
#
# Sweep 1 — Data efficiency: how much data does LoRA need?
#   lora_nsamples ∈ {64, 128, 256, 512, 1024}
#
# Sweep 2 — KL loss: optimal kl_weight × temperature?
#   6 selected (kl_weight, temperature) points
#
# Usage: bash ADC/llama/run_overnight_int4_v9.sh
# Results: ADC/llama/results/overnight_int4_v9_YYYYMMDD.json

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
DATE_TAG="$(date +%Y%m%d)"
RESULTS_JSON="$RESULTS_DIR/overnight_int4_v9_${DATE_TAG}.json"
LOG_DIR="$RESULTS_DIR/logs_v9_${DATE_TAG}"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "============================================================"
echo "INT4 v9: Data efficiency + KL sweep"
echo "Started: $(date)"
echo "Results JSON: $RESULTS_JSON"
echo "Logs:         $LOG_DIR"
echo "============================================================"

MODEL_NAME="meta-llama/Llama-3.2-1B"
OUTPUT_BASE="$SCRIPT_DIR/checkpoints/overnight_int4_v9_${DATE_TAG}"

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
    --seed 42
)

# Best LoRA config from v8 (r4_all_ce_kl)
LORA_BASE=(
    --lora_mode residual
    --lora_epochs 5
    --lora_lr 1e-4
    --lora_rank 4 --lora_alpha 8.0
    --lora_target_modules down_proj up_proj gate_proj q_proj k_proj v_proj o_proj
    --lora_loss ce_kl
    --lora_kl_weight 0.5
    --lora_kl_temperature 2.0
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
        --wandb_run_name "${name}_w4a4_int4_v9" \
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
record = {'run_name': '$name', 'status': 'failed', 'exit_code': $exit_code,
          'timestamp': datetime.datetime.now().isoformat()}
data = [r for r in data if r.get('run_name') != '$name']
data.append(record)
os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
json.dump(data, open(p, 'w'), indent=2)
" 2>/dev/null || true
    fi

    sleep 10
}

# ============================================================
# Sweep 1: Data efficiency — lora_nsamples ∈ {64, 128, 256, 512, 1024}
# ============================================================
for N in 64 128 256 512 1024; do
    run_experiment "r4_all_ce_kl_n${N}" \
        "${LORA_BASE[@]}" \
        --lora_nsamples "$N"
done

# ============================================================
# Sweep 2: KL loss — 6 (kl_weight, temperature) points
# ============================================================
run_experiment "r4_kl025_t1" \
    "${LORA_BASE[@]}" \
    --lora_kl_weight 0.25 --lora_kl_temperature 1.0

run_experiment "r4_kl025_t2" \
    "${LORA_BASE[@]}" \
    --lora_kl_weight 0.25 --lora_kl_temperature 2.0

run_experiment "r4_kl05_t1" \
    "${LORA_BASE[@]}" \
    --lora_kl_weight 0.5 --lora_kl_temperature 1.0

run_experiment "r4_kl05_t2" \
    "${LORA_BASE[@]}" \
    --lora_kl_weight 0.5 --lora_kl_temperature 2.0

run_experiment "r4_kl10_t2" \
    "${LORA_BASE[@]}" \
    --lora_kl_weight 1.0 --lora_kl_temperature 2.0

run_experiment "r4_kl20_t4" \
    "${LORA_BASE[@]}" \
    --lora_kl_weight 2.0 --lora_kl_temperature 4.0

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
print(f\"{'Run':<28}  {'wiki-bypass':>11}  {'wiki-adc':>8}  {'c4-bypass':>9}  {'c4-adc':>6}  status\")
print('-' * 78)
for r in data:
    s = r.get('results', {})
    wb = f\"{s['ppl_bypass_wikitext2']:.2f}\" if s.get('ppl_bypass_wikitext2') else 'n/a'
    wa = f\"{s['ppl_adc_wikitext2']:.2f}\"    if s.get('ppl_adc_wikitext2')    else 'n/a'
    cb = f\"{s['ppl_bypass_c4']:.2f}\"         if s.get('ppl_bypass_c4')         else 'n/a'
    ca = f\"{s['ppl_adc_c4']:.2f}\"            if s.get('ppl_adc_c4')            else 'n/a'
    print(f\"{r['run_name']:<28}  {wb:>11}  {wa:>8}  {cb:>9}  {ca:>6}  {r['status']}\")
" 2>/dev/null || echo "(python3 not found — check \$RESULTS_JSON manually)"

echo ""
echo "WandB: https://wandb.ai/odu1/llama-flat-quant-adc-ptq-blocks"
echo "Logs:  $LOG_DIR/"
