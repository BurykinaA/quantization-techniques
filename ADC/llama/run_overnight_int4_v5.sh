#!/bin/bash
# run_overnight_int4_v5.sh — INT4 v5: stochastic propagation
#
# Motivation: deterministic α=0.5 mixes FP and quant losses every batch.
# Stochastic propagation (QDrop-style) randomly varies α per batch:
#   - Bernoulli: each batch randomly uses fp_inp OR quant_inp (single forward)
#   - Beta(β,β): each batch samples α ~ Beta(β,β), dual forward
#     β=1 → uniform[0,1]; β=2 → concentrated near 0.5
#
# Reference baselines (from v3/v4):
#   propalpha_05:              bypass=24.24, ADC=31.22
#   staged_mlpdiag_then_attn:  bypass=18.50, ADC=27.60  ← current best
#
# Experiments:
#   1. stoch_bern_propalpha05   — Bernoulli vs deterministic α=0.5 (flat, no diag)
#   2. stoch_beta2_propalpha05  — Beta(2,2) vs deterministic α=0.5 (flat, no diag)
#   3. staged_stoch_bern        — best staged config + Bernoulli in Stage B
#   4. staged_stoch_beta2       — best staged config + Beta(2,2) in Stage B
#
# Usage: bash ADC/llama/run_overnight_int4_v5.sh
# Results: ADC/llama/results/overnight_int4_v5_YYYYMMDD.json

set -u  # NO set -e — failures must not stop other runs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
DATE_TAG="$(date +%Y%m%d)"
RESULTS_JSON="$RESULTS_DIR/overnight_int4_v5_${DATE_TAG}.json"
LOG_DIR="$RESULTS_DIR/logs_v5_${DATE_TAG}"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

echo "============================================================"
echo "INT4 v5 Overnight Experiments (stochastic propagation)"
echo "Started: $(date)"
echo "Results JSON: $RESULTS_JSON"
echo "Logs:         $LOG_DIR"
echo "============================================================"

MODEL_NAME="meta-llama/Llama-3.2-1B"
OUTPUT_BASE="$SCRIPT_DIR/checkpoints/overnight_int4_v5_${DATE_TAG}"

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
    --fq_no_diag
    --fq_lwc --fq_lac
    --fq_save_transforms
    --calibration_method percentile
    --eval_datasets wikitext2
    --run_no_adc_eval
    --results_json_path "$RESULTS_JSON"
    --disable_visualizations
    --wandb_project llama-flat-quant-adc-ptq-blocks
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
        --wandb_run_name "${name}_w4a4_int4_v5" \
        "$@" \
        > "$logfile" 2>&1

    local exit_code=$?
    if [ "$exit_code" -eq 0 ]; then
        echo "DONE:  $name  (exit 0)  $(date)"
        local adc_ppl bypass_ppl
        adc_ppl=$(grep -oP "(?<=Perplexity: )\d+\.\d+" "$logfile" | tail -1)
        bypass_ppl=$(grep -oP "(?<=WITHOUT ADC -> WIKITEXT2 Perplexity: )\d+\.\d+" "$logfile" | head -1)
        echo "       bypass PPL = ${bypass_ppl:-n/a}   ADC PPL = ${adc_ppl:-n/a}"
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
# EXPERIMENTS
# ============================================================

# 1. Bernoulli stochastic vs deterministic α=0.5 (flat, no diag)
#    Direct comparison: same config as propalpha_05 but random fp/quant each batch
run_experiment "stoch_bern_propalpha05" \
    --fq_propagate_quant \
    --fq_prop_alpha 0.5 \
    --fq_stochastic_prop \
    --fq_stochastic_mode bernoulli

# 2. Beta(2,2) stochastic vs deterministic α=0.5 (flat, no diag)
#    Alpha sampled near 0.5 each batch; dual forward; smoother than Bernoulli
run_experiment "stoch_beta2_propalpha05" \
    --fq_propagate_quant \
    --fq_prop_alpha 0.5 \
    --fq_stochastic_prop \
    --fq_stochastic_mode beta \
    --fq_beta_param 2.0

# 3. Staged best + Bernoulli in Stage B
#    Stage A: MLP diag, no prop (clean transforms)
#    Stage B: attn diag + Bernoulli stochastic prop, 10ep
run_experiment "staged_stoch_bern" \
    --fq_add_diag \
    --fq_diag_mlp \
    --fq_stage_b_epochs 10 \
    --fq_stage_b_prop_alpha 0.5 \
    --fq_stage_b_diag_attn \
    --fq_stochastic_prop \
    --fq_stochastic_mode bernoulli

# 4. Staged best + Beta(2,2) in Stage B
#    Stage A: MLP diag, no prop (clean transforms)
#    Stage B: attn diag + Beta(2,2) stochastic prop, 10ep
run_experiment "staged_stoch_beta2" \
    --fq_add_diag \
    --fq_diag_mlp \
    --fq_stage_b_epochs 10 \
    --fq_stage_b_prop_alpha 0.5 \
    --fq_stage_b_diag_attn \
    --fq_stochastic_prop \
    --fq_stochastic_mode beta \
    --fq_beta_param 2.0

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
print(f\"{'Run':<42}  {'bypass PPL':>10}  {'ADC PPL':>8}  {'dead%':>6}  status\")
print('-' * 82)
for r in data:
    s = r.get('results', {})
    bypass = f\"{s['ppl_bypass']:.2f}\" if s.get('ppl_bypass') else 'n/a'
    adc    = f\"{s['ppl_adc']:.2f}\"    if s.get('ppl_adc')    else 'n/a'
    dead   = f\"{s['dead_rate_mean']*100:.1f}\" if s.get('dead_rate_mean') else 'n/a'
    print(f\"{r['run_name']:<42}  {bypass:>10}  {adc:>8}  {dead:>6}  {r['status']}\")
" 2>/dev/null || echo "(python3 not found — check $RESULTS_JSON manually)"

echo ""
echo "WandB: https://wandb.ai/odu1/llama-flat-quant-adc-ptq-blocks"
echo "Logs:  $LOG_DIR/"
