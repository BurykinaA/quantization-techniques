#!/bin/bash
# run_overnight_int4_v2.sh — INT4 v2 experiments: partial propagation, 2-stage, bounded LET
#
# New features vs v1:
#   --fq_prop_alpha FLOAT   : 0.5 = dual-forward (1-α)*MSE(FP_inp) + α*MSE(quant_inp)
#   --fq_stage_b_epochs INT : 2-stage: stage A (no prop) → stage B (prop fine-tune)
#   --fq_stage_b_prop_alpha : alpha for stage B
#   --fq_add_diag           : bounded LET (diagonal scaling, bounded [1e-4, 10])
#
# Experiments:
#   1. baseline_1024          — reference: baseline INT4, 1024 samples
#   2. prop_full_1024         — full propagation (α=1.0), 1024 samples
#   3. propalpha_075_1024     — partial prop α=0.75, 1024 samples
#   4. propalpha_05_1024      — partial prop α=0.5,  1024 samples  ← main bet
#   5. propalpha_025_1024     — partial prop α=0.25, 1024 samples
#   6. hadamard_propalpha05   — Hadamard init + α=0.5, 1024 samples
#   7. 2stage_sb10_pa05       — stage A: 30ep no prop → stage B: 10ep α=0.5
#   8. add_diag_propalpha05   — bounded LET (add_diag) + α=0.5, 1024 samples
#
# Usage: bash ADC/llama/run_overnight_int4_v2.sh
# Results: ADC/llama/results/overnight_int4_v2_YYYYMMDD.json

set -u  # error on undefined vars; NO set -e (failures must not stop other runs)

# ============================================================
# Paths & output
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
DATE_TAG="$(date +%Y%m%d)"
RESULTS_JSON="$RESULTS_DIR/overnight_int4_v2_${DATE_TAG}.json"
LOG_DIR="$RESULTS_DIR/logs_v2_${DATE_TAG}"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

echo "============================================================"
echo "INT4 v2 Overnight Experiments"
echo "Started: $(date)"
echo "Results JSON: $RESULTS_JSON"
echo "Logs:         $LOG_DIR"
echo "============================================================"

# ============================================================
# Common base args (INT4 hardware config)
# ============================================================
MODEL_NAME="meta-llama/Llama-3.2-1B"
OUTPUT_BASE="$SCRIPT_DIR/checkpoints/overnight_int4_v2_${DATE_TAG}"

BASE_ARGS=(
    --model_name "$MODEL_NAME"
    --preprocess_method flat_quant
    # INT4 hardware
    --bx 4 --bw 4 --ba 8 --k 16
    --mvm_limit 256
    # FlatQuant training
    --fq_w_bits 4 --fq_a_bits 4
    --fq_epochs 30
    --fq_nsamples 1024
    --fq_cali_bsz 16
    --fq_lr 0.005
    --fq_no_diag
    --fq_lwc --fq_lac
    --fq_save_transforms
    # Calibration
    --calibration_method percentile
    # Eval
    --eval_datasets wikitext2
    --run_no_adc_eval
    # Output / logging
    --results_json_path "$RESULTS_JSON"
    --disable_visualizations
    --wandb_project llama-flat-quant-adc-ptq-blocks
)

# ============================================================
# run_experiment <name> [extra args...]
# ============================================================
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
        --wandb_run_name "${name}_w4a4_int4_v2" \
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
# EXPERIMENTS (all bx=4, bw=4, 1024 samples unless noted)
# ============================================================

# 1. Baseline reference (1024 samples, no prop)
run_experiment "baseline_1024"

# 2. Full propagation α=1.0 (comparison with v1 prop_512s result)
run_experiment "prop_full_1024" \
    --fq_propagate_quant

# 3. Partial propagation α=0.75 — mostly quant, small FP anchor
run_experiment "propalpha_075_1024" \
    --fq_propagate_quant \
    --fq_prop_alpha 0.75

# 4. Partial propagation α=0.5 — equal mix (main bet)
run_experiment "propalpha_05_1024" \
    --fq_propagate_quant \
    --fq_prop_alpha 0.5

# 5. Partial propagation α=0.25 — mostly FP, light quant regularization
run_experiment "propalpha_025_1024" \
    --fq_propagate_quant \
    --fq_prop_alpha 0.25

# 6. Hadamard init + partial propagation α=0.5
run_experiment "hadamard_propalpha05_1024" \
    --fq_kronecker_init hadamard \
    --fq_propagate_quant \
    --fq_prop_alpha 0.5

# 7. Two-stage: stage A = 30ep no prop → stage B = 10ep α=0.5
#    Stage B uses lr=0.005*0.1=0.0005 (applied automatically in code)
run_experiment "2stage_sb10_pa05" \
    --fq_stage_b_epochs 10 \
    --fq_stage_b_prop_alpha 0.5

# 8. Bounded LET (diagonal scaling) + partial prop α=0.5
#    diag_scale is bounded [1e-4, 10] — safer than old unbounded add_diag
run_experiment "add_diag_propalpha05_1024" \
    --fq_add_diag \
    --fq_propagate_quant \
    --fq_prop_alpha 0.5

# ============================================================
# Summary
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
print(f\"{'Run':<35}  {'bypass PPL':>10}  {'ADC PPL':>8}  {'dead%':>6}  status\")
print('-' * 75)
for r in data:
    s = r.get('results', {})
    bypass = f\"{s['ppl_bypass']:.2f}\" if s.get('ppl_bypass') else 'n/a'
    adc    = f\"{s['ppl_adc']:.2f}\"    if s.get('ppl_adc')    else 'n/a'
    dead   = f\"{s['dead_rate_mean']*100:.1f}\" if s.get('dead_rate_mean') else 'n/a'
    print(f\"{r['run_name']:<35}  {bypass:>10}  {adc:>8}  {dead:>6}  {r['status']}\")
" 2>/dev/null || echo "(python3 not found — check $RESULTS_JSON manually)"

echo ""
echo "WandB: https://wandb.ai/odu1/llama-flat-quant-adc-ptq-blocks"
echo "Logs:  $LOG_DIR/"
