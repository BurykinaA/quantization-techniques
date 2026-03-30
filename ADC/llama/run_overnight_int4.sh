#!/bin/bash
# run_overnight_int4.sh — INT4 overnight experiments (bx=bw=4)
# Usage: bash ADC/llama/run_overnight_int4.sh
# Results: ADC/llama/results/overnight_int4_YYYYMMDD.json
#
# Experiments:
#   1. baseline_int4        — clean INT4 baseline
#   2. prop_int4            — propagated calibration
#   3. center_int4          — bin-center loss λ=0.1
#   4. prop+center_int4     — propagated + bin-center
#   5. hadamard_int4        — hadamard Kronecker init
#   6. hadamard+prop_int4   — hadamard + propagated
#   7. prop_512s_int4       — propagated, 512 calibration samples
#   8. decoupled_bw8_int4   — transforms trained at bw=8, activation only at bx=4

set -u  # error on undefined variables; NO set -e (failures must not stop other runs)

# ============================================================
# Paths & output
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
DATE_TAG="$(date +%Y%m%d)"
RESULTS_JSON="$RESULTS_DIR/overnight_int4_${DATE_TAG}.json"
LOG_DIR="$RESULTS_DIR/logs_${DATE_TAG}"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

echo "============================================================"
echo "INT4 Overnight Experiments"
echo "Started: $(date)"
echo "Results JSON: $RESULTS_JSON"
echo "Logs:         $LOG_DIR"
echo "============================================================"

# ============================================================
# Common base args (INT4 hardware config)
# ============================================================
MODEL_NAME="meta-llama/Llama-3.2-1B"
OUTPUT_BASE="$SCRIPT_DIR/checkpoints/overnight_int4_${DATE_TAG}"

BASE_ARGS=(
    --model_name "$MODEL_NAME"
    --preprocess_method flat_quant
    # INT4 hardware
    --bx 4 --bw 4 --ba 8 --k 16
    --mvm_limit 256
    # FlatQuant training
    --fq_w_bits 4 --fq_a_bits 4
    --fq_epochs 30
    --fq_nsamples 128
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
        --wandb_run_name "${name}_w4a4_int4" \
        "$@" \
        > "$logfile" 2>&1

    local exit_code=$?
    if [ "$exit_code" -eq 0 ]; then
        echo "DONE:  $name  (exit 0)  $(date)"
        # Extract final PPL from log for quick summary
        local adc_ppl
        adc_ppl=$(grep -oP "(?<=Perplexity: )\d+\.\d+" "$logfile" | tail -1)
        local bypass_ppl
        bypass_ppl=$(grep -oP "(?<=WITHOUT ADC -> WIKITEXT2 Perplexity: )\d+\.\d+" "$logfile" | head -1)
        echo "       bypass PPL = ${bypass_ppl:-n/a}   ADC PPL = ${adc_ppl:-n/a}"
    else
        echo "FAIL:  $name  (exit $exit_code)  $(date)"
        echo "       Check log: $logfile"
        # Write failure record to JSON
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

    # Small pause between runs to let GPU cache clear
    sleep 10
}

# ============================================================
# EXPERIMENTS
# ============================================================

# 1. Clean INT4 baseline
run_experiment "baseline_int4"

# 2. Propagated calibration
run_experiment "prop_int4" \
    --fq_propagate_quant

# 3. Bin-center loss λ=0.1
run_experiment "center_int4" \
    --fq_lambda_center 0.1

# 4. Propagated + bin-center
run_experiment "prop+center_int4" \
    --fq_propagate_quant \
    --fq_lambda_center 0.1

# 5. Hadamard Kronecker init
run_experiment "hadamard_int4" \
    --fq_kronecker_init hadamard

# 6. Hadamard + propagated calibration
run_experiment "hadamard+prop_int4" \
    --fq_kronecker_init hadamard \
    --fq_propagate_quant

# 7. More calibration samples (512) + propagated
run_experiment "prop_512s_int4" \
    --fq_propagate_quant \
    --fq_nsamples 512

# 8. Decoupled: learn transforms at bw=8 (activation noise only), no weight quantization noise
run_experiment "decoupled_bw8_int4" \
    --bw 8 \
    --fq_w_bits 8

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

# Quick summary table from JSON
python3 -c "
import json, sys
try:
    data = json.load(open('$RESULTS_JSON'))
except Exception as e:
    print(f'Could not read results: {e}')
    sys.exit(0)
print(f'{'Run':<30}  {'bypass PPL':>10}  {'ADC PPL':>8}  {'dead%':>6}  {'status'}')
print('-' * 70)
for r in data:
    s = r.get('results', {})
    bypass = f\"{s['ppl_bypass']:.2f}\" if s.get('ppl_bypass') else 'n/a'
    adc    = f\"{s['ppl_adc']:.2f}\"    if s.get('ppl_adc')    else 'n/a'
    dead   = f\"{s['dead_rate_mean']*100:.1f}\" if s.get('dead_rate_mean') else 'n/a'
    print(f\"{r['run_name']:<30}  {bypass:>10}  {adc:>8}  {dead:>6}  {r['status']}\")
" 2>/dev/null || echo "(install python3 to see summary table)"

echo ""
echo "WandB: https://wandb.ai/odu1/llama-flat-quant-adc-ptq-blocks"
echo "Logs:  $LOG_DIR/"
