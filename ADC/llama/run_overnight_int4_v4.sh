#!/bin/bash
# run_overnight_int4_v4.sh — INT4 v4: MLP diag split + staged selective diag
#
# v3 showed: MLP diag improves bypass, attn diag closes the ADC gap.
# They do different things — can we use that?
#
# New features vs v3:
#   --fq_diag_mlp_up    : train only up_gate_trans.diag_scale inside MLP
#   --fq_diag_mlp_down  : train only down_trans.diag_scale inside MLP
#   --fq_stage_b_diag_attn : Stage B trains only attn diag (overrides Stage A)
#   --fq_stage_b_diag_mlp  : Stage B trains only MLP diag (overrides Stage A)
#
# Experiments:
#   1. diag_up_propalpha05      — up_gate_trans diag only + α=0.5
#   2. diag_down_propalpha05    — down_trans diag only + α=0.5
#   3. staged_mlpdiag_then_attn — Stage A: MLP diag, no prop
#                                  Stage B: attn diag only + α=0.5, 10ep
#
# Usage: bash ADC/llama/run_overnight_int4_v4.sh
# Results: ADC/llama/results/overnight_int4_v4_YYYYMMDD.json

set -u  # NO set -e — failures must not stop other runs

# ============================================================
# Paths & output
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
DATE_TAG="$(date +%Y%m%d)"
RESULTS_JSON="$RESULTS_DIR/overnight_int4_v4_${DATE_TAG}.json"
LOG_DIR="$RESULTS_DIR/logs_v4_${DATE_TAG}"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

echo "============================================================"
echo "INT4 v4 Overnight Experiments (MLP diag split + staged diag)"
echo "Started: $(date)"
echo "Results JSON: $RESULTS_JSON"
echo "Logs:         $LOG_DIR"
echo "============================================================"

# ============================================================
# Common base args (INT4 hardware, 1024 samples)
# ============================================================
MODEL_NAME="meta-llama/Llama-3.2-1B"
OUTPUT_BASE="$SCRIPT_DIR/checkpoints/overnight_int4_v4_${DATE_TAG}"

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
        --wandb_run_name "${name}_w4a4_int4_v4" \
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

# 1. up_gate_trans diag only + α=0.5
#    Hypothesis: up_gate_trans drives the bypass improvement in MLP diag
run_experiment "diag_up_propalpha05" \
    --fq_add_diag \
    --fq_diag_mlp \
    --fq_diag_mlp_up \
    --fq_propagate_quant \
    --fq_prop_alpha 0.5

# 2. down_trans diag only + α=0.5
#    Hypothesis: down_trans (feeds down_proj, closest to ADC output) drives ADC improvement
run_experiment "diag_down_propalpha05" \
    --fq_add_diag \
    --fq_diag_mlp \
    --fq_diag_mlp_down \
    --fq_propagate_quant \
    --fq_prop_alpha 0.5

# 3. Staged: Stage A = MLP diag, no prop → Stage B = attn diag only + α=0.5
#    Stage A: learn good MLP-diag transforms with clean FP inputs (no propagation)
#    Stage B: freeze MLP perspective, add attn robustness via diag + propagation
#    Note: --fq_propagate_quant is NOT set for Stage A; Stage B implicitly uses prop=True
run_experiment "staged_mlpdiag_then_attn" \
    --fq_add_diag \
    --fq_diag_mlp \
    --fq_stage_b_epochs 10 \
    --fq_stage_b_prop_alpha 0.5 \
    --fq_stage_b_diag_attn

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
print(f\"{'Run':<40}  {'bypass PPL':>10}  {'ADC PPL':>8}  {'dead%':>6}  status\")
print('-' * 80)
for r in data:
    s = r.get('results', {})
    bypass = f\"{s['ppl_bypass']:.2f}\" if s.get('ppl_bypass') else 'n/a'
    adc    = f\"{s['ppl_adc']:.2f}\"    if s.get('ppl_adc')    else 'n/a'
    dead   = f\"{s['dead_rate_mean']*100:.1f}\" if s.get('dead_rate_mean') else 'n/a'
    print(f\"{r['run_name']:<40}  {bypass:>10}  {adc:>8}  {dead:>6}  {r['status']}\")
" 2>/dev/null || echo "(python3 not found — check $RESULTS_JSON manually)"

echo ""
echo "WandB: https://wandb.ai/odu1/llama-flat-quant-adc-ptq-blocks"
echo "Logs:  $LOG_DIR/"
