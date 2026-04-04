#!/bin/bash
# run_overnight_int4_v3.sh — INT4 v3 experiments: layer-wise alpha + selective diag
#
# New features vs v2:
#   --fq_prop_alpha_early FLOAT   : α for early layers (0..fq_prop_late_start-1)
#   --fq_prop_late_start INT      : boundary between early/late layers (default 8)
#   --fq_diag_attn                : train diag_scale for attention blocks only
#   --fq_diag_mlp                 : train diag_scale for MLP blocks only
#   (no flag = both blocks trained, backward compatible)
#
# Best v2 result: add_diag + α=0.5 → bypass=20.23, ADC=27.56
#
# Hypotheses:
#   1. Selective diag: down_proj is the hardest projection; MLP diag alone may be enough
#   2. Layer-wise α: early layers need less propagation (less ADC noise accumulation)
#   3. Combination of both: diagmlp + early025/late05 — main candidate
#
# Experiments:
#   1. repro_diagboth_alpha05      — CONTROL: reproduce best v2 result on v3 code
#   2. propalpha_early025_late05   — layer-wise α: 0.25 early / 0.5 late, no diag
#   3. propalpha_early05_late075   — layer-wise α: 0.5 early / 0.75 late, no diag
#   4. diagattn_propalpha05        — attn diag only + α=0.5
#   5. diagmlp_propalpha05         — MLP diag only + α=0.5  ← main bet
#   6. diagmlp_early025_late05     — MLP diag + early025/late05
#   7. diagattn_early025_late05    — attn diag + early025/late05
#   8. diagboth_early025_late05    — both diag + early025/late05  ← best candidate
#
# Usage: bash ADC/llama/run_overnight_int4_v3.sh
# Results: ADC/llama/results/overnight_int4_v3_YYYYMMDD.json

set -u  # error on undefined vars; NO set -e (failures must not stop other runs)

# ============================================================
# Paths & output
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
DATE_TAG="$(date +%Y%m%d)"
RESULTS_JSON="$RESULTS_DIR/overnight_int4_v3_${DATE_TAG}.json"
LOG_DIR="$RESULTS_DIR/logs_v3_${DATE_TAG}"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

echo "============================================================"
echo "INT4 v3 Overnight Experiments (layer-wise alpha + selective diag)"
echo "Started: $(date)"
echo "Results JSON: $RESULTS_JSON"
echo "Logs:         $LOG_DIR"
echo "============================================================"

# ============================================================
# Common base args (INT4 hardware config, 1024 samples)
# ============================================================
MODEL_NAME="meta-llama/Llama-3.2-1B"
OUTPUT_BASE="$SCRIPT_DIR/checkpoints/overnight_int4_v3_${DATE_TAG}"

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
        --wandb_run_name "${name}_w4a4_int4_v3" \
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
# EXPERIMENTS (all bx=4, bw=4, ba=8, k=16, 1024 samples)
# ============================================================

# 1. CONTROL: reproduce best v2 result (add_diag + α=0.5) on v3 code
#    Expected: bypass≈20.23, ADC≈27.56.
run_experiment "repro_diagboth_alpha05" \
    --fq_add_diag \
    --fq_propagate_quant \
    --fq_prop_alpha 0.5

# (2, 3 already ran correctly — no diag, layer-wise α only — skipped)

# 4. Attention diag only + α=0.5
#    Isolate: does attn diag_scale contribute to the add_diag gain?
run_experiment "diagattn_propalpha05" \
    --fq_add_diag \
    --fq_diag_attn \
    --fq_propagate_quant \
    --fq_prop_alpha 0.5

# 5. MLP diag only + α=0.5  ← main bet
#    down_proj has highest reconstruction_rel error; MLP diag should matter most
run_experiment "diagmlp_propalpha05" \
    --fq_add_diag \
    --fq_diag_mlp \
    --fq_propagate_quant \
    --fq_prop_alpha 0.5

# 6. MLP diag + layer-wise α (early=0.25, late=0.5)
#    Combination: best selective diag + less early-layer overfitting
run_experiment "diagmlp_early025_late05" \
    --fq_add_diag \
    --fq_diag_mlp \
    --fq_propagate_quant \
    --fq_prop_alpha 0.5 \
    --fq_prop_alpha_early 0.25 \
    --fq_prop_late_start 8

# 7. Attention diag + layer-wise α (early=0.25, late=0.5)
run_experiment "diagattn_early025_late05" \
    --fq_add_diag \
    --fq_diag_attn \
    --fq_propagate_quant \
    --fq_prop_alpha 0.5 \
    --fq_prop_alpha_early 0.25 \
    --fq_prop_late_start 8

# 8. Both diag + layer-wise α (early=0.25, late=0.5)  ← best candidate
#    Combines two orthogonal improvements: selective diag init + layer-wise regularization
run_experiment "diagboth_early025_late05" \
    --fq_add_diag \
    --fq_propagate_quant \
    --fq_prop_alpha 0.5 \
    --fq_prop_alpha_early 0.25 \
    --fq_prop_late_start 8

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
