#!/bin/bash
# run_overnight_int4_v7.sh — INT4 v7: ADC-LoRA ablation study
#
# v6 showed residual post-ADC LoRA breaks the PTQ plateau:
#   staged base PTQ:  bypass=18.50, ADC=27.60
#   lora_r4_down_o:   bypass=18.68, ADC=15.63  ← best
#
# v7 is the ablation that validates and characterises that result.
#
# NOTE: Each run re-trains PTQ from scratch (FlatQuant + staged).
# This means ablation groups measure end-to-end pipeline variance,
# not "clean LoRA-only variance". The first run (base_no_lora) gives
# a v7-internal PTQ control for fair comparison.
#
# Groups (in recommended run order — mandatory first):
#
#   CTRL base_staged_no_lora      — PTQ control in this branch (no LoRA)
#   A.   seed1/2/3_r4_down_o     — end-to-end seed stability
#   B.   rank1/2/4/8_down_o      — rank sweep, alpha=2*rank (scaling=2 constant)
#   C.   r4_down_o_ce_kl         — loss ablation: CE vs CE+KL
#   E.   pre_adc_r4_down_o       — pre-ADC vs post-ADC (key scientific ablation)
#   D.   r4_down_o_last8/first8  — layer-selective efficiency ablation
#   F.   r4_all_ce / r4_all_ce_kl — full coverage, CE then CE+KL
#
# Usage: bash ADC/llama/run_overnight_int4_v7.sh
# Results: ADC/llama/results/overnight_int4_v7_YYYYMMDD.json

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
DATE_TAG="$(date +%Y%m%d)"
RESULTS_JSON="$RESULTS_DIR/overnight_int4_v7_${DATE_TAG}.json"
LOG_DIR="$RESULTS_DIR/logs_v7_${DATE_TAG}"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

echo "============================================================"
echo "INT4 v7 Overnight Experiments (ADC-LoRA ablation study)"
echo "Started: $(date)"
echo "Results JSON: $RESULTS_JSON"
echo "Logs:         $LOG_DIR"
echo "============================================================"

MODEL_NAME="meta-llama/Llama-3.2-1B"
OUTPUT_BASE="$SCRIPT_DIR/checkpoints/overnight_int4_v7_${DATE_TAG}"

# ── Base PTQ: staged_mlpdiag_then_attn (best from v4) ──────────────────────
# Fixes vs v6 BASE_ARGS:
#   - removed --fq_no_diag (was a no-op conflicting with --fq_add_diag, confusing)
#   - added --calibration_max_length 2048 (default is 512 — must be explicit)
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
    --calibration_max_length 2048
    --eval_datasets wikitext2
    --run_no_adc_eval
    --results_json_path "$RESULTS_JSON"
    --disable_visualizations
    --wandb_project llama-flat-quant-adc-ptq-blocks
)

# LoRA common defaults (mode=residual, 5 epochs, lr=1e-4)
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
        --wandb_run_name "${name}_w4a4_int4_v7" \
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
# CONTROL: PTQ baseline without LoRA (validates this branch)
# ============================================================
run_experiment "base_staged_no_lora" \
    --seed 42

# ============================================================
# GROUP A: SEED REPRODUCIBILITY  (lora_r4_down_o, 3 seeds)
# Validates end-to-end stability; each run re-trains PTQ+LoRA.
# ============================================================
for seed in 1 2 3; do
    run_experiment "seed${seed}_r4_down_o" \
        --seed "$seed" \
        "${LORA_BASE[@]}" \
        --lora_rank 4 --lora_alpha 8.0 \
        --lora_target_modules down_proj o_proj \
        --lora_loss ce
done

# ============================================================
# GROUP B: RANK SWEEP  (down_proj+o_proj, r∈{1,2,4,8})
# alpha = 2*rank → scaling = alpha/r = 2 constant across all ranks.
# ============================================================
for rank in 1 2 4 8; do
    alpha=$(python3 -c "print(float($rank * 2))")
    run_experiment "rank${rank}_down_o" \
        --seed 42 \
        "${LORA_BASE[@]}" \
        --lora_rank "$rank" --lora_alpha "$alpha" \
        --lora_target_modules down_proj o_proj \
        --lora_loss ce
done

# ============================================================
# GROUP C: LOSS ABLATION  (r=4, down+o)
# CE+KL: teacher = frozen FP LLaMA loaded on CPU.
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
# GROUP E: PRE-ADC vs POST-ADC LoRA  (r=4, down+o)
# Key scientific ablation: correction before vs after ADC clamp.
# Pre-ADC: Y = ADC(Qx(X) @ Qw(W + scaling*B@A))  [per tile, fp32 params]
# Post-ADC (residual): Y = ADC(Wx) + scaling*lora_B(lora_A(x))
# ============================================================
run_experiment "pre_adc_r4_down_o" \
    --seed 42 \
    "${LORA_BASE[@]}" \
    --lora_rank 4 --lora_alpha 8.0 \
    --lora_target_modules down_proj o_proj \
    --lora_mode pre_adc \
    --lora_loss ce

# ============================================================
# GROUP D: LAYER-SELECTIVE LoRA  (r=4, down+o)
# Last-8 is the main hypothesis; first-8 is the control.
# Llama-3.2-1B has 16 layers (0–15).
# ============================================================
run_experiment "r4_down_o_last8" \
    --seed 42 \
    "${LORA_BASE[@]}" \
    --lora_rank 4 --lora_alpha 8.0 \
    --lora_target_modules down_proj o_proj \
    --lora_layer_indices 8 9 10 11 12 13 14 15 \
    --lora_loss ce

run_experiment "r4_down_o_first8" \
    --seed 42 \
    "${LORA_BASE[@]}" \
    --lora_rank 4 --lora_alpha 8.0 \
    --lora_target_modules down_proj o_proj \
    --lora_layer_indices 0 1 2 3 4 5 6 7 \
    --lora_loss ce

# ============================================================
# GROUP F: FULL COVERAGE, CE then CE+KL  (r=4, all 7 projections)
# r4_all_ce gives a clean within-v7 CE baseline for the KL comparison.
# ============================================================
run_experiment "r4_all_ce" \
    --seed 42 \
    "${LORA_BASE[@]}" \
    --lora_rank 4 --lora_alpha 8.0 \
    --lora_target_modules down_proj up_proj gate_proj q_proj k_proj v_proj o_proj \
    --lora_loss ce

run_experiment "r4_all_ce_kl" \
    --seed 42 \
    "${LORA_BASE[@]}" \
    --lora_rank 4 --lora_alpha 8.0 \
    --lora_target_modules down_proj up_proj gate_proj q_proj k_proj v_proj o_proj \
    --lora_loss ce_kl \
    --lora_kl_weight 0.5 \
    --lora_kl_temperature 2.0

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
print(f\"{'Run':<35}  {'bypass':>8}  {'ADC':>8}  {'gap':>5}  status\")
print('-' * 72)
for r in data:
    s = r.get('results', {})
    bypass = f\"{s['ppl_bypass']:.2f}\" if s.get('ppl_bypass') else 'n/a'
    adc    = f\"{s['ppl_adc']:.2f}\"    if s.get('ppl_adc')    else 'n/a'
    gap    = f\"{s['ppl_adc']/s['ppl_bypass']:.2f}\" if s.get('ppl_adc') and s.get('ppl_bypass') else 'n/a'
    print(f\"{r['run_name']:<35}  {bypass:>8}  {adc:>8}  {gap:>5}  {r['status']}\")
" 2>/dev/null || echo "(python3 not found — check \$RESULTS_JSON manually)"

echo ""
echo "WandB: https://wandb.ai/odu1/llama-flat-quant-adc-ptq-blocks"
echo "Logs:  $LOG_DIR/"
