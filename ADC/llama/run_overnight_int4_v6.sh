#!/bin/bash
# run_overnight_int4_v6.sh — INT4 v6: ADC-LoRA post-correction
#
# Motivation: Pure PTQ (v4/v5) appears near a plateau at ~27.5–28.5 ADC PPL.
# After reparameterize+ADC calibration, we apply per-tile LoRA adapters that
# are trained through the full ADC quantization pipeline:
#   Y = QA(Qx(X) @ Qw(W + scaling * A_i @ B_i))  [per tile]
#
# Base PTQ config: best from v4 — staged_mlpdiag_then_attn
#   bypass=18.50, ADC=27.60
#
# Experiments (all add LoRA on top of the same staged PTQ base):
#   1. lora_r4_down       — rank=4, target=down_proj (driver of ADC gap per v4)
#   2. lora_r8_down       — rank=8, target=down_proj (is more rank worth it?)
#   3. lora_r4_down_o     — rank=4, target=down_proj+o_proj (add attn output)
#   4. lora_r4_all        — rank=4, all 7 projections (full coverage ceiling)
#
# Usage: bash ADC/llama/run_overnight_int4_v6.sh
# Results: ADC/llama/results/overnight_int4_v6_YYYYMMDD.json

set -u  # NO set -e — failures must not stop other runs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
DATE_TAG="$(date +%Y%m%d)"
RESULTS_JSON="$RESULTS_DIR/overnight_int4_v6_${DATE_TAG}.json"
LOG_DIR="$RESULTS_DIR/logs_v6_${DATE_TAG}"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

echo "============================================================"
echo "INT4 v6 Overnight Experiments (ADC-LoRA post-correction)"
echo "Started: $(date)"
echo "Results JSON: $RESULTS_JSON"
echo "Logs:         $LOG_DIR"
echo "============================================================"

MODEL_NAME="meta-llama/Llama-3.2-1B"
OUTPUT_BASE="$SCRIPT_DIR/checkpoints/overnight_int4_v6_${DATE_TAG}"

# Base PTQ: best staged config from v4 (staged_mlpdiag_then_attn)
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
    # Staged PTQ base (same as staged_mlpdiag_then_attn in v4)
    --fq_add_diag
    --fq_diag_mlp
    --fq_stage_b_epochs 10
    --fq_stage_b_prop_alpha 0.5
    --fq_stage_b_diag_attn
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
        --wandb_run_name "${name}_w4a4_int4_v6" \
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

# 1. LoRA rank=4 on down_proj only
#    down_proj is the key ADC contributor (v4: down-only ADC=31.26 vs up-only=39.59)
#    Baseline question: does any LoRA correction help?
run_experiment "lora_r4_down" \
    --lora_rank 4 \
    --lora_alpha 8.0 \
    --lora_target_modules down_proj \
    --lora_epochs 30 \
    --lora_lr 1e-3

# 2. LoRA rank=8 on down_proj only
#    Higher rank = more correction capacity; compare vs rank=4
run_experiment "lora_r8_down" \
    --lora_rank 8 \
    --lora_alpha 16.0 \
    --lora_target_modules down_proj \
    --lora_epochs 30 \
    --lora_lr 1e-3

# 3. LoRA rank=4 on down_proj + o_proj
#    o_proj is the attn output projection (feeds into residual stream directly)
#    Hypothesis: ADC gap partially lives in attn output path
run_experiment "lora_r4_down_o" \
    --lora_rank 4 \
    --lora_alpha 8.0 \
    --lora_target_modules down_proj o_proj \
    --lora_epochs 30 \
    --lora_lr 1e-3

# 4. LoRA rank=4 on all 7 projections
#    Full coverage — ceiling of what LoRA can achieve at rank=4
run_experiment "lora_r4_all" \
    --lora_rank 4 \
    --lora_alpha 8.0 \
    --lora_target_modules down_proj up_proj gate_proj q_proj k_proj v_proj o_proj \
    --lora_epochs 30 \
    --lora_lr 1e-3

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
