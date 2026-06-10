#!/bin/bash
# run_overnight_int4_v11_noise.sh — INT4 v11: multiplicative Gaussian noise before ADC
#
# Fixed M = mvm_limit = 256. Injects multiplicative Gaussian noise on the analog
# partial sum before the ADC:
#     y = (x_int4 * w_int4) * N(1, std^2)
#     y_adc = floor(y / delta)
#
# Sweeps the noise std and measures WITH-ADC perplexity on wikitext2 and c4:
#   - BEFORE LoRA  (--eval_pre_lora, evaluated right after PTQ/FlatQuant)
#   - AFTER LoRA   (best v8/v9 config: r4, all 7 projs, CE+KL; trained noise-aware)
#
# PTQ is fixed at fq_nsamples=1024, staged FQ. Only the noise std varies.
#
# Usage: bash ADC/llama/run_overnight_int4_v11_noise.sh
# Results: ADC/llama/results/overnight_int4_v11_noise_YYYYMMDD.json

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
DATE_TAG="$(date +%Y%m%d)"
RESULTS_JSON="$RESULTS_DIR/overnight_int4_v11_noise_${DATE_TAG}.json"
LOG_DIR="$RESULTS_DIR/logs_v11_noise_${DATE_TAG}"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "============================================================"
echo "INT4 v11: multiplicative Gaussian noise before ADC (M=256)"
echo "noise_std ∈ {0.01, 0.05, 0.1, 0.2}"
echo "Started: $(date)"
echo "Results JSON: $RESULTS_JSON"
echo "Logs:         $LOG_DIR"
echo "============================================================"

MODEL_NAME="meta-llama/Llama-3.2-1B"
OUTPUT_BASE="$SCRIPT_DIR/checkpoints/overnight_int4_v11_noise_${DATE_TAG}"

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
    --eval_pre_lora
    --results_json_path "$RESULTS_JSON"
    --disable_visualizations
    --wandb_project llama-flat-quant-adc-ptq-blocks
    --seed 42
)

# Best LoRA config from v8/v9 (r4_all_ce_kl)
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
        --wandb_run_name "${name}_w4a4_int4_v11" \
        "$@" \
        > "$logfile" 2>&1

    local exit_code=$?
    if [ "$exit_code" -eq 0 ]; then
        echo "DONE:  $name  (exit 0)  $(date)"
        local wiki_pre wiki_post c4_pre c4_post
        wiki_pre=$(grep -oP "(?<=BEFORE LoRA -> WIKITEXT2 Perplexity: )\d+\.\d+" "$logfile" | head -1)
        wiki_post=$(grep -oP "(?<=WIKITEXT2 Perplexity: )\d+\.\d+" "$logfile" | tail -1)
        c4_pre=$(grep -oP "(?<=BEFORE LoRA -> C4 Perplexity: )\d+\.\d+" "$logfile" | head -1)
        c4_post=$(grep -oP "(?<=C4 Perplexity: )\d+\.\d+" "$logfile" | tail -1)
        echo "       wiki: pre-LoRA=${wiki_pre:-n/a}  post-LoRA=${wiki_post:-n/a}"
        echo "       c4:   pre-LoRA=${c4_pre:-n/a}    post-LoRA=${c4_post:-n/a}"
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
# Sweep: multiplicative noise std ∈ {0.0, 0.01, 0.02, 0.05, 0.1, 0.2} (M=256)
# ============================================================
for STD in 0.01 0.05 0.1 0.2; do
    # Build a filesystem-friendly tag, e.g. 0.05 -> noise0p05
    tag="noise$(echo "$STD" | tr '.' 'p')"
    run_experiment "$tag" \
        "${LORA_BASE[@]}" \
        --adc_mult_noise_std "$STD"
done

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

def noise_key(r):
    return r.get('config', {}).get('adc_mult_noise_std', 0.0) or 0.0

data = sorted(data, key=noise_key)
print(f\"{'Run':<12}  {'std':>5}  {'wiki-pre':>8}  {'wiki-post':>9}  {'c4-pre':>7}  {'c4-post':>8}  status\")
print('-' * 74)
for r in data:
    s = r.get('results', {})
    std = r.get('config', {}).get('adc_mult_noise_std', '?')
    wpre  = f\"{s['ppl_adc_prelora_wikitext2']:.2f}\" if s.get('ppl_adc_prelora_wikitext2') else 'n/a'
    wpost = f\"{s['ppl_adc_wikitext2']:.2f}\"         if s.get('ppl_adc_wikitext2')         else 'n/a'
    cpre  = f\"{s['ppl_adc_prelora_c4']:.2f}\"        if s.get('ppl_adc_prelora_c4')        else 'n/a'
    cpost = f\"{s['ppl_adc_c4']:.2f}\"                if s.get('ppl_adc_c4')                else 'n/a'
    print(f\"{r['run_name']:<12}  {str(std):>5}  {wpre:>8}  {wpost:>9}  {cpre:>7}  {cpost:>8}  {r['status']}\")
" 2>/dev/null || echo "(python3 not found — check \$RESULTS_JSON manually)"

echo ""
echo "WandB: https://wandb.ai/odu1/llama-flat-quant-adc-ptq-blocks"
echo "Logs:  $LOG_DIR/"
