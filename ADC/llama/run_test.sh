#!/bin/bash
# run_test.sh — Compare mvm_limit=256 vs 1024, with and without best LoRA (r4_all_ce_kl)
#
# 4 experiments:
#   mvm256_no_lora        — tile=256, no LoRA   (PTQ control)
#   mvm256_r4_all_ce_kl   — tile=256, r4 all 7 projs, CE+KL  (v8 best)
#   mvm1024_no_lora       — tile=1024, no LoRA  (coarser ADC, delta ≈8065 vs 2016)
#   mvm1024_r4_all_ce_kl  — tile=1024, r4 all 7 projs, CE+KL
#
# Each run saves:
#   checkpoints/test_DATE/<name>/model_full.pt     (torch.save for chat server)
#   checkpoints/test_DATE/<name>/model_info.json   (metadata + PPL)
#
# Usage: bash ADC/llama/run_test.sh
# Results: ADC/llama/results/test_YYYYMMDD.json

# python ADC/llama/serve_chat.py --checkpoints-dir /home/coder/project/ADC/llama/checkpoints/test_20260413
#python ADC/llama/chat_cli.py --checkpoints-dir /home/coder/project/ADC/llama/checkpoints/test_20260413

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
DATE_TAG="$(date +%Y%m%d)"
RESULTS_JSON="$RESULTS_DIR/test_${DATE_TAG}.json"
LOG_DIR="$RESULTS_DIR/logs_test_${DATE_TAG}"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# Use cached datasets — server has no internet access
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "============================================================"
echo "INT4 Test: mvm256 vs mvm1024, with/without r4_all_ce_kl LoRA"
echo "Started: $(date)"
echo "Results JSON: $RESULTS_JSON"
echo "Logs:         $LOG_DIR"
echo "============================================================"

MODEL_NAME="meta-llama/Llama-3.2-1B"
OUTPUT_BASE="$SCRIPT_DIR/checkpoints/test_${DATE_TAG}"

# ── Shared PTQ config (staged best from v4) ─────────────────────────────────
BASE_ARGS_COMMON=(
    --model_name "$MODEL_NAME"
    --preprocess_method flat_quant
    --bx 4 --bw 4 --ba 8 --k 16
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
    --save_full_model_pt
    --seed 42
)

# ── mvm_limit-specific base args ─────────────────────────────────────────────
BASE_MVM256=(  "${BASE_ARGS_COMMON[@]}" --mvm_limit 256  )
BASE_MVM1024=( "${BASE_ARGS_COMMON[@]}" --mvm_limit 1024 )

# ── Best LoRA config from v8 ─────────────────────────────────────────────────
LORA_BEST=(
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
        --output_dir "$out_dir" \
        --wandb_run_name "${name}_w4a4_test" \
        "$@" \
        > "$logfile" 2>&1

    local exit_code=$?
    if [ "$exit_code" -eq 0 ]; then
        echo "DONE:  $name  (exit 0)  $(date)"
        local wiki_adc wiki_bypass c4_adc c4_bypass
        wiki_bypass=$(grep -oP "(?<=WITHOUT ADC -> WIKITEXT2 Perplexity: )\d+\.\d+" "$logfile" | head -1)
        wiki_adc=$(grep -oP "(?<=WIKITEXT2 Perplexity: )\d+\.\d+" "$logfile" | tail -1)
        c4_bypass=$(grep -oP "(?<=WITHOUT ADC -> C4 Perplexity: )\d+\.\d+" "$logfile" | head -1)
        c4_adc=$(grep -oP "(?<=C4 Perplexity: )\d+\.\d+" "$logfile" | tail -1)
        echo "       wiki: bypass=${wiki_bypass:-n/a}  ADC=${wiki_adc:-n/a}"
        echo "       c4:   bypass=${c4_bypass:-n/a}    ADC=${c4_adc:-n/a}"
        if [ -f "${out_dir}/model_full.pt" ]; then
            local sz
            sz=$(du -sh "${out_dir}/model_full.pt" | cut -f1)
            echo "       model_full.pt: $sz"
        fi
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
record = {'run_name': '${name}_w4a4_test', 'status': 'failed', 'exit_code': $exit_code,
          'timestamp': datetime.datetime.now().isoformat()}
data = [r for r in data if r.get('run_name') != '${name}_w4a4_test']
data.append(record)
os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
json.dump(data, open(p, 'w'), indent=2)
" 2>/dev/null || true
    fi

    sleep 10
}

# ============================================================
# 1. mvm_limit=256, no LoRA (PTQ control)
# ============================================================
run_experiment "mvm256_no_lora" "${BASE_MVM256[@]}"

# ============================================================
# 2. mvm_limit=256, r4_all_ce_kl (v8 best)
# ============================================================
run_experiment "mvm256_r4_all_ce_kl" "${BASE_MVM256[@]}" "${LORA_BEST[@]}"

# ============================================================
# 3. mvm_limit=1024, no LoRA
# ============================================================
run_experiment "mvm1024_no_lora" "${BASE_MVM1024[@]}"

# ============================================================
# 4. mvm_limit=1024, r4_all_ce_kl
# ============================================================
run_experiment "mvm1024_r4_all_ce_kl" "${BASE_MVM1024[@]}" "${LORA_BEST[@]}"

# ============================================================
echo ""
echo "============================================================"
echo "ALL EXPERIMENTS DONE"
echo "Finished: $(date)"
echo "============================================================"
echo ""
echo "Results JSON: $RESULTS_JSON"
echo "Checkpoints:  $OUTPUT_BASE"
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
echo "To start the chat server:"
echo "  python ADC/llama/serve_chat.py --checkpoints-dir $OUTPUT_BASE"
echo ""
echo "WandB: https://wandb.ai/odu1/llama-flat-quant-adc-ptq-blocks"
echo "Logs:  $LOG_DIR/"
