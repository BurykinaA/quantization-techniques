#!/usr/bin/env bash
# Downstream accuracy of the recomputed "FlatQuant, no ADC" row.
#
# Reloads the converter-free FlatQuant transforms saved by
# run_matched_lora_control.sh (BACKBONE=flatquant) and evaluates them with the
# ADC bypassed: WikiText-2 / C4 perplexity (must reproduce the control's
# pre-LoRA numbers) and the eight lm-eval tasks of Table 2. No training.
#
#   DRY_RUN=1 bash ADC/llama/run_flatquant_noadc_downstream.sh
#   bash ADC/llama/run_flatquant_noadc_downstream.sh > ADC/llama/flatquant_noadc_downstream.out 2>&1
#
# Knobs: GPU0_MODELS (llama32_1b tinyllama_11b), GPU1_MODELS (smollm2_17b),
# TORCH_DTYPE (bfloat16; float16 reads matched_control_results_float16),
# LM_EVAL_BATCH_SIZE (16), CONTROL_ROOT, PYTHON_BIN, DRY_RUN, SKIP_COMPLETED (1).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
LM_EVAL_BATCH_SIZE="${LM_EVAL_BATCH_SIZE:-16}"
GPU0_MODELS="${GPU0_MODELS-llama32_1b tinyllama_11b}"
GPU1_MODELS="${GPU1_MODELS-smollm2_17b}"
CONTROL_ROOT="${CONTROL_ROOT:-${SCRIPT_DIR}/matched_control_results}"
if [[ "${TORCH_DTYPE}" != "bfloat16" && -z "${CONTROL_ROOT_SET:-}" ]]; then
  CONTROL_ROOT="${CONTROL_ROOT}_${TORCH_DTYPE}"
fi
OUTPUT_ROOT="${CONTROL_ROOT}/downstream"
RESULTS_DIR="${OUTPUT_ROOT}/results"
LOG_ROOT="${OUTPUT_ROOT}/logs"
RESULTS_LOG="${OUTPUT_ROOT}/results.log"
LM_EVAL_TASKS=(hellaswag mmlu winogrande arc_easy arc_challenge piqa openbookqa boolq)

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
die() { log "ERROR: $*"; exit 1; }

model_id_for() {
  case "$1" in
    llama32_1b) echo "meta-llama/Llama-3.2-1B" ;;
    smollm2_17b) echo "HuggingFaceTB/SmolLM2-1.7B" ;;
    tinyllama_11b) echo "TinyLlama/TinyLlama_v1.1" ;;
    *) die "unknown model key '$1'" ;;
  esac
}

headline() {
  local run_name="$1" json="${RESULTS_DIR}/$1.json"
  [[ -f "${json}" ]] || return 0
  local line
  line="$("${PYTHON_BIN}" - "${json}" "${run_name}" <<'PY'
import json, sys
rec = json.load(open(sys.argv[1]))
tasks = rec.get("downstream_int4_ptq") or {}
accs = [v["accuracy_pct"] for k, v in tasks.items()
        if isinstance(v, dict) and "accuracy_pct" in v and k != "mean"]
avg = f"{sum(accs) / len(accs):.2f}%" if accs else "n/a"
print(f"{sys.argv[2]}: FlatQuant no ADC Wiki {rec.get('ppl_int4_ptq_wikitext2', float('nan')):.2f} "
      f"C4 {rec.get('ppl_int4_ptq_c4', float('nan')):.2f} Avg {avg} ({len(accs)} tasks)")
PY
)"
  log "RESULT ${line}"
  printf '%s %s\n' "$(date '+%F %T')" "${line}" >> "${RESULTS_LOG}"
}

run_eval() {
  local gpu="$1" key="$2"
  local run_name="${key}_flatquant_noadc_eval"
  local transforms="${CONTROL_ROOT}/checkpoints/${key}/flat_quant_transforms.pt"
  local json="${RESULTS_DIR}/${run_name}.json"
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${json}" ]]; then
    log "[gpu${gpu}] skip completed ${run_name}"
    return 0
  fi
  if [[ "${DRY_RUN}" != "1" && ! -f "${transforms}" ]]; then
    log "[gpu${gpu}] FAIL  ${run_name}: ${transforms} not found"
    return 1
  fi
  local -a cmd=(
    "${PYTHON_BIN}" -m ADC.llama.runs.llama_smooth_quant_adc_ptq
    --model_name "$(model_id_for "${key}")"
    --output_dir "${OUTPUT_ROOT}/checkpoints/${key}"
    --no_date_suffix
    --torch_dtype "${TORCH_DTYPE}"
    --preprocess_method flat_quant
    --fq_w_bits 4 --fq_a_bits 4
    --fq_nsamples 1024 --fq_cali_bsz 16 --fq_epochs 30 --fq_diag_mlp
    --fq_bypass_adc_floor --fq_stage_b_epochs 0
    --fq_reload_path "${transforms}" --fq_skip_stage_b_on_reload
    --bx 4 --bw 4 --ba 8 --k 16 --mvm_limit 256
    --activation_quant symmetric
    --calibration_dataset wikitext2
    --num_calibration_batches 100 --calibration_batch_size 4 --calibration_max_length 512
    --eval_datasets wikitext2 c4 --eval_split test
    --max_eval_samples 1000 --max_length 2048 --stride 1024
    --lora_rank 0
    --adc_off_eval_only
    --run_lm_eval
    --lm_eval_tasks "${LM_EVAL_TASKS[@]}"
    --lm_eval_batch_size "${LM_EVAL_BATCH_SIZE}"
    --skip_model_save
    --disable_visualizations
    --disable_wandb
    --wandb_run_name "${run_name}"
    --results_json_path "${json}"
  )
  if [[ "${DRY_RUN}" == "1" ]]; then
    log "[gpu${gpu}] DRY_RUN ${run_name}:"
    printf '  %q' "${cmd[@]}"; printf '\n'
    return 0
  fi
  local logfile="${LOG_ROOT}/${run_name}.log"
  log "[gpu${gpu}] START ${run_name} (log: ${logfile})"
  local start=${SECONDS}
  if CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" > "${logfile}" 2>&1; then
    log "[gpu${gpu}] DONE  ${run_name} ($(( (SECONDS - start) / 60 )) min)"
    headline "${run_name}"
  else
    log "[gpu${gpu}] FAIL  ${run_name} (exit $?, see ${logfile})"
    return 1
  fi
}

run_lane() {
  local gpu="$1"; shift
  local failed=0 key
  for key in "$@"; do
    run_eval "${gpu}" "${key}" || failed=1
  done
  return "${failed}"
}

[[ "${DRY_RUN}" == "1" ]] || mkdir -p "${RESULTS_DIR}" "${LOG_ROOT}" "${OUTPUT_ROOT}/checkpoints"
log "FlatQuant no-ADC downstream: CONTROL_ROOT=${CONTROL_ROOT} TORCH_DTYPE=${TORCH_DTYPE} DRY_RUN=${DRY_RUN}"
log "GPU 0: ${GPU0_MODELS:-<idle>}; GPU 1: ${GPU1_MODELS:-<idle>}"

pids=()
# shellcheck disable=SC2086
[[ -n "${GPU0_MODELS}" ]] && { run_lane 0 ${GPU0_MODELS} & pids+=($!); }
# shellcheck disable=SC2086
[[ -n "${GPU1_MODELS}" ]] && { run_lane 1 ${GPU1_MODELS} & pids+=($!); }
(( ${#pids[@]} > 0 )) || die "both GPU0_MODELS and GPU1_MODELS are empty"

status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
if [[ "${DRY_RUN}" != "1" ]]; then
  log "All lanes finished. Headlines: ${RESULTS_LOG}"
  [[ -f "${RESULTS_LOG}" ]] && cat "${RESULTS_LOG}"
fi
exit "${status}"
