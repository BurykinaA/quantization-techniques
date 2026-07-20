#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
SMOKE="${SMOKE:-0}"
ONLY="${ONLY:-}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
RESULTS_JSON="${RESULTS_JSON:-${SCRIPT_DIR}/transfer_results/multi_arch_transfer.json}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SCRIPT_DIR}/transfer_results/checkpoints}"
LOG_ROOT="${LOG_ROOT:-${SCRIPT_DIR}/transfer_results/logs}"

MODEL_KEYS=(
  "llama32_1b"
  "qwen25_15b"
  "olmo_1b"
  "tinyllama_11b"
)
MODEL_IDS=(
  "meta-llama/Llama-3.2-1B"
  "Qwen/Qwen2.5-1.5B"
  "allenai/OLMo-1B-hf"
  "TinyLlama/TinyLlama_v1.1"
)

mkdir -p "$(dirname "${RESULTS_JSON}")" "${CHECKPOINT_ROOT}" "${LOG_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

is_selected() {
  local key="$1"
  local model_id="$2"
  local requested

  if [[ -z "${ONLY}" ]]; then
    return 0
  fi

  IFS=',' read -r -a requested <<< "${ONLY}"
  local item
  for item in "${requested[@]}"; do
    if [[ "${item}" == "${key}" || "${item}" == "${model_id}" ]]; then
      return 0
    fi
  done
  return 1
}

is_completed() {
  local run_name="$1"
  if [[ "${SKIP_COMPLETED}" != "1" || ! -f "${RESULTS_JSON}" ]]; then
    return 1
  fi

  "${PYTHON_BIN}" - "${RESULTS_JSON}" "${run_name}" <<'PY'
import json
import sys

path, run_name = sys.argv[1:3]
try:
    with open(path, encoding="utf-8") as handle:
        records = json.load(handle)
except (FileNotFoundError, json.JSONDecodeError):
    raise SystemExit(1)

found = any(
    record.get("run_name") == run_name and record.get("status") == "success"
    for record in records
)
raise SystemExit(0 if found else 1)
PY
}

run_logged() {
  local log_path="$1"
  shift
  printf 'Running:'
  printf ' %q' "$@"
  printf '\n'
  "$@" 2>&1 | tee "${log_path}"
}

run_bf16() {
  local key="$1"
  local model_id="$2"
  local run_name="${key}_bf16"
  local output_dir="${CHECKPOINT_ROOT}/${key}/bf16"
  local log_path="${LOG_ROOT}/${run_name}.log"

  if is_completed "${run_name}"; then
    echo "Skipping completed run: ${run_name}"
    return
  fi

  run_logged "${log_path}" \
    "${PYTHON_BIN}" -m ADC.llama.runs.llama_smooth_quant_adc_ptq \
    --model_name "${model_id}" \
    --output_dir "${output_dir}" \
    --no_date_suffix \
    --torch_dtype bfloat16 \
    --preprocess_method none \
    --fp_only_eval \
    --eval_datasets wikitext2 c4 \
    --eval_split test \
    --max_eval_samples 1000 \
    --max_length 2048 \
    --stride 1024 \
    --run_lm_eval \
    --lm_eval_tasks hellaswag mmlu winogrande arc_easy arc_challenge piqa openbookqa boolq \
    --lm_eval_batch_size auto \
    --disable_visualizations \
    --disable_wandb \
    --wandb_run_name "${run_name}" \
    --results_json_path "${RESULTS_JSON}"
}

run_adc_transfer() {
  local key="$1"
  local model_id="$2"
  local run_suffix="adc_transfer"
  local output_suffix="adc_transfer"
  local fq_nsamples=1024
  local fq_epochs=30
  local stage_b_epochs=10
  local lora_nsamples=1024
  local lora_epochs=5
  local calibration_batches=100
  local calibration_batch_size=4
  local calibration_length=512
  local max_eval_samples=1000
  local max_length=2048
  local stride=1024
  local -a optional_args=(
    --run_lm_eval
    --lm_eval_tasks hellaswag mmlu winogrande arc_easy arc_challenge piqa openbookqa boolq
    --lm_eval_batch_size auto
  )

  if [[ "${SMOKE}" == "1" ]]; then
    run_suffix="adc_transfer_smoke"
    output_suffix="adc_transfer_smoke"
    fq_nsamples=2
    fq_epochs=1
    stage_b_epochs=1
    lora_nsamples=2
    lora_epochs=1
    calibration_batches=1
    calibration_batch_size=1
    calibration_length=64
    max_eval_samples=2
    max_length=64
    stride=32
    optional_args=()
  fi

  local run_name="${key}_${run_suffix}"
  local output_dir="${CHECKPOINT_ROOT}/${key}/${output_suffix}"
  local log_path="${LOG_ROOT}/${run_name}.log"

  if is_completed "${run_name}"; then
    echo "Skipping completed run: ${run_name}"
    return
  fi

  run_logged "${log_path}" \
    "${PYTHON_BIN}" -m ADC.llama.runs.llama_smooth_quant_adc_ptq \
    --model_name "${model_id}" \
    --output_dir "${output_dir}" \
    --no_date_suffix \
    --torch_dtype bfloat16 \
    --preprocess_method flat_quant \
    --fq_w_bits 4 \
    --fq_a_bits 4 \
    --fq_nsamples "${fq_nsamples}" \
    --fq_cali_bsz "${calibration_batch_size}" \
    --fq_epochs "${fq_epochs}" \
    --fq_stage_b_epochs "${stage_b_epochs}" \
    --fq_stage_b_prop_alpha 0.5 \
    --bx 4 \
    --bw 4 \
    --ba 8 \
    --k 16 \
    --mvm_limit 256 \
    --activation_quant symmetric \
    --calibration_dataset wikitext2 \
    --num_calibration_batches "${calibration_batches}" \
    --calibration_batch_size "${calibration_batch_size}" \
    --calibration_max_length "${calibration_length}" \
    --eval_datasets wikitext2 c4 \
    --eval_split test \
    --max_eval_samples "${max_eval_samples}" \
    --max_length "${max_length}" \
    --stride "${stride}" \
    --run_no_adc_eval \
    --lora_rank 4 \
    --lora_alpha 8 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --lora_epochs "${lora_epochs}" \
    --lora_nsamples "${lora_nsamples}" \
    --lora_loss ce_kl \
    --lora_kl_weight 0.5 \
    --lora_kl_temperature 2.0 \
    --eval_pre_lora \
    --enforce_transfer_quant_config \
    --disable_visualizations \
    --disable_wandb \
    --wandb_run_name "${run_name}" \
    --results_json_path "${RESULTS_JSON}" \
    "${optional_args[@]}"
}

echo "Results: ${RESULTS_JSON}"
echo "Mode: SMOKE=${SMOKE} ONLY=${ONLY:-all} SKIP_COMPLETED=${SKIP_COMPLETED}"

for index in "${!MODEL_KEYS[@]}"; do
  key="${MODEL_KEYS[${index}]}"
  model_id="${MODEL_IDS[${index}]}"
  if ! is_selected "${key}" "${model_id}"; then
    continue
  fi

  echo "======================================================================"
  echo "Model: ${model_id} (${key})"
  echo "======================================================================"
  if [[ "${SMOKE}" != "1" ]]; then
    run_bf16 "${key}" "${model_id}"
  fi
  run_adc_transfer "${key}" "${model_id}"
done

echo "Transfer batch complete. Results: ${RESULTS_JSON}"
