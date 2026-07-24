#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
SMOKE="${SMOKE:-0}"
ONLY="${ONLY:-}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
FQ_START_STAGE_B="${FQ_START_STAGE_B:-0}"
FQ_STAGE_A_PATH="${FQ_STAGE_A_PATH:-}"
FORCE_FQ_RETRAIN="${FORCE_FQ_RETRAIN:-0}"
FQ_DIAGNOSTIC_STAGE="${FQ_DIAGNOSTIC_STAGE:-}"
DIAGNOSTIC_WINDOWS="${DIAGNOSTIC_WINDOWS:-8}"
INT4_ADC_OFF_ONLY="${INT4_ADC_OFF_ONLY:-0}"
INT4_ADC_OFF_TRANSFORMS_PATH="${INT4_ADC_OFF_TRANSFORMS_PATH:-}"
RESULTS_JSON="${RESULTS_JSON:-${SCRIPT_DIR}/transfer_results/multi_arch_transfer.json}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SCRIPT_DIR}/transfer_results/checkpoints}"
LOG_ROOT="${LOG_ROOT:-${SCRIPT_DIR}/transfer_results/logs}"

MODEL_KEYS=(
  "llama32_1b"
  "qwen25_15b"
  "smollm2_17b"
  "tinyllama_11b"
)
MODEL_IDS=(
  "meta-llama/Llama-3.2-1B"
  "Qwen/Qwen2.5-1.5B"
  "HuggingFaceTB/SmolLM2-1.7B"
  "TinyLlama/TinyLlama_v1.1"
)

mkdir -p "$(dirname "${RESULTS_JSON}")" "${CHECKPOINT_ROOT}" "${LOG_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

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

has_complete_pre_lora_metrics() {
  local log_path="$1"
  "${PYTHON_BIN}" - "${log_path}" <<'PY'
import sys

with open(sys.argv[1], encoding="utf-8", errors="replace") as handle:
    text = handle.read()

datasets = ("WIKITEXT2", "C4")
tasks = (
    "hellaswag", "mmlu", "winogrande", "arc_easy",
    "arc_challenge", "piqa", "openbookqa", "boolq",
)
complete = (
    all(f"BEFORE LoRA -> {dataset} Perplexity:" in text for dataset in datasets)
    and all(f"lm-eval adc_ptq/{task}:" in text for task in tasks)
)
raise SystemExit(0 if complete else 1)
PY
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

run_int4_adc_off() {
  local key="$1"
  local model_id="$2"
  local run_name="${key}_int4_ptq_adc_off"
  local output_dir="${CHECKPOINT_ROOT}/${key}/int4_ptq_adc_off"
  local log_path="${LOG_ROOT}/${run_name}.log"
  local source_suffix="adc_transfer"

  if [[ "${key}" == "qwen25_15b" ]]; then
    source_suffix="adc_transfer_best_epoch_v2"
  fi

  local transforms_path="${CHECKPOINT_ROOT}/${key}/${source_suffix}/flat_quant_transforms.pt"
  if [[ -n "${INT4_ADC_OFF_TRANSFORMS_PATH}" ]]; then
    transforms_path="${INT4_ADC_OFF_TRANSFORMS_PATH}"
  fi

  if is_completed "${run_name}"; then
    echo "Skipping completed run: ${run_name}"
    return
  fi
  if [[ ! -f "${transforms_path}" ]]; then
    echo "Final pre-LoRA FlatQuant checkpoint not found: ${transforms_path}" >&2
    return 1
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
    --fq_nsamples 1024 \
    --fq_cali_bsz 16 \
    --fq_epochs 30 \
    --fq_diag_mlp \
    --fq_stage_b_epochs 10 \
    --fq_stage_b_prop_alpha 0.5 \
    --fq_stage_b_diag_attn \
    --fq_reload_path "${transforms_path}" \
    --fq_skip_stage_b_on_reload \
    --bx 4 \
    --bw 4 \
    --ba 8 \
    --k 16 \
    --mvm_limit 256 \
    --activation_quant symmetric \
    --calibration_dataset wikitext2 \
    --num_calibration_batches 100 \
    --calibration_batch_size 4 \
    --calibration_max_length 512 \
    --eval_datasets wikitext2 c4 \
    --eval_split test \
    --max_eval_samples 1000 \
    --max_length 2048 \
    --stride 1024 \
    --lora_rank 0 \
    --adc_off_eval_only \
    --run_lm_eval \
    --lm_eval_tasks hellaswag mmlu winogrande arc_easy arc_challenge piqa openbookqa boolq \
    --lm_eval_batch_size auto \
    --skip_model_save \
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
  local fq_cali_bsz=16
  local adc_calibration_batch_size=4
  local calibration_length=512
  local max_eval_samples=1000
  local max_length=2048
  local stride=1024
  local lora_microbatch_size="${LORA_MICROBATCH_SIZE:-4}"
  local lora_gradient_accumulation_steps="${LORA_GRADIENT_ACCUMULATION_STEPS:-1}"
  local -a eval_limit_args=()
  local -a quality_guard_args=(--pre_lora_ppl_threshold 500)
  local -a resume_args=()
  local -a stage_a_propagation_args=()
  local -a optional_args=(
    --fq_save_transforms
    --run_lm_eval
    --lm_eval_tasks hellaswag mmlu winogrande arc_easy arc_challenge piqa openbookqa boolq
    --lm_eval_batch_size auto
  )

  if [[ "${key}" == "qwen25_15b" && -z "${LORA_MICROBATCH_SIZE+x}" ]]; then
    output_suffix="adc_transfer_best_epoch_v2"
    lora_microbatch_size=2
    lora_gradient_accumulation_steps=2
  fi
  if [[ "${key}" == "smollm2_17b" && -z "${LORA_MICROBATCH_SIZE+x}" ]]; then
    lora_microbatch_size=2
    lora_gradient_accumulation_steps=2
  fi
  if [[ "${key}" == "tinyllama_11b" && -z "${LORA_MICROBATCH_SIZE+x}" ]]; then
    lora_microbatch_size=2
    lora_gradient_accumulation_steps=2
  fi
  if [[ "${SMOKE}" == "1" ]]; then
    run_suffix="adc_transfer_smoke"
    output_suffix="adc_transfer_smoke"
    fq_nsamples=2
    fq_epochs=1
    stage_b_epochs=1
    lora_nsamples=2
    lora_epochs=1
    calibration_batches=1
    fq_cali_bsz=1
    adc_calibration_batch_size=1
    calibration_length=64
    max_eval_samples=2
    max_length=64
    stride=32
    lora_microbatch_size=1
    lora_gradient_accumulation_steps=1
    eval_limit_args=(--max_eval_windows 8 --skip_model_save)
    quality_guard_args=()
    optional_args=()
  fi

  local standard_output_dir="${CHECKPOINT_ROOT}/${key}/${output_suffix}"
  local run_name="${key}_${run_suffix}"
  local output_dir="${standard_output_dir}"
  local log_path="${LOG_ROOT}/${run_name}.log"
  local transforms_path="${standard_output_dir}/flat_quant_transforms.pt"
  local stage_a_transforms_path="${standard_output_dir}/flat_quant_transforms_stage_a.pt"

  if [[ -n "${FQ_DIAGNOSTIC_STAGE}" ]]; then
    case "${FQ_DIAGNOSTIC_STAGE}" in
      stage_a)
        transforms_path="${stage_a_transforms_path}"
        ;;
      stage_b)
        ;;
      *)
        echo "FQ_DIAGNOSTIC_STAGE must be stage_a or stage_b" >&2
        return 1
        ;;
    esac
    if [[ ! -f "${transforms_path}" ]]; then
      echo "Diagnostic checkpoint not found: ${transforms_path}" >&2
      return 1
    fi
    run_name="${key}_diagnostic_${FQ_DIAGNOSTIC_STAGE}"
    output_dir="${CHECKPOINT_ROOT}/${key}/diagnostic_${FQ_DIAGNOSTIC_STAGE}"
    log_path="${LOG_ROOT}/${run_name}.log"
    optional_args=(
      --stage_eval
      --stage_eval_max_windows "${DIAGNOSTIC_WINDOWS}"
      --diagnostic_only_after_adc_calibration
      --skip_model_save
    )
    quality_guard_args=()
    resume_args=(--fq_reload_path "${transforms_path}" --fq_skip_stage_b_on_reload)
  elif is_completed "${run_name}"; then
    echo "Skipping completed run: ${run_name}"
    return
  fi

  if [[ -n "${FQ_DIAGNOSTIC_STAGE}" ]]; then
    :
  elif [[ "${FQ_START_STAGE_B}" == "1" ]]; then
    if [[ -n "${FQ_STAGE_A_PATH}" ]]; then
      stage_a_transforms_path="${FQ_STAGE_A_PATH}"
    fi
    if [[ ! -f "${stage_a_transforms_path}" ]]; then
      echo "Stage A checkpoint not found: ${stage_a_transforms_path}" >&2
      return 1
    fi
    resume_args=(--fq_start_stage_b_from "${stage_a_transforms_path}")
  elif [[ "${SMOKE}" != "1" && "${FORCE_FQ_RETRAIN}" != "1" && -f "${transforms_path}" ]]; then
    resume_args=(--fq_reload_path "${transforms_path}" --fq_skip_stage_b_on_reload)
    local pre_lora_resume_log="${log_path%.log}.pre_lora_resume.log"
    if [[ -f "${log_path}" ]] && has_complete_pre_lora_metrics "${log_path}"; then
      cp "${log_path}" "${pre_lora_resume_log}"
      resume_args+=(--pre_lora_metrics_log "${pre_lora_resume_log}")
    elif [[ -f "${pre_lora_resume_log}" ]] && has_complete_pre_lora_metrics "${pre_lora_resume_log}"; then
      resume_args+=(--pre_lora_metrics_log "${pre_lora_resume_log}")
    fi
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
    --fq_cali_bsz "${fq_cali_bsz}" \
    --fq_epochs "${fq_epochs}" \
    --fq_diag_mlp \
    --fq_stage_b_epochs "${stage_b_epochs}" \
    --fq_stage_b_prop_alpha 0.5 \
    --fq_stage_b_diag_attn \
    --bx 4 \
    --bw 4 \
    --ba 8 \
    --k 16 \
    --mvm_limit 256 \
    --activation_quant symmetric \
    --calibration_dataset wikitext2 \
    --num_calibration_batches "${calibration_batches}" \
    --calibration_batch_size "${adc_calibration_batch_size}" \
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
    --lora_microbatch_size "${lora_microbatch_size}" \
    --lora_gradient_accumulation_steps "${lora_gradient_accumulation_steps}" \
    --eval_pre_lora \
    --enforce_transfer_quant_config \
    --disable_visualizations \
    --disable_wandb \
    --wandb_run_name "${run_name}" \
    --results_json_path "${RESULTS_JSON}" \
    "${eval_limit_args[@]}" \
    "${quality_guard_args[@]}" \
    "${stage_a_propagation_args[@]}" \
    "${optional_args[@]}" \
    "${resume_args[@]}"
}

echo "Results: ${RESULTS_JSON}"
echo "Mode: SMOKE=${SMOKE} ONLY=${ONLY:-all} SKIP_COMPLETED=${SKIP_COMPLETED} FQ_START_STAGE_B=${FQ_START_STAGE_B} FORCE_FQ_RETRAIN=${FORCE_FQ_RETRAIN} FQ_DIAGNOSTIC_STAGE=${FQ_DIAGNOSTIC_STAGE:-off} INT4_ADC_OFF_ONLY=${INT4_ADC_OFF_ONLY}"

for index in "${!MODEL_KEYS[@]}"; do
  key="${MODEL_KEYS[${index}]}"
  model_id="${MODEL_IDS[${index}]}"
  if ! is_selected "${key}" "${model_id}"; then
    continue
  fi

  echo "======================================================================"
  echo "Model: ${model_id} (${key})"
  echo "======================================================================"
  if [[ "${INT4_ADC_OFF_ONLY}" == "1" ]]; then
    run_int4_adc_off "${key}" "${model_id}"
    continue
  fi
  if [[ "${SMOKE}" != "1" && -z "${FQ_DIAGNOSTIC_STAGE}" ]]; then
    run_bf16 "${key}" "${model_id}"
  fi
  run_adc_transfer "${key}" "${model_id}"
done

echo "Transfer batch complete. Results: ${RESULTS_JSON}"
