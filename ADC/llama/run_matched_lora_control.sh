#!/usr/bin/env bash
# Matched no-ADC control for the post-ADC LoRA residual (Stage 3).
#
# Per model, one run:
#   1. the backbone of the model's "FlatQuant, no ADC" row (BACKBONE): FlatQuant
#      calibrated from scratch without the ADC floor/clamp, or the transfer
#      run's ADC-Quant transforms reloaded
#   2. perplexity before LoRA, ADC bypassed (reproduces that row)
#   3. the transfer-protocol residual (rank 4, all seven projections, CE+KL)
#      trained with the ADC bypassed (--lora_adc_off_control)
#   4. perplexity after LoRA, ADC bypassed
#
# The gain of step 4 over step 2 is compared with the Stage 3 gain of the
# ADC arm, in that arm's dtype (TORCH_DTYPE). See
# ADC/llama/MATCHED_LORA_CONTROL.md.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
SMOKE="${SMOKE:-0}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
FORCE_FQ_RETRAIN="${FORCE_FQ_RETRAIN:-0}"
# Use the dtype of the ADC arm being compared against: the Llama rows of
# Table 2 ran in float16 (the runner default), the transfer runs (SmolLM2,
# TinyLlama) in bfloat16.
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
# Backbone under the no-ADC adapter; pick the one that produced the model's
# "FlatQuant, no ADC" row in Table 2:
#   flatquant  FlatQuant recalibrated without the floor (the Llama row)
#   adcquant   the transfer run's final ADC-Quant transforms with the ADC
#              bypassed, i.e. <key>_int4_ptq_adc_off (the SmolLM2/TinyLlama rows)
BACKBONE="${BACKBONE:-flatquant}"
TRANSFER_CHECKPOINT_ROOT="${TRANSFER_CHECKPOINT_ROOT:-${SCRIPT_DIR}/transfer_results/checkpoints}"
# Stage A dominates the run time (about 8 h for SmolLM2 per
# MULTI_ARCH_TRANSFER.md), so SmolLM2 gets a GPU of its own.
GPU0_MODELS="${GPU0_MODELS-llama32_1b tinyllama_11b}"
GPU1_MODELS="${GPU1_MODELS-smollm2_17b}"
CONTROL_ROOT="${CONTROL_ROOT:-${SCRIPT_DIR}/matched_control_results}"
# Other dtypes get their own directory, so a run never reuses transforms or
# completed records calibrated in a different dtype.
if [[ "${TORCH_DTYPE}" != "bfloat16" ]]; then
  CONTROL_ROOT="${CONTROL_ROOT}_${TORCH_DTYPE}"
fi

LORA_TARGETS=(q_proj k_proj v_proj o_proj gate_proj up_proj down_proj)

OUTPUT_ROOT="${CONTROL_ROOT}"
RUN_SUFFIX=""
if [[ "${SMOKE}" == "1" ]]; then
  OUTPUT_ROOT="${CONTROL_ROOT}/smoke"
  RUN_SUFFIX="_smoke"
fi
RESULTS_DIR="${OUTPUT_ROOT}/results"
LOG_ROOT="${OUTPUT_ROOT}/logs"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

log() {
  printf '[%s] [%s] %s\n' "$(date '+%F %T')" "${LANE_LABEL:-main}" "$*"
}

die() {
  log "ERROR: $*" >&2
  exit 1
}

model_id_for() {
  case "$1" in
    llama32_1b) echo "meta-llama/Llama-3.2-1B" ;;
    smollm2_17b) echo "HuggingFaceTB/SmolLM2-1.7B" ;;
    tinyllama_11b) echo "TinyLlama/TinyLlama_v1.1" ;;
    *) return 1 ;;
  esac
}

# Same physical microbatch / accumulation as run_multi_arch_transfer.sh, so the
# effective LoRA batch (4) and the memory footprint match the ADC arm.
lora_microbatch_for() {
  case "$1" in
    llama32_1b) echo "4 1" ;;
    *) echo "2 2" ;;
  esac
}

is_completed() {
  local run_name="$1"
  local json_path="${RESULTS_DIR}/${run_name}.json"
  if [[ "${SKIP_COMPLETED}" != "1" || ! -f "${json_path}" ]]; then
    return 1
  fi
  "${PYTHON_BIN}" - "${json_path}" "${run_name}" <<'PY'
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

run_step() {
  local label="$1" log_path="$2"
  shift 2
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[dry-run] %s\n ' "${label}"
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi

  local started="${SECONDS}" status=0
  log "START ${label} (log: ${log_path})"
  {
    printf 'Running:'
    printf ' %q' "$@"
    printf '\n'
  } > "${log_path}"
  "$@" >> "${log_path}" 2>&1 || status=$?
  local minutes=$(( (SECONDS - started) / 60 ))
  if (( status == 0 )); then
    log "DONE  ${label} (${minutes} min)"
  else
    log "FAIL  ${label} (exit ${status} after ${minutes} min); see ${log_path}"
  fi
  return "${status}"
}

run_model() {
  local key="$1"
  local model_id micro accum
  model_id="$(model_id_for "${key}")"
  read -r micro accum <<< "$(lora_microbatch_for "${key}")"
  local run_name="${key}_${BACKBONE}_noadc_lora${RUN_SUFFIX}"
  local output_dir="${OUTPUT_ROOT}/checkpoints/${key}"
  if [[ "${BACKBONE}" == "adcquant" ]]; then
    output_dir="${output_dir}_adcquant"
  fi
  local transforms="${output_dir}/flat_quant_transforms.pt"
  local progress="${output_dir}/flat_quant_stage_a_progress.pt"

  if is_completed "${run_name}"; then
    log "skip completed ${run_name}"
    return 0
  fi

  # Transfer protocol (run_multi_arch_transfer.sh), so both arms share Stage A
  # settings, the LoRA batches, and the evaluation corpora.
  local fq_nsamples=1024 fq_epochs=30 fq_cali_bsz=16
  local lora_nsamples=1024 lora_epochs=5
  local calibration_batches=100 adc_calibration_batch_size=4 calibration_length=512
  local max_eval_samples=1000 max_length=2048 stride=1024
  local -a mode_args=(--pre_lora_ppl_threshold 500)
  if [[ "${SMOKE}" == "1" ]]; then
    fq_nsamples=2
    fq_epochs=1
    fq_cali_bsz=1
    lora_nsamples=2
    lora_epochs=1
    calibration_batches=1
    adc_calibration_batch_size=1
    calibration_length=64
    max_eval_samples=2
    max_length=64
    stride=32
    micro=1
    accum=1
    mode_args=(--max_eval_windows 8)
  fi

  local -a fq_args
  if [[ "${BACKBONE}" == "adcquant" ]]; then
    local override_name
    override_name="TRANSFORMS_$(printf '%s' "${key}" | tr '[:lower:]' '[:upper:]')"
    local transfer_transforms="${!override_name:-}"
    if [[ -z "${transfer_transforms}" ]]; then
      transfer_transforms="${TRANSFER_CHECKPOINT_ROOT}/${key}/adc_transfer/flat_quant_transforms.pt"
    fi
    if [[ "${DRY_RUN}" != "1" && ! -f "${transfer_transforms}" ]]; then
      log "FAIL  ${run_name}: ${transfer_transforms} not found (set ${override_name})"
      return 1
    fi
    # Same arguments as run_int4_adc_off in run_multi_arch_transfer.sh; the
    # Stage A/B settings are recorded only, the reloaded transforms are final.
    fq_args=(
      --fq_stage_b_epochs 10
      --fq_stage_b_prop_alpha 0.5
      --fq_stage_b_diag_attn
      --fq_reload_path "${transfer_transforms}"
      --fq_skip_stage_b_on_reload
    )
  # flatquant: reuse finished transforms, resume an interrupted Stage A, or
  # start fresh.
  elif [[ "${FORCE_FQ_RETRAIN}" != "1" && -f "${transforms}" ]]; then
    log "reusing FlatQuant transforms: ${transforms}"
    fq_args=(
      --fq_bypass_adc_floor --fq_stage_b_epochs 0
      --fq_reload_path "${transforms}" --fq_skip_stage_b_on_reload
    )
  else
    fq_args=(
      --fq_bypass_adc_floor --fq_stage_b_epochs 0
      --fq_save_transforms --fq_layer_checkpoint_path "${progress}"
    )
    if [[ "${FORCE_FQ_RETRAIN}" != "1" && -f "${progress}" ]]; then
      log "resuming FlatQuant Stage A from ${progress}"
      fq_args+=(--fq_resume_layer_checkpoint)
    fi
  fi

  mkdir -p "${output_dir}"
  run_step "${run_name}" "${LOG_ROOT}/${run_name}.log" \
    "${PYTHON_BIN}" -m ADC.llama.runs.llama_smooth_quant_adc_ptq \
    --model_name "${model_id}" \
    --output_dir "${output_dir}" \
    --no_date_suffix \
    --torch_dtype "${TORCH_DTYPE}" \
    --preprocess_method flat_quant \
    --fq_w_bits 4 \
    --fq_a_bits 4 \
    --fq_nsamples "${fq_nsamples}" \
    --fq_cali_bsz "${fq_cali_bsz}" \
    --fq_epochs "${fq_epochs}" \
    --fq_diag_mlp \
    "${fq_args[@]}" \
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
    --lora_rank 4 \
    --lora_alpha 8 \
    --lora_target_modules "${LORA_TARGETS[@]}" \
    --lora_epochs "${lora_epochs}" \
    --lora_nsamples "${lora_nsamples}" \
    --lora_loss ce_kl \
    --lora_kl_weight 0.5 \
    --lora_kl_temperature 2.0 \
    --lora_microbatch_size "${micro}" \
    --lora_gradient_accumulation_steps "${accum}" \
    --lora_adc_off_control \
    --eval_pre_lora \
    "${mode_args[@]}" \
    --skip_model_save \
    --disable_visualizations \
    --disable_wandb \
    --wandb_run_name "${run_name}" \
    --results_json_path "${RESULTS_DIR}/${run_name}.json"
}

run_lane() {
  local key failed=0
  for key in "$@"; do
    run_model "${key}" || failed=1
  done
  log "lane finished"
  return "${failed}"
}

# ---------------------------------------------------------------------------

ACTIVE_LANES=0
for lane_models in "${GPU0_MODELS}" "${GPU1_MODELS}"; do
  for key in ${lane_models}; do
    model_id_for "${key}" >/dev/null || die "unknown model key '${key}'"
  done
  if [[ -n "${lane_models// /}" ]]; then
    ACTIVE_LANES=$((ACTIVE_LANES + 1))
  fi
done
(( ACTIVE_LANES > 0 )) || die "both GPU0_MODELS and GPU1_MODELS are empty"
case "${BACKBONE}" in
  flatquant|adcquant) ;;
  *) die "BACKBONE must be flatquant or adcquant, got '${BACKBONE}'" ;;
esac

mkdir -p "${RESULTS_DIR}" "${LOG_ROOT}"

log "Mode: BACKBONE=${BACKBONE} TORCH_DTYPE=${TORCH_DTYPE} SMOKE=${SMOKE} DRY_RUN=${DRY_RUN} SKIP_COMPLETED=${SKIP_COMPLETED} FORCE_FQ_RETRAIN=${FORCE_FQ_RETRAIN}"
log "GPU 0: ${GPU0_MODELS:-<idle>}"
log "GPU 1: ${GPU1_MODELS:-<idle>}"
log "Outputs: ${OUTPUT_ROOT}"

if [[ "${DRY_RUN}" != "1" ]] && command -v flock >/dev/null 2>&1; then
  exec 9>"${OUTPUT_ROOT}/.run.lock"
  flock -n 9 || die "another run_matched_lora_control.sh is already using ${OUTPUT_ROOT}"
fi

if [[ -z "${THREADS_PER_LANE:-}" ]]; then
  cpu_count="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)"
  THREADS_PER_LANE=$(( cpu_count / ACTIVE_LANES ))
  (( THREADS_PER_LANE >= 1 )) || THREADS_PER_LANE=1
fi
# The CE+KL teacher runs on the CPU; split the cores so the two lanes do not
# oversubscribe them while both train adapters.
log "CPU threads per lane: ${THREADS_PER_LANE}"

any_failed=0
lane_pids=()
lane_index=0
for lane_models in "${GPU0_MODELS}" "${GPU1_MODELS}"; do
  read -r -a lane_keys <<< "${lane_models}"
  if (( ${#lane_keys[@]} > 0 )); then
    if [[ "${DRY_RUN}" == "1" ]]; then
      LANE_LABEL="gpu${lane_index}" run_lane "${lane_keys[@]}" || any_failed=1
    else
      (
        export CUDA_VISIBLE_DEVICES="${lane_index}"
        export OMP_NUM_THREADS="${THREADS_PER_LANE}"
        export MKL_NUM_THREADS="${THREADS_PER_LANE}"
        LANE_LABEL="gpu${lane_index}"
        run_lane "${lane_keys[@]}"
      ) &
      lane_pids+=("$!")
    fi
  fi
  lane_index=$((lane_index + 1))
done

for pid in ${lane_pids[@]+"${lane_pids[@]}"}; do
  wait "${pid}" || any_failed=1
done

if [[ "${DRY_RUN}" != "1" ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_matched_lora_control.py" \
    --results-dir "${RESULTS_DIR}" \
    --run-suffix "${RUN_SUFFIX}" \
    --markdown-output "${OUTPUT_ROOT}/summary.md" \
    || log "summary failed; rerun summarize_matched_lora_control.py manually"
fi
if (( any_failed )); then
  log "Some runs FAILED; see the FAIL lines above"
fi
log "All lanes finished. Summary: ${OUTPUT_ROOT}/summary.md"
# Non-zero when any run failed, so `SMOKE=1 ... && ...` gates the full run.
exit "${any_failed}"
