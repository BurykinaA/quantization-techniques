#!/usr/bin/env bash
# Midpoint ADC reconstruction: dequantize each non-saturated ADC code at the
# centre of its bin, (q + 0.5) * delta, instead of its lower edge q * delta.
# Floor leaves a one-sided error of about delta/2 per conversion that does not
# average out across tiles; this measures how much of the Stage 3 gain a free
# constant correction already recovers.
#
# Both modes reload the final Stage 1-2 transforms of the transfer run
# (<key>/adc_transfer/flat_quant_transforms.pt) and keep the converter on.
#   MODE=eval    (default) WikiText-2/C4 perplexity with floor (reproduces the
#                Stage 1-2 row) and with midpoint reconstruction; no training
#   MODE=stage3  the transfer-protocol residual (rank 4, all seven projections,
#                CE+KL) retrained with midpoint reconstruction, evaluated
#                before and after
#
# See ADC/llama/ADC_MIDPOINT.md.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODE="${MODE:-eval}"
SMOKE="${SMOKE:-0}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
# The transfer runs (and their transforms) are bfloat16.
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
GPU0_MODELS="${GPU0_MODELS-llama32_1b tinyllama_11b}"
GPU1_MODELS="${GPU1_MODELS-smollm2_17b}"
TRANSFER_CHECKPOINT_ROOT="${TRANSFER_CHECKPOINT_ROOT:-${SCRIPT_DIR}/transfer_results/checkpoints}"
TRANSFER_RESULTS_JSON="${TRANSFER_RESULTS_JSON:-${SCRIPT_DIR}/transfer_results/multi_arch_transfer.json}"
MIDPOINT_ROOT="${MIDPOINT_ROOT:-${SCRIPT_DIR}/adc_midpoint_results}"
if [[ "${TORCH_DTYPE}" != "bfloat16" ]]; then
  MIDPOINT_ROOT="${MIDPOINT_ROOT}_${TORCH_DTYPE}"
fi

LORA_TARGETS=(q_proj k_proj v_proj o_proj gate_proj up_proj down_proj)

OUTPUT_ROOT="${MIDPOINT_ROOT}"
RUN_SUFFIX=""
if [[ "${SMOKE}" == "1" ]]; then
  OUTPUT_ROOT="${MIDPOINT_ROOT}/smoke"
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

# Same physical microbatch / accumulation as run_multi_arch_transfer.sh.
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

# One runner invocation on the reloaded transfer transforms, converter on.
# Extra arguments select floor/midpoint and, for stage3, the residual.
run_one() {
  local key="$1" run_name="$2"
  shift 2
  if is_completed "${run_name}"; then
    log "skip completed ${run_name}"
    return 0
  fi

  local override_name
  override_name="TRANSFORMS_$(printf '%s' "${key}" | tr '[:lower:]' '[:upper:]')"
  local transforms="${!override_name:-}"
  if [[ -z "${transforms}" ]]; then
    transforms="${TRANSFER_CHECKPOINT_ROOT}/${key}/adc_transfer/flat_quant_transforms.pt"
  fi
  if [[ "${DRY_RUN}" != "1" && ! -f "${transforms}" ]]; then
    log "FAIL  ${run_name}: ${transforms} not found (set ${override_name})"
    return 1
  fi

  local calibration_batches=100 adc_calibration_batch_size=4 calibration_length=512
  local max_eval_samples=1000 max_length=2048 stride=1024
  local -a smoke_args=()
  if [[ "${SMOKE}" == "1" ]]; then
    calibration_batches=1
    adc_calibration_batch_size=1
    calibration_length=64
    max_eval_samples=2
    max_length=64
    stride=32
    smoke_args=(--max_eval_windows 8)
  fi

  local output_dir="${OUTPUT_ROOT}/checkpoints/${run_name}"
  mkdir -p "${output_dir}"
  # Same arguments as run_int4_adc_off / run_adc_transfer in
  # run_multi_arch_transfer.sh; the Stage A/B settings are recorded only, the
  # reloaded transforms are final.
  run_step "${run_name}" "${LOG_ROOT}/${run_name}.log" \
    "${PYTHON_BIN}" -m ADC.llama.runs.llama_smooth_quant_adc_ptq \
    --model_name "$(model_id_for "${key}")" \
    --output_dir "${output_dir}" \
    --no_date_suffix \
    --torch_dtype "${TORCH_DTYPE}" \
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
    --fq_reload_path "${transforms}" \
    --fq_skip_stage_b_on_reload \
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
    --skip_model_save \
    --disable_visualizations \
    --disable_wandb \
    --wandb_run_name "${run_name}" \
    --results_json_path "${RESULTS_DIR}/${run_name}.json" \
    ${smoke_args[@]+"${smoke_args[@]}"} \
    "$@"
}

run_model() {
  local key="$1"
  if [[ "${MODE}" == "eval" ]]; then
    local failed=0
    run_one "${key}" "${key}_adc_floor_eval${RUN_SUFFIX}" --lora_rank 0 || failed=1
    run_one "${key}" "${key}_adc_midpoint_eval${RUN_SUFFIX}" --lora_rank 0 --adc_midpoint || failed=1
    return "${failed}"
  fi

  local micro accum lora_nsamples=1024 lora_epochs=5
  read -r micro accum <<< "$(lora_microbatch_for "${key}")"
  local -a mode_args=(--pre_lora_ppl_threshold 500)
  if [[ "${SMOKE}" == "1" ]]; then
    micro=1
    accum=1
    lora_nsamples=2
    lora_epochs=1
    mode_args=()
  fi
  # The transfer protocol check accepts only bfloat16.
  if [[ "${TORCH_DTYPE}" == "bfloat16" ]]; then
    mode_args+=(--enforce_transfer_quant_config)
  fi
  run_one "${key}" "${key}_adc_midpoint_stage3${RUN_SUFFIX}" \
    --adc_midpoint \
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
    --eval_pre_lora \
    ${mode_args[@]+"${mode_args[@]}"}
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

case "${MODE}" in
  eval|stage3) ;;
  *) die "MODE must be eval or stage3, got '${MODE}'" ;;
esac

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

mkdir -p "${RESULTS_DIR}" "${LOG_ROOT}"

log "Mode: MODE=${MODE} TORCH_DTYPE=${TORCH_DTYPE} SMOKE=${SMOKE} DRY_RUN=${DRY_RUN} SKIP_COMPLETED=${SKIP_COMPLETED}"
log "GPU 0: ${GPU0_MODELS:-<idle>}"
log "GPU 1: ${GPU1_MODELS:-<idle>}"
log "Outputs: ${OUTPUT_ROOT}"

if [[ "${DRY_RUN}" != "1" ]] && command -v flock >/dev/null 2>&1; then
  exec 9>"${OUTPUT_ROOT}/.run.lock"
  flock -n 9 || die "another run_adc_midpoint.sh is already using ${OUTPUT_ROOT}"
fi

if [[ -z "${THREADS_PER_LANE:-}" ]]; then
  cpu_count="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)"
  THREADS_PER_LANE=$(( cpu_count / ACTIVE_LANES ))
  (( THREADS_PER_LANE >= 1 )) || THREADS_PER_LANE=1
fi
# MODE=stage3 runs the CE+KL teacher on the CPU; split the cores between lanes.
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
  summary_args=(--results-dir "${RESULTS_DIR}" --run-suffix "${RUN_SUFFIX}"
                --markdown-output "${OUTPUT_ROOT}/summary.md")
  # Smoke numbers use truncated corpora and are not compared with the ADC arm.
  if [[ "${SMOKE}" != "1" && -f "${TRANSFER_RESULTS_JSON}" ]]; then
    summary_args+=(--transfer-json "${TRANSFER_RESULTS_JSON}")
  fi
  "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_adc_midpoint.py" "${summary_args[@]}" \
    || log "summary failed; rerun summarize_adc_midpoint.py manually"
fi
if (( any_failed )); then
  log "Some runs FAILED; see the FAIL lines above"
fi
log "All lanes finished. Summary: ${OUTPUT_ROOT}/summary.md"
# Non-zero when any run failed, so `SMOKE=1 ... && ...` gates the full run.
exit "${any_failed}"
