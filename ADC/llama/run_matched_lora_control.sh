#!/usr/bin/env bash
# Matched no-ADC control for the post-ADC LoRA residual (Stage 3).
#
# The ADC arm already exists (multi-arch transfer runs). This runner trains
# the identical rank-4 residual (all seven projections, CE+KL, same data and
# schedule) with only the ADC floor/clamp bypassed, so the adapter gain with
# and without the converter can be compared on the same backbone.
#
# Arms:
#   flatquant      FlatQuant recalibrated from scratch without the ADC floor
#                  (--fq_bypass_adc_floor, Stage A of the transfer protocol,
#                  no Stage B): one digital reference built the same way for
#                  every model
#   same_backbone  final Stage 1-2 transforms of the ADC-Quant transfer run,
#                  reloaded unchanged, so the converter is the only
#                  difference from the ADC arm (needs the transfer checkpoints)
#
# Each GPU lane runs phase 1 (LoRA training + perplexity, adapter saved) for
# all of its arms first, then phase 2 (downstream suite with the reloaded
# adapter; for flatquant also before LoRA), so every perplexity result lands
# before the long lm-eval runs.
#
# See ADC/llama/MATCHED_LORA_CONTROL.md for usage.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
SMOKE="${SMOKE:-0}"
PREFLIGHT="${PREFLIGHT:-0}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
RUN_DOWNSTREAM="${RUN_DOWNSTREAM:-1}"
FORCE_FQ_RETRAIN="${FORCE_FQ_RETRAIN:-0}"
ADC_LM_EVAL_BATCH_SIZE="${ADC_LM_EVAL_BATCH_SIZE:-1}"
# FlatQuant arms first: their Stage A dominates the run time.
GPU0_ARMS="${GPU0_ARMS-llama32_1b:flatquant tinyllama_11b:flatquant llama32_1b:same_backbone}"
GPU1_ARMS="${GPU1_ARMS-smollm2_17b:flatquant smollm2_17b:same_backbone tinyllama_11b:same_backbone}"
TRANSFER_ROOT="${TRANSFER_ROOT:-${SCRIPT_DIR}/transfer_results}"
TRANSFER_CHECKPOINT_ROOT="${TRANSFER_CHECKPOINT_ROOT:-${TRANSFER_ROOT}/checkpoints}"
TRANSFER_RESULTS_JSON="${TRANSFER_RESULTS_JSON:-${TRANSFER_ROOT}/multi_arch_transfer.json}"
ARCHIVE_DIR="${ARCHIVE_DIR:-}"
CONTROL_ROOT="${CONTROL_ROOT:-${SCRIPT_DIR}/matched_control_results}"

LM_EVAL_TASKS=(hellaswag mmlu winogrande arc_easy arc_challenge piqa openbookqa boolq)
LORA_TARGETS=(q_proj k_proj v_proj o_proj gate_proj up_proj down_proj)

OUTPUT_ROOT="${CONTROL_ROOT}"
RUN_SUFFIX=""
if [[ "${SMOKE}" == "1" ]]; then
  OUTPUT_ROOT="${CONTROL_ROOT}/smoke"
  RUN_SUFFIX="_smoke"
fi
RESULTS_DIR="${OUTPUT_ROOT}/results"
LOG_ROOT="${OUTPUT_ROOT}/logs"
EXTRACT_ROOT="${CONTROL_ROOT}/extracted_transforms"

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

num_layers_for() {
  case "$1" in
    llama32_1b) echo 16 ;;
    smollm2_17b) echo 24 ;;
    tinyllama_11b) echo 22 ;;
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

upper() {
  printf '%s' "$1" | tr '[:lower:]' '[:upper:]'
}

validate_arm_spec() {
  local key="${1%%:*}" arm="${1#*:}"
  model_id_for "${key}" >/dev/null || die "unknown model key in arm '$1'"
  case "${arm}" in
    same_backbone|flatquant) ;;
    *) die "unknown arm '${arm}' in '$1' (use same_backbone or flatquant)" ;;
  esac
}

arm_dir_for() {
  echo "${OUTPUT_ROOT}/checkpoints/$1/$2"
}

run_name_for() {
  echo "$1_noadc_lora_$2${RUN_SUFFIX}"
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

# Locate the final Stage 1-2 transforms of the ADC-Quant transfer run:
# TRANSFORMS_<KEY> override, the transfer checkpoint root, a previous
# extraction, or the member of <key>.tar.gz in the archive directory.
resolve_transfer_transforms() {
  local key="$1"
  local override_name
  override_name="TRANSFORMS_$(upper "${key}")"
  local override="${!override_name:-}"
  if [[ -n "${override}" ]]; then
    [[ -f "${override}" ]] || { log "${override_name}=${override} does not exist" >&2; return 1; }
    echo "${override}"
    return 0
  fi

  local standard="${TRANSFER_CHECKPOINT_ROOT}/${key}/adc_transfer/flat_quant_transforms.pt"
  if [[ -f "${standard}" ]]; then
    echo "${standard}"
    return 0
  fi

  local extracted="${EXTRACT_ROOT}/${key}/adc_transfer/flat_quant_transforms.pt"
  if [[ -f "${extracted}" ]]; then
    echo "${extracted}"
    return 0
  fi

  local archive="" candidate
  for candidate in \
    "${ARCHIVE_DIR}" \
    "${SCRIPT_DIR}/transfer_results_archives/checkpoints/checkpoints" \
    "${SCRIPT_DIR}/transfer_results_archives/checkpoints" \
    "${REPO_ROOT}/transfer_results_archives/checkpoints/checkpoints" \
    "${REPO_ROOT}/transfer_results_archives/checkpoints"; do
    if [[ -n "${candidate}" && -f "${candidate}/${key}.tar.gz" ]]; then
      archive="${candidate}/${key}.tar.gz"
      break
    fi
  done
  if [[ -z "${archive}" ]]; then
    log "no transforms for ${key}: looked for ${standard}, ${extracted}, and ${key}.tar.gz" >&2
    return 1
  fi

  log "searching ${archive} for adc_transfer/flat_quant_transforms.pt" >&2
  local members
  members="$(tar -tzf "${archive}" | grep -E '(^|/)adc_transfer/flat_quant_transforms\.pt$' || true)"
  local count
  count="$(printf '%s' "${members}" | grep -c . || true)"
  if [[ "${count}" != "1" ]]; then
    log "expected exactly one adc_transfer/flat_quant_transforms.pt in ${archive}, found ${count}:" >&2
    printf '%s\n' "${members}" >&2
    return 1
  fi

  local staging="${EXTRACT_ROOT}/.staging_${key}_$$_${RANDOM}"
  mkdir -p "${staging}" "$(dirname "${extracted}")"
  tar -xzf "${archive}" -C "${staging}" "${members}"
  mv "${staging}/${members}" "${extracted}"
  rm -rf "${staging}"
  log "extracted ${archive}:${members} -> ${extracted}" >&2
  echo "${extracted}"
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

# Shared by both arms and both phases: the transfer protocol of the ADC arm,
# with only the converter bypassed (--lora_adc_off_control).
build_common_args() {
  local key="$1"
  local model_id micro accum
  model_id="$(model_id_for "${key}")"
  read -r micro accum <<< "$(lora_microbatch_for "${key}")"

  local fq_nsamples=1024 fq_epochs=30 fq_cali_bsz=16
  local lora_nsamples=1024 lora_epochs=5
  local calibration_batches=100 adc_calibration_batch_size=4 calibration_length=512
  local max_eval_samples=1000 max_length=2048 stride=1024
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
  fi

  COMMON_ARGS=(
    -m ADC.llama.runs.llama_smooth_quant_adc_ptq
    --model_name "${model_id}"
    --no_date_suffix
    --torch_dtype bfloat16
    --preprocess_method flat_quant
    --fq_w_bits 4
    --fq_a_bits 4
    --fq_nsamples "${fq_nsamples}"
    --fq_cali_bsz "${fq_cali_bsz}"
    --fq_epochs "${fq_epochs}"
    --fq_diag_mlp
    --bx 4
    --bw 4
    --ba 8
    --k 16
    --mvm_limit 256
    --activation_quant symmetric
    --calibration_dataset wikitext2
    --num_calibration_batches "${calibration_batches}"
    --calibration_batch_size "${adc_calibration_batch_size}"
    --calibration_max_length "${calibration_length}"
    --eval_datasets wikitext2 c4
    --eval_split test
    --max_eval_samples "${max_eval_samples}"
    --max_length "${max_length}"
    --stride "${stride}"
    --lora_rank 4
    --lora_alpha 8
    --lora_target_modules "${LORA_TARGETS[@]}"
    --lora_epochs "${lora_epochs}"
    --lora_nsamples "${lora_nsamples}"
    --lora_loss ce_kl
    --lora_kl_weight 0.5
    --lora_kl_temperature 2.0
    --lora_microbatch_size "${micro}"
    --lora_gradient_accumulation_steps "${accum}"
    --lora_adc_off_control
    --skip_model_save
    --disable_visualizations
    --disable_wandb
  )
  if [[ "${SMOKE}" == "1" ]]; then
    COMMON_ARGS+=(--max_eval_windows 8)
  fi
}

# Backbone-specific FlatQuant arguments. Phase 2 always reloads.
build_backbone_args() {
  local key="$1" arm="$2" phase="$3"
  local arm_dir
  arm_dir="$(arm_dir_for "${key}" "${arm}")"
  BACKBONE_ARGS=()

  if [[ "${arm}" == "same_backbone" ]]; then
    local transforms
    transforms="$(resolve_transfer_transforms "${key}")" || return 1
    # Stage A/B settings are recorded only; the reloaded transforms are final.
    BACKBONE_ARGS=(
      --fq_stage_b_epochs 10
      --fq_stage_b_prop_alpha 0.5
      --fq_stage_b_diag_attn
      --fq_reload_path "${transforms}"
      --fq_skip_stage_b_on_reload
      --enforce_transfer_quant_config
    )
    return 0
  fi

  local transforms="${arm_dir}/flat_quant_transforms.pt"
  local progress="${arm_dir}/flat_quant_stage_a_progress.pt"
  BACKBONE_ARGS=(--fq_bypass_adc_floor --fq_stage_b_epochs 0)
  if [[ "${phase}" == "2" ]]; then
    BACKBONE_ARGS+=(--fq_reload_path "${transforms}" --fq_skip_stage_b_on_reload)
  elif [[ "${FORCE_FQ_RETRAIN}" != "1" && -f "${transforms}" ]]; then
    log "reusing converter-free FlatQuant transforms: ${transforms}"
    BACKBONE_ARGS+=(--fq_reload_path "${transforms}" --fq_skip_stage_b_on_reload)
  else
    BACKBONE_ARGS+=(--fq_save_transforms --fq_layer_checkpoint_path "${progress}")
    if [[ "${FORCE_FQ_RETRAIN}" != "1" && -f "${progress}" ]]; then
      log "resuming converter-free FlatQuant Stage A from ${progress}"
      BACKBONE_ARGS+=(--fq_resume_layer_checkpoint)
    fi
  fi
}

run_phase1() {
  local spec="$1"
  local key="${spec%%:*}" arm="${spec#*:}"
  local run_name arm_dir
  run_name="$(run_name_for "${key}" "${arm}")"
  arm_dir="$(arm_dir_for "${key}" "${arm}")"

  if is_completed "${run_name}"; then
    log "skip completed ${run_name}"
    return 0
  fi
  build_common_args "${key}"
  build_backbone_args "${key}" "${arm}" 1 || {
    log "FAIL  ${run_name}: backbone transforms unavailable"
    return 1
  }
  local -a guard_args=(--pre_lora_ppl_threshold 500)
  if [[ "${SMOKE}" == "1" ]]; then
    guard_args=()
  fi
  mkdir -p "${arm_dir}"

  run_step "${run_name} [phase 1: LoRA without ADC + PPL]" \
    "${LOG_ROOT}/${run_name}.log" \
    "${PYTHON_BIN}" "${COMMON_ARGS[@]}" "${BACKBONE_ARGS[@]}" \
    ${guard_args[@]+"${guard_args[@]}"} \
    --output_dir "${arm_dir}" \
    --eval_pre_lora \
    --run_adc_on_eval \
    --lora_save_path "${arm_dir}/lora_adapter.pt" \
    --wandb_run_name "${run_name}" \
    --results_json_path "${RESULTS_DIR}/${run_name}.json"
}

run_phase2() {
  local spec="$1"
  local key="${spec%%:*}" arm="${spec#*:}"
  local base_name run_name arm_dir
  base_name="$(run_name_for "${key}" "${arm}")"
  run_name="${base_name}_downstream"
  arm_dir="$(arm_dir_for "${key}" "${arm}")"

  if is_completed "${run_name}"; then
    log "skip completed ${run_name}"
    return 0
  fi
  if [[ "${DRY_RUN}" != "1" && ! -f "${arm_dir}/lora_adapter.pt" ]]; then
    log "skip ${run_name}: phase 1 adapter missing (${arm_dir}/lora_adapter.pt)"
    return 1
  fi
  build_common_args "${key}"
  build_backbone_args "${key}" "${arm}" 2 || {
    log "FAIL  ${run_name}: backbone transforms unavailable"
    return 1
  }
  local -a lm_eval_args=(
    --run_lm_eval
    --lm_eval_tasks "${LM_EVAL_TASKS[@]}"
    --lm_eval_batch_size "${ADC_LM_EVAL_BATCH_SIZE}"
  )
  if [[ "${SMOKE}" == "1" ]]; then
    lm_eval_args=(--run_lm_eval --lm_eval_tasks boolq --lm_eval_batch_size 1 --lm_eval_limit 2)
  fi
  # The recomputed FlatQuant backbone also needs its own pre-LoRA downstream
  # row; the same_backbone one already exists as <key>_int4_ptq_adc_off.
  if [[ "${arm}" == "flatquant" ]]; then
    lm_eval_args+=(--eval_pre_lora)
  fi
  mkdir -p "${arm_dir}/downstream"

  run_step "${run_name} [phase 2: downstream with saved adapter]" \
    "${LOG_ROOT}/${run_name}.log" \
    "${PYTHON_BIN}" "${COMMON_ARGS[@]}" "${BACKBONE_ARGS[@]}" \
    "${lm_eval_args[@]}" \
    --output_dir "${arm_dir}/downstream" \
    --lora_load_path "${arm_dir}/lora_adapter.pt" \
    --wandb_run_name "${run_name}" \
    --results_json_path "${RESULTS_DIR}/${run_name}.json"
}

run_lane() {
  local -a specs=("$@")
  local spec
  for spec in "${specs[@]}"; do
    run_phase1 "${spec}" || true
  done
  if [[ "${RUN_DOWNSTREAM}" == "1" ]]; then
    for spec in "${specs[@]}"; do
      run_phase2 "${spec}" || true
    done
  fi
  log "lane finished"
}

preflight() {
  log "Python: $(command -v "${PYTHON_BIN}" || echo "${PYTHON_BIN} not found")"
  "${PYTHON_BIN}" - "${RUN_DOWNSTREAM}" <<'PY' || die "Python environment check failed"
import importlib
import sys

# accelerate: the CE+KL teacher is loaded with device_map="cpu".
modules = ["torch", "transformers", "datasets", "accelerate", "pytest"]
if sys.argv[1] == "1":
    modules.append("lm_eval")
for name in modules:
    module = importlib.import_module(name)
    print(f"  {name} {getattr(module, '__version__', '?')}")
import torch
print(f"  CUDA devices visible: {torch.cuda.device_count()}")
PY
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader \
      | sed 's/^/  GPU /'
  fi

  "${PYTHON_BIN}" -m pytest -q -p no:cacheprovider \
    "${SCRIPT_DIR}/tests/test_matched_lora_control.py" \
    || die "unit tests failed"

  local lane spec key arm transforms layers
  for lane in "${GPU0_ARMS}" "${GPU1_ARMS}"; do
    for spec in ${lane}; do
      key="${spec%%:*}"
      arm="${spec#*:}"
      [[ "${arm}" == "same_backbone" ]] || continue
      transforms="$(resolve_transfer_transforms "${key}")" \
        || die "transforms for ${key} not found; set TRANSFORMS_$(upper "${key}") or ARCHIVE_DIR, or drop ${spec} from GPU0_ARMS/GPU1_ARMS"
      layers="$(num_layers_for "${key}")"
      "${PYTHON_BIN}" - "${transforms}" "${layers}" "${key}" <<'PY' \
        || die "transforms check failed for ${key}"
import sys

import torch

path, expected_layers, key = sys.argv[1], int(sys.argv[2]), sys.argv[3]
state = torch.load(path, map_location="cpu")
layers = sorted(index for index in state if isinstance(index, int))
if layers != list(range(expected_layers)):
    raise SystemExit(f"{path}: layers {layers[:3]}..{layers[-3:]} != 0..{expected_layers - 1}")
sample = state[layers[0]]
for name in ("attn_ln_trans", "mlp_up_gate_trans", "mlp_down_trans"):
    if name not in sample:
        raise SystemExit(f"{path}: layer 0 lacks {name}")
print(f"  {key}: {path} ({len(layers)} layers) OK")
PY
    done
  done
  if [[ -f "${TRANSFER_RESULTS_JSON}" ]]; then
    log "ADC-arm results for the summary: ${TRANSFER_RESULTS_JSON}"
  else
    log "WARNING: ${TRANSFER_RESULTS_JSON} not found; the summary will lack the ADC arm (set TRANSFER_RESULTS_JSON)"
  fi

  "${PYTHON_BIN}" - "${ACTIVE_LANES}" <<'PY' || die "not enough CUDA devices for the configured lanes"
import sys

import torch

needed = int(sys.argv[1])
if torch.cuda.device_count() < needed:
    raise SystemExit(f"{needed} lanes need {needed} GPUs, found {torch.cuda.device_count()}")
PY
}

# ---------------------------------------------------------------------------

ACTIVE_LANES=0
for lane_arms in "${GPU0_ARMS}" "${GPU1_ARMS}"; do
  for spec in ${lane_arms}; do
    validate_arm_spec "${spec}"
  done
  if [[ -n "${lane_arms// /}" ]]; then
    ACTIVE_LANES=$((ACTIVE_LANES + 1))
  fi
done
(( ACTIVE_LANES > 0 )) || die "both GPU0_ARMS and GPU1_ARMS are empty"

mkdir -p "${RESULTS_DIR}" "${LOG_ROOT}" "${EXTRACT_ROOT}"

log "Mode: SMOKE=${SMOKE} PREFLIGHT=${PREFLIGHT} DRY_RUN=${DRY_RUN} RUN_DOWNSTREAM=${RUN_DOWNSTREAM} SKIP_COMPLETED=${SKIP_COMPLETED}"
log "GPU 0: ${GPU0_ARMS:-<idle>}"
log "GPU 1: ${GPU1_ARMS:-<idle>}"
log "Outputs: ${OUTPUT_ROOT}"

if [[ "${PREFLIGHT}" == "1" ]]; then
  preflight
  log "Preflight passed"
  exit 0
fi

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

# Resolve (and if needed extract) the transfer transforms before the lanes
# start, so two lanes never unpack the same archive concurrently and a missing
# checkpoint is reported now rather than hours into the run.
for lane_arms in "${GPU0_ARMS}" "${GPU1_ARMS}"; do
  for spec in ${lane_arms}; do
    if [[ "${spec#*:}" == "same_backbone" ]]; then
      resolve_transfer_transforms "${spec%%:*}" >/dev/null \
        || log "WARNING: ${spec} will fail: transfer transforms not found"
    fi
  done
done

lane_pids=()
lane_index=0
for lane_arms in "${GPU0_ARMS}" "${GPU1_ARMS}"; do
  read -r -a lane_specs <<< "${lane_arms}"
  if (( ${#lane_specs[@]} > 0 )); then
    if [[ "${DRY_RUN}" == "1" ]]; then
      LANE_LABEL="gpu${lane_index}" run_lane "${lane_specs[@]}"
    else
      (
        export CUDA_VISIBLE_DEVICES="${lane_index}"
        export OMP_NUM_THREADS="${THREADS_PER_LANE}"
        export MKL_NUM_THREADS="${THREADS_PER_LANE}"
        LANE_LABEL="gpu${lane_index}"
        run_lane "${lane_specs[@]}"
      ) &
      lane_pids+=("$!")
    fi
  fi
  lane_index=$((lane_index + 1))
done

for pid in ${lane_pids[@]+"${lane_pids[@]}"}; do
  wait "${pid}" || true
done

if [[ "${DRY_RUN}" != "1" ]]; then
  # Smoke numbers use truncated corpora, so they are not compared with the
  # full ADC-arm records.
  summary_args=(--control-dir "${RESULTS_DIR}" --markdown-output "${OUTPUT_ROOT}/summary.md")
  if [[ "${SMOKE}" == "1" ]]; then
    summary_args+=(--run-suffix "${RUN_SUFFIX}")
  else
    summary_args+=(--transfer-json "${TRANSFER_RESULTS_JSON}")
  fi
  "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_matched_lora_control.py" "${summary_args[@]}" \
    || log "summary failed; rerun summarize_matched_lora_control.py manually"
fi
log "All lanes finished. Summary: ${OUTPUT_ROOT}/summary.md"
