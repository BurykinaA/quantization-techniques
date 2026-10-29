#!/usr/bin/env bash
# Stage 3 without Stages 1-2 on Llama-3.2-1B: the same post-ADC residual
# (rank 4, all seven projections, CE + 0.5 KL, 5 epochs) trained on backbones
# with less pre-ADC calibration, to separate the two calibration stages from
# the residual itself. Arms (ARMS):
#   no_calib  no learned transforms (--preprocess_method none), then Stage 3
#   stage1    Stage 1 only (clean inputs, mu = 0): Stage A with the converter
#             in the loss and no Stage B, evaluated, then Stage 3 on the
#             reloaded transforms
#   stage12   Stage 2 started from the same Stage 1 transforms, then Stage 3:
#             the same-code counterpart of the Table 2 Stages 1-3 row
#
# Stage 1-3 settings follow run_overnight_int4_v8.sh (r4_all_ce_kl), which
# produced the Table 2 row 26.66 -> 14.03, except the LoRA microbatch: 2 x 2
# accumulation instead of 4 x 1, which no longer fits in 80 GB with the
# converter on.
#
# See ADC/llama/STAGE3_ABLATION.md.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
ARMS="${ARMS:-no_calib stage1 stage12}"
SMOKE="${SMOKE:-0}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
# The Table 2 Llama series is float16.
TORCH_DTYPE="${TORCH_DTYPE:-float16}"
LORA_MICROBATCH_SIZE="${LORA_MICROBATCH_SIZE:-2}"
LORA_GRADIENT_ACCUMULATION_STEPS="${LORA_GRADIENT_ACCUMULATION_STEPS:-2}"
ABLATION_ROOT="${ABLATION_ROOT:-${SCRIPT_DIR}/stage3_ablation_results}"

MODEL_KEY="llama32_1b"
MODEL_ID="meta-llama/Llama-3.2-1B"

OUTPUT_ROOT="${ABLATION_ROOT}"
RUN_SUFFIX=""
if [[ "${SMOKE}" == "1" ]]; then
  OUTPUT_ROOT="${ABLATION_ROOT}/smoke"
  RUN_SUFFIX="_smoke"
fi
RESULTS_DIR="${OUTPUT_ROOT}/results"
LOG_ROOT="${OUTPUT_ROOT}/logs"

STAGE1_RUN="${MODEL_KEY}_stage1${RUN_SUFFIX}"
STAGE1_TRANSFORMS="${OUTPUT_ROOT}/checkpoints/${STAGE1_RUN}/flat_quant_transforms.pt"
# Lane 0 signals lane 1, which runs stage12 from the Stage 1 transforms.
STAGE1_READY="${OUTPUT_ROOT}/.stage1_ready"
STAGE1_FAILED="${OUTPUT_ROOT}/.stage1_failed"

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

arm_enabled() {
  [[ " ${ARMS} " == *" $1 "* ]]
}

has_success() {
  local run_name="$1"
  local json_path="${RESULTS_DIR}/${run_name}.json"
  [[ -f "${json_path}" ]] || return 1
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

is_completed() {
  [[ "${SKIP_COMPLETED}" == "1" ]] && has_success "$1"
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

# ---------------------------------------------------------------------------
# Runner arguments

calibration_batches=100
adc_calibration_batch_size=4
calibration_length=512
max_eval_samples=1000
max_length=2048
stride=1024
fq_nsamples=1024
fq_cali_bsz=16
fq_epochs=30
stage_b_epochs=10
lora_nsamples=1024
lora_epochs=5
lora_micro="${LORA_MICROBATCH_SIZE}"
lora_accum="${LORA_GRADIENT_ACCUMULATION_STEPS}"
smoke_args=()
if [[ "${SMOKE}" == "1" ]]; then
  calibration_batches=1
  adc_calibration_batch_size=1
  calibration_length=64
  max_eval_samples=2
  max_length=64
  stride=32
  fq_nsamples=2
  fq_cali_bsz=1
  fq_epochs=1
  stage_b_epochs=1
  lora_nsamples=2
  lora_epochs=1
  lora_micro=1
  lora_accum=1
  smoke_args=(--max_eval_windows 8)
fi

COMMON_ARGS=(
  --model_name "${MODEL_ID}"
  --no_date_suffix
  --torch_dtype "${TORCH_DTYPE}"
  --seed 42
  --bx 4
  --bw 4
  --ba 8
  --k 16
  --mvm_limit 256
  --activation_quant symmetric
  --calibration_dataset wikitext2
  --calibration_method percentile
  --num_calibration_batches "${calibration_batches}"
  --calibration_batch_size "${adc_calibration_batch_size}"
  --calibration_max_length "${calibration_length}"
  --eval_datasets wikitext2 c4
  --eval_split test
  --max_eval_samples "${max_eval_samples}"
  --max_length "${max_length}"
  --stride "${stride}"
  --skip_model_save
  --disable_visualizations
  --disable_wandb
  ${smoke_args[@]+"${smoke_args[@]}"}
)

# Stage 1: clean-input calibration with the converter in the loss, MLP
# diagonals only. fq_nsamples / fq_cali_bsz also set the residual's data.
STAGE1_ARGS=(
  --preprocess_method flat_quant
  --fq_w_bits 4
  --fq_a_bits 4
  --fq_epochs "${fq_epochs}"
  --fq_nsamples "${fq_nsamples}"
  --fq_cali_bsz "${fq_cali_bsz}"
  --fq_lr 0.005
  --fq_diag_mlp
)

# Stage 2: propagated states mixed with clean ones (mu = 0.5), attention
# diagonals released.
STAGE2_ARGS=(
  --fq_stage_b_epochs "${stage_b_epochs}"
  --fq_stage_b_prop_alpha 0.5
  --fq_stage_b_diag_attn
)

STAGE3_ARGS=(
  --lora_mode residual
  --lora_rank 4
  --lora_alpha 8
  --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
  --lora_epochs "${lora_epochs}"
  --lora_lr 1e-4
  --lora_nsamples "${lora_nsamples}"
  --lora_loss ce_kl
  --lora_kl_weight 0.5
  --lora_kl_temperature 2.0
  --lora_microbatch_size "${lora_micro}"
  --lora_gradient_accumulation_steps "${lora_accum}"
  --eval_pre_lora
)

run_runner() {
  local run_name="$1"
  shift
  if is_completed "${run_name}"; then
    log "skip completed ${run_name}"
    return 0
  fi
  local output_dir="${OUTPUT_ROOT}/checkpoints/${run_name}"
  mkdir -p "${output_dir}"
  run_step "${run_name}" "${LOG_ROOT}/${run_name}.log" \
    "${PYTHON_BIN}" -m ADC.llama.runs.llama_smooth_quant_adc_ptq \
    --output_dir "${output_dir}" \
    --wandb_run_name "${run_name}" \
    --results_json_path "${RESULTS_DIR}/${run_name}.json" \
    "${COMMON_ARGS[@]}" \
    "$@"
}

# ---------------------------------------------------------------------------
# Lanes

# GPU 0: Stage 1, then Stage 3 on the reloaded Stage 1 transforms.
lane_stage1() {
  if ! run_runner "${STAGE1_RUN}" \
      "${STAGE1_ARGS[@]}" \
      --fq_stage_b_epochs 0 \
      --fq_save_transforms \
      --lora_rank 0; then
    touch "${STAGE1_FAILED}"
    log "lane finished"
    return 1
  fi
  touch "${STAGE1_READY}"

  local failed=0
  run_runner "${MODEL_KEY}_stage1_stage3${RUN_SUFFIX}" \
    "${STAGE1_ARGS[@]}" \
    --fq_stage_b_epochs 0 \
    --fq_reload_path "${STAGE1_TRANSFORMS}" \
    --fq_skip_stage_b_on_reload \
    "${STAGE3_ARGS[@]}" || failed=1
  log "lane finished"
  return "${failed}"
}

wait_for_stage1() {
  if ! arm_enabled stage1; then
    [[ "${DRY_RUN}" == "1" ]] || has_success "${STAGE1_RUN}" || {
      log "ERROR: stage12 needs a finished ${STAGE1_RUN}; add stage1 to ARMS"
      return 1
    }
    return 0
  fi
  [[ "${DRY_RUN}" == "1" ]] && return 0
  log "waiting for ${STAGE1_RUN} (Stage 1 transforms)"
  until [[ -f "${STAGE1_READY}" ]]; do
    if [[ -f "${STAGE1_FAILED}" ]]; then
      log "ERROR: ${STAGE1_RUN} failed; stage12 not started"
      return 1
    fi
    sleep 60
  done
}

# GPU 1: Stage 3 without calibration, then Stages 1-2 + Stage 3.
lane_other() {
  local failed=0
  if arm_enabled no_calib; then
    run_runner "${MODEL_KEY}_no_calib_stage3${RUN_SUFFIX}" \
      --preprocess_method none \
      --fq_nsamples "${fq_nsamples}" \
      --fq_cali_bsz "${fq_cali_bsz}" \
      "${STAGE3_ARGS[@]}" || failed=1
  fi
  if arm_enabled stage12; then
    if wait_for_stage1; then
      run_runner "${MODEL_KEY}_stage12_stage3${RUN_SUFFIX}" \
        "${STAGE1_ARGS[@]}" \
        "${STAGE2_ARGS[@]}" \
        --fq_start_stage_b_from "${STAGE1_TRANSFORMS}" \
        --fq_save_transforms \
        "${STAGE3_ARGS[@]}" || failed=1
    else
      failed=1
    fi
  fi
  log "lane finished"
  return "${failed}"
}

# ---------------------------------------------------------------------------

[[ -n "${ARMS// /}" ]] || die "ARMS is empty"
for arm in ${ARMS}; do
  case "${arm}" in
    no_calib|stage1|stage12) ;;
    *) die "unknown arm '${arm}'; use no_calib, stage1, stage12" ;;
  esac
done

[[ "${LORA_MICROBATCH_SIZE}" =~ ^[1-9][0-9]*$ && "${LORA_GRADIENT_ACCUMULATION_STEPS}" =~ ^[1-9][0-9]*$ ]] \
  || die "LORA_MICROBATCH_SIZE and LORA_GRADIENT_ACCUMULATION_STEPS must be positive integers"
(( LORA_MICROBATCH_SIZE * LORA_GRADIENT_ACCUMULATION_STEPS == 4 )) \
  || die "LORA_MICROBATCH_SIZE x LORA_GRADIENT_ACCUMULATION_STEPS must equal the effective batch 4"

LANES=()
arm_enabled stage1 && LANES+=("0:lane_stage1")
if arm_enabled no_calib || arm_enabled stage12; then
  LANES+=("1:lane_other")
fi

mkdir -p "${RESULTS_DIR}" "${LOG_ROOT}"

log "ARMS=${ARMS} TORCH_DTYPE=${TORCH_DTYPE} SMOKE=${SMOKE} DRY_RUN=${DRY_RUN} SKIP_COMPLETED=${SKIP_COMPLETED}"
log "LoRA microbatch ${lora_micro} x ${lora_accum} accumulation"
log "Outputs: ${OUTPUT_ROOT}"

if [[ "${DRY_RUN}" != "1" ]] && command -v flock >/dev/null 2>&1; then
  exec 9>"${OUTPUT_ROOT}/.run.lock"
  flock -n 9 || die "another run_stage3_ablation.sh is already using ${OUTPUT_ROOT}"
fi
rm -f "${STAGE1_READY}" "${STAGE1_FAILED}"

if [[ -z "${THREADS_PER_LANE:-}" ]]; then
  cpu_count="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)"
  THREADS_PER_LANE=$(( cpu_count / ${#LANES[@]} ))
  (( THREADS_PER_LANE >= 1 )) || THREADS_PER_LANE=1
fi
# The CE+KL teacher runs on the CPU; split the cores between lanes.
log "CPU threads per lane: ${THREADS_PER_LANE}"

any_failed=0
lane_pids=()
for lane in "${LANES[@]}"; do
  gpu="${lane%%:*}"
  lane_fn="${lane#*:}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    LANE_LABEL="gpu${gpu}" "${lane_fn}" || any_failed=1
  else
    (
      export CUDA_VISIBLE_DEVICES="${gpu}"
      export OMP_NUM_THREADS="${THREADS_PER_LANE}"
      export MKL_NUM_THREADS="${THREADS_PER_LANE}"
      LANE_LABEL="gpu${gpu}"
      "${lane_fn}"
    ) &
    lane_pids+=("$!")
  fi
done

for pid in ${lane_pids[@]+"${lane_pids[@]}"}; do
  wait "${pid}" || any_failed=1
done

if [[ "${DRY_RUN}" != "1" ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_stage3_ablation.py" \
    --results-dir "${RESULTS_DIR}" \
    --run-suffix "${RUN_SUFFIX}" \
    --markdown-output "${OUTPUT_ROOT}/summary.md" \
    || log "summary failed; rerun summarize_stage3_ablation.py manually"
fi
if (( any_failed )); then
  log "Some runs FAILED; see the FAIL lines above"
  exit 1
fi
log "All lanes finished. Summary: ${OUTPUT_ROOT}/summary.md"
