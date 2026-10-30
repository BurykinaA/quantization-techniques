#!/usr/bin/env bash
# Stage 3 without Stages 1-2: the same post-ADC residual (rank 4, all seven
# projections, CE + 0.5 KL, 5 epochs) trained on backbones with less pre-ADC
# calibration, to separate the two calibration stages from the residual
# itself. Arms (ARMS):
#   no_calib  no learned transforms (--preprocess_method none), then Stage 3;
#             every model in GPU0_MODELS / GPU1_MODELS
#   stage1    Llama only: Stage 1 alone (clean inputs, mu = 0), Stage A with
#             the converter in the loss and no Stage B, evaluated, then Stage 3
#             on the reloaded transforms
#   stage12   Llama only: Stage 2 started from the same Stage 1 transforms,
#             then Stage 3; the same-code counterpart of the Table 2 row
#
# Settings follow the runs behind the Table 2 Stages 1-3 rows:
# run_overnight_int4_v8.sh (r4_all_ce_kl, float16) for Llama and
# run_multi_arch_transfer.sh (bfloat16) for SmolLM2 and TinyLlama. The LoRA
# microbatch is 2 x 2 accumulation for every model; Llama's 4 x 1 no longer
# fits in 80 GB with the converter on.
#
# LM_EVAL=1 adds the paper's eight downstream tasks, evaluated after Stage 3
# only. Every Stage 3 run saves its adapters as
# checkpoints/<run>/lora_weights.pt for later eval-only reruns.
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
# Models of the no_calib arm per GPU. GPU 0 also runs stage1 first, GPU 1
# runs stage12 last.
GPU0_MODELS="${GPU0_MODELS-smollm2_17b}"
GPU1_MODELS="${GPU1_MODELS-llama32_1b tinyllama_11b}"
LORA_MICROBATCH_SIZE="${LORA_MICROBATCH_SIZE:-2}"
LORA_GRADIENT_ACCUMULATION_STEPS="${LORA_GRADIENT_ACCUMULATION_STEPS:-2}"
ABLATION_ROOT="${ABLATION_ROOT:-${SCRIPT_DIR}/stage3_ablation_results}"
# Overrides the per-model dtype for every run, and LORA_DTYPE the dtype of
# the Stage 3 residual (weights, arithmetic and optimizer state). Such runs
# get their own directory, so they never reuse records made in the Table 2
# setting.
TORCH_DTYPE="${TORCH_DTYPE:-}"
LORA_DTYPE="${LORA_DTYPE:-float32}"
if [[ -n "${TORCH_DTYPE}" ]]; then
  ABLATION_ROOT="${ABLATION_ROOT}_${TORCH_DTYPE}"
fi
if [[ "${LORA_DTYPE}" != "float32" ]]; then
  ABLATION_ROOT="${ABLATION_ROOT}_lora_${LORA_DTYPE}"
fi
# LM_EVAL=1 adds lm-evaluation-harness on the paper's eight tasks (MMLU
# 5-shot, the rest 0-shot) after Stage 3; the uncalibrated backbones are
# collapsed before it. The batch size follows the transfer runs.
LM_EVAL="${LM_EVAL:-0}"
LM_EVAL_BATCH_SIZE="${LM_EVAL_BATCH_SIZE:-1}"
# The CE+KL teacher of Stage 3. On the GPU it needs about 5 GB next to the
# student; fall back to cpu if Stage 3 runs out of memory.
LORA_TEACHER_DEVICE="${LORA_TEACHER_DEVICE:-cuda}"

OUTPUT_ROOT="${ABLATION_ROOT}"
RUN_SUFFIX=""
if [[ "${SMOKE}" == "1" ]]; then
  OUTPUT_ROOT="${ABLATION_ROOT}/smoke"
  RUN_SUFFIX="_smoke"
fi
RESULTS_DIR="${OUTPUT_ROOT}/results"
LOG_ROOT="${OUTPUT_ROOT}/logs"

STAGE1_RUN="llama32_1b_stage1${RUN_SUFFIX}"
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

model_id_for() {
  case "$1" in
    llama32_1b) echo "meta-llama/Llama-3.2-1B" ;;
    smollm2_17b) echo "HuggingFaceTB/SmolLM2-1.7B" ;;
    tinyllama_11b) echo "TinyLlama/TinyLlama_v1.1" ;;
    *) return 1 ;;
  esac
}

# dtype of the model's Table 2 runs: the Llama series is float16, the
# transfer runs are bfloat16. TORCH_DTYPE overrides it.
dtype_for() {
  if [[ -n "${TORCH_DTYPE}" ]]; then
    echo "${TORCH_DTYPE}"
    return 0
  fi
  case "$1" in
    llama32_1b) echo "float16" ;;
    *) echo "bfloat16" ;;
  esac
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
  # A rerun (SKIP_COMPLETED=0) keeps the previous log.
  if [[ -f "${log_path}" ]]; then
    mv "${log_path}" "${log_path}.$(date '+%Y%m%d_%H%M%S')"
  fi
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

lm_eval_args=()
if [[ "${LM_EVAL}" == "1" ]]; then
  lm_eval_args=(
    --run_lm_eval
    --lm_eval_tasks hellaswag mmlu winogrande arc_easy arc_challenge piqa openbookqa boolq
    --lm_eval_batch_size "${LM_EVAL_BATCH_SIZE}"
    --no_pre_lora_lm_eval
  )
  if [[ "${SMOKE}" == "1" ]]; then
    lm_eval_args+=(--lm_eval_limit 4)
  fi
fi

COMMON_ARGS=(
  --no_date_suffix
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
  ${lm_eval_args[@]+"${lm_eval_args[@]}"}
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
  --lora_dtype "${LORA_DTYPE}"
  --eval_pre_lora
  --save_lora_weights
  --lora_teacher_device "${LORA_TEACHER_DEVICE}"
)

run_runner() {
  local key="$1" run_name="$2"
  shift 2
  if is_completed "${run_name}"; then
    log "skip completed ${run_name}"
    return 0
  fi
  local output_dir="${OUTPUT_ROOT}/checkpoints/${run_name}"
  mkdir -p "${output_dir}"
  run_step "${run_name}" "${LOG_ROOT}/${run_name}.log" \
    "${PYTHON_BIN}" -m ADC.llama.runs.llama_smooth_quant_adc_ptq \
    --model_name "$(model_id_for "${key}")" \
    --torch_dtype "$(dtype_for "${key}")" \
    --output_dir "${output_dir}" \
    --wandb_run_name "${run_name}" \
    --results_json_path "${RESULTS_DIR}/${run_name}.json" \
    "${COMMON_ARGS[@]}" \
    "$@"
}

run_no_calib() {
  local key="$1"
  run_runner "${key}" "${key}_no_calib_stage3${RUN_SUFFIX}" \
    --preprocess_method none \
    --fq_nsamples "${fq_nsamples}" \
    --fq_cali_bsz "${fq_cali_bsz}" \
    "${STAGE3_ARGS[@]}"
}

run_stage1() {
  if ! run_runner llama32_1b "${STAGE1_RUN}" \
      "${STAGE1_ARGS[@]}" \
      --fq_stage_b_epochs 0 \
      --fq_save_transforms \
      --lora_rank 0; then
    touch "${STAGE1_FAILED}"
    return 1
  fi
  touch "${STAGE1_READY}"
  run_runner llama32_1b "llama32_1b_stage1_stage3${RUN_SUFFIX}" \
    "${STAGE1_ARGS[@]}" \
    --fq_stage_b_epochs 0 \
    --fq_reload_path "${STAGE1_TRANSFORMS}" \
    --fq_skip_stage_b_on_reload \
    "${STAGE3_ARGS[@]}"
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

run_stage12() {
  wait_for_stage1 || return 1
  run_runner llama32_1b "llama32_1b_stage12_stage3${RUN_SUFFIX}" \
    "${STAGE1_ARGS[@]}" \
    "${STAGE2_ARGS[@]}" \
    --fq_start_stage_b_from "${STAGE1_TRANSFORMS}" \
    --fq_save_transforms \
    "${STAGE3_ARGS[@]}"
}

# ---------------------------------------------------------------------------
# Lanes

# GPU 0: stage1, then the no_calib arm for GPU0_MODELS.
lane_gpu0() {
  local key failed=0
  if arm_enabled stage1; then
    run_stage1 || failed=1
  fi
  if arm_enabled no_calib; then
    for key in ${GPU0_MODELS}; do
      run_no_calib "${key}" || failed=1
    done
  fi
  log "lane finished"
  return "${failed}"
}

# GPU 1: the no_calib arm for GPU1_MODELS, then stage12.
lane_gpu1() {
  local key failed=0
  if arm_enabled no_calib; then
    for key in ${GPU1_MODELS}; do
      run_no_calib "${key}" || failed=1
    done
  fi
  if arm_enabled stage12; then
    run_stage12 || failed=1
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
for key in ${GPU0_MODELS} ${GPU1_MODELS}; do
  model_id_for "${key}" >/dev/null || die "unknown model key '${key}'"
done

[[ "${LORA_MICROBATCH_SIZE}" =~ ^[1-9][0-9]*$ && "${LORA_GRADIENT_ACCUMULATION_STEPS}" =~ ^[1-9][0-9]*$ ]] \
  || die "LORA_MICROBATCH_SIZE and LORA_GRADIENT_ACCUMULATION_STEPS must be positive integers"
(( LORA_MICROBATCH_SIZE * LORA_GRADIENT_ACCUMULATION_STEPS == 4 )) \
  || die "LORA_MICROBATCH_SIZE x LORA_GRADIENT_ACCUMULATION_STEPS must equal the effective batch 4"
[[ -z "${TORCH_DTYPE}" || "${TORCH_DTYPE}" =~ ^(float16|bfloat16|float32)$ ]] \
  || die "TORCH_DTYPE must be float16, bfloat16 or float32"
[[ "${LORA_DTYPE}" =~ ^(float32|bfloat16|float16)$ ]] \
  || die "LORA_DTYPE must be float32, bfloat16 or float16"
[[ "${LM_EVAL}" =~ ^[01]$ ]] || die "LM_EVAL must be 0 or 1"
[[ "${LM_EVAL_BATCH_SIZE}" =~ ^(auto|[1-9][0-9]*)$ ]] \
  || die "LM_EVAL_BATCH_SIZE must be auto or a positive integer"
[[ "${LORA_TEACHER_DEVICE}" =~ ^(cpu|cuda)$ ]] || die "LORA_TEACHER_DEVICE must be cpu or cuda"

LANES=()
if arm_enabled stage1 || { arm_enabled no_calib && [[ -n "${GPU0_MODELS// /}" ]]; }; then
  LANES+=("0:lane_gpu0")
fi
if arm_enabled stage12 || { arm_enabled no_calib && [[ -n "${GPU1_MODELS// /}" ]]; }; then
  LANES+=("1:lane_gpu1")
fi
(( ${#LANES[@]} > 0 )) || die "nothing to run: no_calib needs GPU0_MODELS or GPU1_MODELS"

mkdir -p "${RESULTS_DIR}" "${LOG_ROOT}"

log "ARMS=${ARMS} SMOKE=${SMOKE} DRY_RUN=${DRY_RUN} SKIP_COMPLETED=${SKIP_COMPLETED} TORCH_DTYPE=${TORCH_DTYPE:-per model} LORA_DTYPE=${LORA_DTYPE}"
log "no_calib models: GPU 0: ${GPU0_MODELS:-<none>}; GPU 1: ${GPU1_MODELS:-<none>}"
log "LoRA microbatch ${lora_micro} x ${lora_accum} accumulation"
log "LM_EVAL=${LM_EVAL} (batch size ${LM_EVAL_BATCH_SIZE}); Stage 3 teacher on ${LORA_TEACHER_DEVICE}"
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
# With LORA_TEACHER_DEVICE=cpu the CE+KL teacher runs on these cores; split
# them between lanes.
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
