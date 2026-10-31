#!/usr/bin/env bash
# Seed variance of Stage 3 (asked for by reviewers dEPN and AI): the paper's
# residual (rank 4, all seven projections, CE + 0.5 KL at T = 2, 5 epochs)
# retrained with several seeds on the Table 2 Stages 1-2 backbone, with
# converter-on perplexity and the eight downstream tasks per seed. TinyLlama's
# Table 2 downstream gain is only +0.39, so its spread across seeds decides
# whether that gain is real.
#
# Per seed the run passes --seed <s> (adapter initialization) and, with
# DATA_SEED=1 (default), --lora_data_seed <s>, which permutes the residual's
# calibration pool so the 256 training sequences and their order change too.
# DATA_SEED=0 keeps the Table 2 data (the first 64 batches in corpus order).
#
# The backbone is reloaded from the transfer run's final Stage 1-2 transforms
# (<key>/adc_transfer/flat_quant_transforms.pt, override TRANSFORMS_<KEY>).
# Llama has no saved Table 2 transforms; set TRANSFORMS_LLAMA32_1B, e.g. to
# the v6 transforms used by run_adc_midpoint.sh (Stages 1-2 floor 26.99/47.65
# against 26.66/45.62 in Table 2).
#
# Lanes: GPU 0 runs GPU0_MODELS x GPU0_SEEDS, GPU 1 runs GPU1_MODELS x
# GPU1_SEEDS, in parallel. Each lane takes its own lock, so a second
# invocation may use the other GPU with the same output directory.
#
# See ADC/llama/STAGE3_SEEDS.md.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
SMOKE="${SMOKE:-0}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
SEEDS="${SEEDS:-42 43 44}"
GPU0_MODELS="${GPU0_MODELS-}"
GPU1_MODELS="${GPU1_MODELS-tinyllama_11b}"
GPU0_SEEDS="${GPU0_SEEDS:-${SEEDS}}"
GPU1_SEEDS="${GPU1_SEEDS:-${SEEDS}}"
read -r -a SEED_LIST <<< "${SEEDS}"
# The run with this seed also measures the downstream tasks on the Stages 1-2
# backbone before Stage 3. That backbone is the same for every seed, so it is
# measured once; the other seeds pass --no_pre_lora_lm_eval.
PRE_LORA_LM_EVAL_SEED="${PRE_LORA_LM_EVAL_SEED:-${SEED_LIST[0]:-}}"
DATA_SEED="${DATA_SEED:-1}"
# Empty: the model's Table 2 dtype (Llama float16, the transfer models
# bfloat16). A value runs every model in that dtype and suffixes the output
# directory.
TORCH_DTYPE="${TORCH_DTYPE:-}"
# dtype of the residual's weights, arithmetic and optimizer state; Table 2
# used float32 adapters.
LORA_DTYPE="${LORA_DTYPE:-float32}"
LM_EVAL="${LM_EVAL:-1}"
# lm-evaluation-harness batch size; the tiled simulation at batch 1 is
# kernel-launch bound (about a day per model), batch 16 is ~10x faster.
LM_EVAL_BATCH_SIZE="${LM_EVAL_BATCH_SIZE:-16}"
LORA_TEACHER_DEVICE="${LORA_TEACHER_DEVICE:-cuda}"
# Physical microbatch x accumulation, product 4 as in every Table 2 run. The
# transfer runs used 2 x 2; Llama's 4 x 1 no longer fits with the converter on.
LORA_MICROBATCH_SIZE="${LORA_MICROBATCH_SIZE:-2}"
LORA_GRADIENT_ACCUMULATION_STEPS="${LORA_GRADIENT_ACCUMULATION_STEPS:-2}"
# REUSE_LORA=1 reloads a run's saved adapters instead of training them, to
# repeat or extend an evaluation.
REUSE_LORA="${REUSE_LORA:-0}"
TRANSFER_CHECKPOINT_ROOT="${TRANSFER_CHECKPOINT_ROOT:-${SCRIPT_DIR}/transfer_results/checkpoints}"

SEEDS_ROOT="${SEEDS_ROOT:-${SCRIPT_DIR}/stage3_seeds_results}"
if [[ -n "${TORCH_DTYPE}" ]]; then
  SEEDS_ROOT="${SEEDS_ROOT}_${TORCH_DTYPE}"
fi
if [[ "${LORA_DTYPE}" != "float32" ]]; then
  SEEDS_ROOT="${SEEDS_ROOT}_lora_${LORA_DTYPE}"
fi
OUTPUT_ROOT="${SEEDS_ROOT}"
RUN_SUFFIX=""
if [[ "${SMOKE}" == "1" ]]; then
  OUTPUT_ROOT="${SEEDS_ROOT}/smoke"
  RUN_SUFFIX="_smoke"
fi
RESULTS_DIR="${OUTPUT_ROOT}/results"
LOG_ROOT="${OUTPUT_ROOT}/logs"

LORA_TARGETS=(q_proj k_proj v_proj o_proj gate_proj up_proj down_proj)
LM_EVAL_TASKS=(hellaswag mmlu winogrande arc_easy arc_challenge piqa openbookqa boolq)

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

# dtype of the model's Table 2 runs unless TORCH_DTYPE overrides it.
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

# Stage 1-2 transforms of the model's Table 2 row, or TRANSFORMS_<KEY>.
transforms_for() {
  local key="$1"
  local override_name
  override_name="TRANSFORMS_$(printf '%s' "${key}" | tr '[:lower:]' '[:upper:]')"
  local transforms="${!override_name:-}"
  if [[ -z "${transforms}" ]]; then
    if [[ "${key}" == "llama32_1b" ]]; then
      log "FAIL  ${key}: no Table 2 Stages 1-2 transforms are saved for Llama; set ${override_name} (run_adc_midpoint.sh used checkpoints/overnight_int4_v6_20260406/lora_r4_down_20260406/flat_quant_transforms.pt)" >&2
      return 1
    fi
    transforms="${TRANSFER_CHECKPOINT_ROOT}/${key}/adc_transfer/flat_quant_transforms.pt"
  fi
  if [[ "${DRY_RUN}" != "1" && ! -f "${transforms}" ]]; then
    log "FAIL  ${key}: ${transforms} not found (set ${override_name})" >&2
    return 1
  fi
  printf '%s\n' "${transforms}"
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
  if [[ -f "${log_path}" ]]; then
    mv "${log_path}" "${log_path}.$(date '+%Y%m%d_%H%M%S')"
  fi
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
# Runner arguments: the Table 2 protocol of run_multi_arch_transfer.sh
# (run_adc_transfer), with the Stage 1-2 transforms reloaded.

calibration_batches=100
adc_calibration_batch_size=4
calibration_length=512
max_eval_samples=1000
max_length=2048
stride=1024
fq_nsamples=1024
fq_cali_bsz=16
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
  lora_nsamples=2
  lora_epochs=1
  lora_micro=1
  lora_accum=1
  smoke_args=(--max_eval_windows 8)
fi

# Stage A/B settings are recorded only: the reloaded transforms are final and
# --fq_skip_stage_b_on_reload skips Stage B. fq_nsamples / fq_cali_bsz also
# set the residual's data budget (64 batches of 4 sequences).
BACKBONE_ARGS=(
  --preprocess_method flat_quant
  --fq_w_bits 4
  --fq_a_bits 4
  --fq_nsamples "${fq_nsamples}"
  --fq_cali_bsz "${fq_cali_bsz}"
  --fq_epochs 30
  --fq_diag_mlp
  --fq_stage_b_epochs 10
  --fq_stage_b_prop_alpha 0.5
  --fq_stage_b_diag_attn
  --fq_skip_stage_b_on_reload
)

HW_ARGS=(
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
)

EVAL_ARGS=(
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

STAGE3_ARGS=(
  --lora_mode residual
  --lora_rank 4
  --lora_alpha 8
  --lora_target_modules "${LORA_TARGETS[@]}"
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

run_seed() {
  local key="$1" seed="$2"
  local run_name="${key}_stage3_seed${seed}${RUN_SUFFIX}"
  if is_completed "${run_name}"; then
    log "skip completed ${run_name}"
    return 0
  fi
  local transforms dtype
  transforms="$(transforms_for "${key}")" || return 1
  dtype="$(dtype_for "${key}")"
  local output_dir="${OUTPUT_ROOT}/checkpoints/${run_name}"
  mkdir -p "${output_dir}"

  local -a seed_args=(--seed "${seed}")
  if [[ "${DATA_SEED}" == "1" ]]; then
    seed_args+=(--lora_data_seed "${seed}")
  fi
  local -a protocol_args=()
  if [[ "${SMOKE}" != "1" ]]; then
    protocol_args+=(--pre_lora_ppl_threshold 500)
  fi
  # The transfer protocol check accepts only bfloat16.
  if [[ "${dtype}" == "bfloat16" ]]; then
    protocol_args+=(--enforce_transfer_quant_config)
  fi
  local -a lm_eval_args=()
  if [[ "${LM_EVAL}" == "1" ]]; then
    lm_eval_args=(
      --run_lm_eval
      --lm_eval_tasks "${LM_EVAL_TASKS[@]}"
      --lm_eval_batch_size "${LM_EVAL_BATCH_SIZE}"
    )
    if [[ "${seed}" != "${PRE_LORA_LM_EVAL_SEED}" ]]; then
      lm_eval_args+=(--no_pre_lora_lm_eval)
    fi
    if [[ "${SMOKE}" == "1" ]]; then
      lm_eval_args+=(--lm_eval_limit 4)
    fi
  fi
  local -a reuse_args=()
  if [[ "${REUSE_LORA}" == "1" && -f "${output_dir}/lora_weights.pt" ]]; then
    reuse_args=(--lora_reload_path "${output_dir}/lora_weights.pt")
    log "reusing saved adapters of ${run_name}"
  fi

  run_step "${run_name}" "${LOG_ROOT}/${run_name}.log" \
    "${PYTHON_BIN}" -m ADC.llama.runs.llama_smooth_quant_adc_ptq \
    --model_name "$(model_id_for "${key}")" \
    --torch_dtype "${dtype}" \
    --output_dir "${output_dir}" \
    --no_date_suffix \
    --wandb_run_name "${run_name}" \
    --results_json_path "${RESULTS_DIR}/${run_name}.json" \
    "${BACKBONE_ARGS[@]}" \
    --fq_reload_path "${transforms}" \
    "${HW_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${STAGE3_ARGS[@]}" \
    "${seed_args[@]}" \
    ${protocol_args[@]+"${protocol_args[@]}"} \
    ${lm_eval_args[@]+"${lm_eval_args[@]}"} \
    ${reuse_args[@]+"${reuse_args[@]}"}
}

run_lane() {
  local gpu="$1" models="$2" seeds="$3"
  local failed=0 key seed
  if [[ "${DRY_RUN}" != "1" ]] && command -v flock >/dev/null 2>&1; then
    exec 9>"${OUTPUT_ROOT}/.run_gpu${gpu}.lock"
    if ! flock -n 9; then
      log "FAIL  GPU ${gpu}: another run_stage3_seeds.sh lane already uses this GPU in ${OUTPUT_ROOT}"
      return 1
    fi
  fi
  for key in ${models}; do
    for seed in ${seeds}; do
      run_seed "${key}" "${seed}" || failed=1
    done
  done
  log "lane finished"
  return "${failed}"
}

# ---------------------------------------------------------------------------

for flag in SMOKE DRY_RUN SKIP_COMPLETED DATA_SEED LM_EVAL REUSE_LORA; do
  [[ "${!flag}" =~ ^[01]$ ]] || die "${flag} must be 0 or 1, got '${!flag}'"
done
[[ "${LM_EVAL_BATCH_SIZE}" =~ ^(auto|[1-9][0-9]*)$ ]] \
  || die "LM_EVAL_BATCH_SIZE must be a positive integer or auto, got '${LM_EVAL_BATCH_SIZE}'"
[[ -z "${TORCH_DTYPE}" || "${TORCH_DTYPE}" =~ ^(float16|bfloat16|float32)$ ]] \
  || die "TORCH_DTYPE must be empty, float16, bfloat16 or float32, got '${TORCH_DTYPE}'"
[[ "${LORA_DTYPE}" =~ ^(float32|bfloat16|float16)$ ]] \
  || die "LORA_DTYPE must be float32, bfloat16 or float16, got '${LORA_DTYPE}'"
[[ "${LORA_TEACHER_DEVICE}" =~ ^(cpu|cuda)$ ]] \
  || die "LORA_TEACHER_DEVICE must be cpu or cuda, got '${LORA_TEACHER_DEVICE}'"
[[ "${LORA_MICROBATCH_SIZE}" =~ ^[1-9][0-9]*$ && "${LORA_GRADIENT_ACCUMULATION_STEPS}" =~ ^[1-9][0-9]*$ ]] \
  || die "LORA_MICROBATCH_SIZE and LORA_GRADIENT_ACCUMULATION_STEPS must be positive integers"
if [[ "${SMOKE}" != "1" ]]; then
  (( LORA_MICROBATCH_SIZE * LORA_GRADIENT_ACCUMULATION_STEPS == 4 )) \
    || die "LORA_MICROBATCH_SIZE x LORA_GRADIENT_ACCUMULATION_STEPS must equal the effective batch 4"
fi
for seed in ${SEEDS} ${GPU0_SEEDS} ${GPU1_SEEDS}; do
  [[ "${seed}" =~ ^[0-9]+$ ]] || die "seeds must be non-negative integers, got '${seed}'"
done
[[ "${PRE_LORA_LM_EVAL_SEED}" =~ ^[0-9]+$ ]] \
  || die "PRE_LORA_LM_EVAL_SEED must be a seed, got '${PRE_LORA_LM_EVAL_SEED}'"
for key in ${GPU0_MODELS} ${GPU1_MODELS}; do
  model_id_for "${key}" >/dev/null || die "unknown model key '${key}'"
done
LANES=()
[[ -n "${GPU0_MODELS// /}" ]] && LANES+=("0")
[[ -n "${GPU1_MODELS// /}" ]] && LANES+=("1")
(( ${#LANES[@]} > 0 )) || die "nothing to run: both GPU0_MODELS and GPU1_MODELS are empty"

mkdir -p "${RESULTS_DIR}" "${LOG_ROOT}"

log "SEEDS=${SEEDS} DATA_SEED=${DATA_SEED} PRE_LORA_LM_EVAL_SEED=${PRE_LORA_LM_EVAL_SEED} SMOKE=${SMOKE} DRY_RUN=${DRY_RUN} SKIP_COMPLETED=${SKIP_COMPLETED} REUSE_LORA=${REUSE_LORA}"
log "GPU 0: ${GPU0_MODELS:-<idle>} x seeds ${GPU0_SEEDS}; GPU 1: ${GPU1_MODELS:-<idle>} x seeds ${GPU1_SEEDS}"
log "TORCH_DTYPE=${TORCH_DTYPE:-per model} LORA_DTYPE=${LORA_DTYPE}; LoRA microbatch ${lora_micro} x ${lora_accum}; teacher on ${LORA_TEACHER_DEVICE}"
log "LM_EVAL=${LM_EVAL} (batch size ${LM_EVAL_BATCH_SIZE})"
log "Outputs: ${OUTPUT_ROOT}"

if [[ -z "${THREADS_PER_LANE:-}" ]]; then
  cpu_count="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)"
  THREADS_PER_LANE=$(( cpu_count / ${#LANES[@]} ))
  (( THREADS_PER_LANE >= 1 )) || THREADS_PER_LANE=1
fi
log "CPU threads per lane: ${THREADS_PER_LANE}"

any_failed=0
lane_pids=()
for gpu in "${LANES[@]}"; do
  if [[ "${gpu}" == "0" ]]; then
    lane_models="${GPU0_MODELS}"
    lane_seeds="${GPU0_SEEDS}"
  else
    lane_models="${GPU1_MODELS}"
    lane_seeds="${GPU1_SEEDS}"
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    LANE_LABEL="gpu${gpu}" run_lane "${gpu}" "${lane_models}" "${lane_seeds}" || any_failed=1
  else
    (
      export CUDA_VISIBLE_DEVICES="${gpu}"
      export OMP_NUM_THREADS="${THREADS_PER_LANE}"
      export MKL_NUM_THREADS="${THREADS_PER_LANE}"
      LANE_LABEL="gpu${gpu}"
      run_lane "${gpu}" "${lane_models}" "${lane_seeds}"
    ) &
    lane_pids+=("$!")
  fi
done

for pid in ${lane_pids[@]+"${lane_pids[@]}"}; do
  wait "${pid}" || any_failed=1
done

if [[ "${DRY_RUN}" != "1" ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_stage3_seeds.py" \
    --results-dir "${RESULTS_DIR}" \
    --run-suffix "${RUN_SUFFIX}" \
    --markdown-output "${OUTPUT_ROOT}/summary.md" \
    || log "summary failed; rerun summarize_stage3_seeds.py manually"
fi
if (( any_failed )); then
  log "Some runs FAILED; see the FAIL lines above"
fi
log "All lanes finished. Summary: ${OUTPUT_ROOT}/summary.md"
exit "${any_failed}"
