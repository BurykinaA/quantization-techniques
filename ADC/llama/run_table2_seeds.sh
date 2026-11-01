#!/usr/bin/env bash
# Table 2 retrained from scratch with several seeds and midpoint ADC
# reconstruction. For every (model, seed) a lane runs the full ADC-Quant
# protocol of run_multi_arch_transfer.sh (run_adc_transfer): FlatQuant
# Stage A, the converter-aware Stage B, ADC scale calibration, converter-on
# perplexity and the eight downstream tasks of the Stages 1-2 backbone, the
# Stage 3 residual (rank 4, all seven projections, CE + 0.5 KL, 5 epochs),
# and perplexity and downstream again. With ADC_MIDPOINT=1 (default) every
# converter in the run, the Stage A/B calibration loss included, reads
# non-saturated codes at (q + 0.5) * delta; ADC_MIDPOINT=0 keeps the paper's
# floor reading.
#
# The seed sets the random orthogonal initialization of the Kronecker
# transforms and of the adapters (--seed) and, with DATA_SEED=1, the
# residual's 256 calibration sequences (--lora_data_seed). The FlatQuant
# calibration data is the same for every seed.
#
# Checkpoints per run, under checkpoints/<run>/: the Stage A transforms
# (flat_quant_transforms_stage_a.pt), the final Stage 1-2 transforms
# (flat_quant_transforms.pt), the adapters (lora_weights.pt) and, during
# Stage A/B, a progress checkpoint per completed layer. Rerunning the same
# command resumes from whatever exists and reuses the pre-Stage 3 metrics of
# the old log; a failed run is retried MAX_ATTEMPTS times before the lane
# moves on. SAVE_MODEL=1 also saves the merged model (save_pretrained).
#
# Lanes: GPU 0 runs GPU0_MODELS x GPU0_SEEDS, GPU 1 runs GPU1_MODELS x
# GPU1_SEEDS, in parallel and seed-major (every model's first seed before any
# second seed), so a complete one-seed table exists early. GPU0_JOBS /
# GPU1_JOBS ("key:seed ..." lists) replace a lane's product with an explicit
# order. After every run the lane appends the run's headline numbers to
# results.log and rebuilds summary.md and table_rows.tex.
#
# See ADC/llama/TABLE2_SEEDS.md.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
SMOKE="${SMOKE:-0}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
SEEDS="${SEEDS:-42 43 44}"
# SmolLM2 is the slowest model and gets a lane of its own.
GPU0_MODELS="${GPU0_MODELS-smollm2_17b}"
GPU1_MODELS="${GPU1_MODELS-llama32_1b tinyllama_11b}"
GPU0_SEEDS="${GPU0_SEEDS:-${SEEDS}}"
GPU1_SEEDS="${GPU1_SEEDS:-${SEEDS}}"
GPU0_JOBS="${GPU0_JOBS-}"
GPU1_JOBS="${GPU1_JOBS-}"
# 1: every converter in the run reads non-saturated codes at (q + 0.5) * delta,
# the FlatQuant Stage A/B calibration loss included. 0: the paper's floor.
ADC_MIDPOINT="${ADC_MIDPOINT:-1}"
# 1: --lora_data_seed <seed> permutes the residual's calibration pool, so the
# 256 training sequences change with the seed. 0: the Table 2 data every time.
DATA_SEED="${DATA_SEED:-1}"
# One dtype for every model (the transfer protocol). Table 2's Llama rows were
# float16; the Table 2 SmolLM2 and TinyLlama rows bfloat16.
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
LORA_DTYPE="${LORA_DTYPE:-float32}"
LM_EVAL="${LM_EVAL:-1}"
# lm-evaluation-harness batch size; batch 1 takes about a day per pass.
LM_EVAL_BATCH_SIZE="${LM_EVAL_BATCH_SIZE:-16}"
LORA_TEACHER_DEVICE="${LORA_TEACHER_DEVICE:-cuda}"
# Physical microbatch x accumulation, product 4 as in every Table 2 run.
LORA_MICROBATCH_SIZE="${LORA_MICROBATCH_SIZE:-2}"
LORA_GRADIENT_ACCUMULATION_STEPS="${LORA_GRADIENT_ACCUMULATION_STEPS:-2}"
# Abort before the downstream tasks and Stage 3 when the calibrated backbone's
# converter-on perplexity exceeds this value; empty disables the guard.
PRE_LORA_PPL_THRESHOLD="${PRE_LORA_PPL_THRESHOLD-500}"
# Attempts per run before the lane moves on; a retry resumes from the
# checkpoints the failed attempt left behind.
MAX_ATTEMPTS="${MAX_ATTEMPTS:-2}"
# 1: also save the merged model with save_pretrained (a few GB per run).
SAVE_MODEL="${SAVE_MODEL:-0}"
# 1: after each run, evaluate this seed's transforms with the converter
# bypassed (the "FlatQuant, no ADC" perplexity of this seed; no downstream).
NO_ADC_EVAL="${NO_ADC_EVAL:-0}"
TABLE2_ROOT="${TABLE2_ROOT:-${SCRIPT_DIR}/table2_seeds_results}"

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

# ---------------------------------------------------------------------------
# Validation

for flag in SMOKE DRY_RUN SKIP_COMPLETED ADC_MIDPOINT DATA_SEED LM_EVAL SAVE_MODEL NO_ADC_EVAL; do
  [[ "${!flag}" =~ ^[01]$ ]] || die "${flag} must be 0 or 1, got '${!flag}'"
done
[[ "${MAX_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]] \
  || die "MAX_ATTEMPTS must be a positive integer, got '${MAX_ATTEMPTS}'"
[[ "${LM_EVAL_BATCH_SIZE}" =~ ^(auto|[1-9][0-9]*)$ ]] \
  || die "LM_EVAL_BATCH_SIZE must be a positive integer or auto, got '${LM_EVAL_BATCH_SIZE}'"
[[ "${TORCH_DTYPE}" =~ ^(float16|bfloat16|float32)$ ]] \
  || die "TORCH_DTYPE must be float16, bfloat16 or float32, got '${TORCH_DTYPE}'"
[[ "${LORA_DTYPE}" =~ ^(float32|bfloat16|float16)$ ]] \
  || die "LORA_DTYPE must be float32, bfloat16 or float16, got '${LORA_DTYPE}'"
[[ -z "${PRE_LORA_PPL_THRESHOLD}" || "${PRE_LORA_PPL_THRESHOLD}" =~ ^[0-9]+(\.[0-9]+)?$ ]] \
  || die "PRE_LORA_PPL_THRESHOLD must be empty or a number, got '${PRE_LORA_PPL_THRESHOLD}'"
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
for key in ${GPU0_MODELS} ${GPU1_MODELS}; do
  model_id_for "${key}" >/dev/null || die "unknown model key '${key}'"
done
for job in ${GPU0_JOBS} ${GPU1_JOBS}; do
  [[ "${job}" =~ ^[a-z0-9_]+:[0-9]+$ ]] || die "jobs must be key:seed, got '${job}'"
  model_id_for "${job%%:*}" >/dev/null || die "unknown model key '${job%%:*}' in job '${job}'"
done

# ---------------------------------------------------------------------------
# Output layout

if [[ "${ADC_MIDPOINT}" == "1" ]]; then
  CONVENTION="midpoint"
else
  CONVENTION="floor"
  TABLE2_ROOT="${TABLE2_ROOT}_floor"
fi
if [[ "${TORCH_DTYPE}" != "bfloat16" ]]; then
  TABLE2_ROOT="${TABLE2_ROOT}_${TORCH_DTYPE}"
fi
if [[ "${LORA_DTYPE}" != "float32" ]]; then
  TABLE2_ROOT="${TABLE2_ROOT}_lora_${LORA_DTYPE}"
fi
OUTPUT_ROOT="${TABLE2_ROOT}"
RUN_SUFFIX=""
if [[ "${SMOKE}" == "1" ]]; then
  OUTPUT_ROOT="${TABLE2_ROOT}/smoke"
  RUN_SUFFIX="_smoke"
fi
RESULTS_DIR="${OUTPUT_ROOT}/results"
LOG_ROOT="${OUTPUT_ROOT}/logs"
RESULTS_LOG="${OUTPUT_ROOT}/results.log"

# Lane job lists: explicit GPUn_JOBS, or seed-major products of models and seeds.
lane_jobs() {
  local jobs="$1" models="$2" seeds="$3"
  if [[ -n "${jobs// /}" ]]; then
    printf '%s\n' ${jobs}
    return 0
  fi
  local seed key
  for seed in ${seeds}; do
    for key in ${models}; do
      printf '%s:%s\n' "${key}" "${seed}"
    done
  done
}
GPU0_JOB_LIST="$(lane_jobs "${GPU0_JOBS}" "${GPU0_MODELS}" "${GPU0_SEEDS}")"
GPU1_JOB_LIST="$(lane_jobs "${GPU1_JOBS}" "${GPU1_MODELS}" "${GPU1_SEEDS}")"
LANES=()
[[ -n "${GPU0_JOB_LIST// /}" ]] && LANES+=("0")
[[ -n "${GPU1_JOB_LIST// /}" ]] && LANES+=("1")
(( ${#LANES[@]} > 0 )) || die "nothing to run: both lanes are empty"

# ---------------------------------------------------------------------------
# Runner arguments: the Table 2 protocol of run_multi_arch_transfer.sh
# (run_adc_transfer), trained from scratch.

calibration_batches=100
adc_calibration_batch_size=4
calibration_length=512
max_eval_samples=1000
max_length=2048
stride=1024
fq_nsamples=1024
fq_cali_bsz=16
fq_epochs=30
fq_stage_b_epochs=10
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
  fq_stage_b_epochs=1
  lora_nsamples=2
  lora_epochs=1
  lora_micro=1
  lora_accum=1
  smoke_args=(--max_eval_windows 8)
fi

FQ_ARGS=(
  --preprocess_method flat_quant
  --fq_w_bits 4
  --fq_a_bits 4
  --fq_nsamples "${fq_nsamples}"
  --fq_cali_bsz "${fq_cali_bsz}"
  --fq_epochs "${fq_epochs}"
  --fq_diag_mlp
  --fq_stage_b_epochs "${fq_stage_b_epochs}"
  --fq_stage_b_prop_alpha 0.5
  --fq_stage_b_diag_attn
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

# ---------------------------------------------------------------------------
# Helpers

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

# True when a log of an earlier attempt holds the complete pre-Stage 3
# metrics (converter-on perplexity and, with LM_EVAL=1, the eight tasks), so
# the runner can take them from the log instead of measuring again.
has_complete_pre_lora_metrics() {
  local log_path="$1"
  "${PYTHON_BIN}" - "${log_path}" "${LM_EVAL}" <<'PY'
import sys

path, lm_eval = sys.argv[1:3]
with open(path, encoding="utf-8", errors="replace") as handle:
    text = handle.read()

datasets = ("WIKITEXT2", "C4")
tasks = (
    "hellaswag", "mmlu", "winogrande", "arc_easy",
    "arc_challenge", "piqa", "openbookqa", "boolq",
) if lm_eval == "1" else ()
complete = (
    all(f"BEFORE LoRA -> {dataset} Perplexity:" in text for dataset in datasets)
    and all(f"lm-eval adc_ptq/{task}:" in text for task in tasks)
)
raise SystemExit(0 if complete else 1)
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

# One line with the run's final numbers, appended to results.log.
append_headline() {
  local run_name="$1"
  local json_path="${RESULTS_DIR}/${run_name}.json"
  local line
  if ! line="$("${PYTHON_BIN}" - "${json_path}" "${run_name}" <<'PY'
import json
import sys

path, run_name = sys.argv[1:3]
with open(path, encoding="utf-8") as handle:
    records = json.load(handle)
record = next(
    (r for r in records if r.get("run_name") == run_name and r.get("status") == "success"),
    None,
)
if record is None:
    raise SystemExit(1)
results = record.get("results", {})


def ppl(key):
    value = results.get(key)
    return "—" if value is None else f"{value:.2f}"


def acc(key):
    value = ((results.get(key) or {}).get("mean") or {}).get("accuracy_pct")
    return "—" if value is None else f"{value:.2f}%"


print(
    f"{run_name}: Stages 1-2 Wiki {ppl('ppl_adc_prelora_wikitext2')} "
    f"C4 {ppl('ppl_adc_prelora_c4')} Avg {acc('downstream_adc_ptq')} | "
    f"Stages 1-3 Wiki {ppl('ppl_adc_wikitext2')} C4 {ppl('ppl_adc_c4')} "
    f"Avg {acc('downstream_adc_lora')}"
)
PY
)"; then
    log "could not read the final numbers of ${run_name} from ${json_path}"
    return 0
  fi
  log "RESULT ${line}"
  printf '%s %s\n' "$(date '+%F %T')" "${line}" >> "${RESULTS_LOG}"
}

# Rebuild summary.md and table_rows.tex from every finished run. Lanes
# serialize on a lock so two of them never write the files at once.
refresh_summary() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  local -a cmd=(
    "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_table2_seeds.py"
    --results-dir "${RESULTS_DIR}"
    --run-suffix "${RUN_SUFFIX}"
    --latex-convention "${CONVENTION}"
    --markdown-output "${OUTPUT_ROOT}/summary.md"
    --latex-output "${OUTPUT_ROOT}/table_rows.tex"
  )
  if command -v flock >/dev/null 2>&1; then
    flock "${OUTPUT_ROOT}/.summary.lock" "${cmd[@]}" >/dev/null \
      || log "summary failed; rerun summarize_table2_seeds.py manually"
  else
    "${cmd[@]}" >/dev/null || log "summary failed; rerun summarize_table2_seeds.py manually"
  fi
}

# ---------------------------------------------------------------------------
# Runs

# One attempt of the full protocol for (key, seed), resuming from whatever
# the checkpoint directory already holds.
run_protocol() {
  local key="$1" seed="$2" run_name="$3" output_dir="$4" log_path="$5"
  local transforms="${output_dir}/flat_quant_transforms.pt"
  local stage_a="${output_dir}/flat_quant_transforms_stage_a.pt"
  local stage_a_progress="${output_dir}/flat_quant_stage_a_progress.pt"
  local stage_b_progress="${output_dir}/flat_quant_stage_b_progress.pt"
  local lora_weights="${output_dir}/lora_weights.pt"
  local resume_log="${log_path%.log}.pre_lora_resume.log"

  local -a resume_args=()
  local -a save_args=(--fq_save_transforms)
  if [[ "${SMOKE}" != "1" ]]; then
    if [[ -f "${transforms}" ]]; then
      log "${run_name}: reusing the final Stages 1-2 transforms"
      resume_args=(--fq_reload_path "${transforms}" --fq_skip_stage_b_on_reload)
      save_args=()
      if [[ -f "${log_path}" ]] && has_complete_pre_lora_metrics "${log_path}"; then
        cp "${log_path}" "${resume_log}"
      fi
      if [[ -f "${resume_log}" ]] && has_complete_pre_lora_metrics "${resume_log}"; then
        log "${run_name}: reusing the pre-Stage 3 metrics of the previous log"
        resume_args+=(--pre_lora_metrics_log "${resume_log}")
      fi
    elif [[ -f "${stage_a}" ]]; then
      log "${run_name}: resuming Stage B from the saved Stage A transforms"
      resume_args=(
        --fq_start_stage_b_from "${stage_a}"
        --fq_stage_b_layer_checkpoint_path "${stage_b_progress}"
      )
      if [[ -f "${stage_b_progress}" ]]; then
        resume_args+=(--fq_resume_stage_b_layer_checkpoint)
      fi
    else
      resume_args=(
        --fq_layer_checkpoint_path "${stage_a_progress}"
        --fq_stage_b_layer_checkpoint_path "${stage_b_progress}"
      )
      if [[ -f "${stage_a_progress}" ]]; then
        log "${run_name}: resuming Stage A from its layer checkpoint"
        resume_args+=(--fq_resume_layer_checkpoint)
      fi
    fi
    if [[ -f "${lora_weights}" ]]; then
      log "${run_name}: reusing the saved adapters"
      resume_args+=(--lora_reload_path "${lora_weights}")
    fi
  fi
  if [[ "${SAVE_MODEL}" != "1" ]]; then
    save_args+=(--skip_model_save)
  fi

  local -a seed_args=(--seed "${seed}")
  if [[ "${DATA_SEED}" == "1" ]]; then
    seed_args+=(--lora_data_seed "${seed}")
  fi
  local -a convention_args=()
  if [[ "${ADC_MIDPOINT}" == "1" ]]; then
    convention_args=(--adc_midpoint)
  fi
  local -a protocol_args=()
  if [[ "${SMOKE}" != "1" ]]; then
    if [[ -n "${PRE_LORA_PPL_THRESHOLD}" ]]; then
      protocol_args+=(--pre_lora_ppl_threshold "${PRE_LORA_PPL_THRESHOLD}")
    fi
    # The transfer protocol check accepts only bfloat16.
    if [[ "${TORCH_DTYPE}" == "bfloat16" ]]; then
      protocol_args+=(--enforce_transfer_quant_config)
    fi
  fi
  local -a lm_eval_args=()
  if [[ "${LM_EVAL}" == "1" ]]; then
    lm_eval_args=(
      --run_lm_eval
      --lm_eval_tasks "${LM_EVAL_TASKS[@]}"
      --lm_eval_batch_size "${LM_EVAL_BATCH_SIZE}"
    )
    if [[ "${SMOKE}" == "1" ]]; then
      lm_eval_args+=(--lm_eval_limit 4)
    fi
  fi

  run_step "${run_name}" "${log_path}" \
    "${PYTHON_BIN}" -m ADC.llama.runs.llama_smooth_quant_adc_ptq \
    --model_name "$(model_id_for "${key}")" \
    --torch_dtype "${TORCH_DTYPE}" \
    --output_dir "${output_dir}" \
    --no_date_suffix \
    --wandb_run_name "${run_name}" \
    --results_json_path "${RESULTS_DIR}/${run_name}.json" \
    "${FQ_ARGS[@]}" \
    "${HW_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${STAGE3_ARGS[@]}" \
    "${seed_args[@]}" \
    ${save_args[@]+"${save_args[@]}"} \
    ${convention_args[@]+"${convention_args[@]}"} \
    ${protocol_args[@]+"${protocol_args[@]}"} \
    ${lm_eval_args[@]+"${lm_eval_args[@]}"} \
    ${resume_args[@]+"${resume_args[@]}"}
}

# This seed's Stages 1-2 transforms with the converter bypassed (W4A4 kept):
# the "FlatQuant, no ADC" perplexity of the same transforms. No downstream.
run_noadc() {
  local key="$1" seed="$2" transforms="$3"
  local run_name="${key}_${CONVENTION}_seed${seed}_noadc${RUN_SUFFIX}"
  if is_completed "${run_name}"; then
    log "skip completed ${run_name}"
    return 0
  fi
  if [[ "${DRY_RUN}" != "1" && ! -f "${transforms}" ]]; then
    log "FAIL  ${run_name}: ${transforms} not found"
    return 1
  fi
  local output_dir="${OUTPUT_ROOT}/checkpoints/${run_name}"
  if [[ "${DRY_RUN}" != "1" ]]; then
    mkdir -p "${output_dir}"
  fi
  run_step "${run_name}" "${LOG_ROOT}/${run_name}.log" \
    "${PYTHON_BIN}" -m ADC.llama.runs.llama_smooth_quant_adc_ptq \
    --model_name "$(model_id_for "${key}")" \
    --torch_dtype "${TORCH_DTYPE}" \
    --output_dir "${output_dir}" \
    --no_date_suffix \
    --wandb_run_name "${run_name}" \
    --results_json_path "${RESULTS_DIR}/${run_name}.json" \
    "${FQ_ARGS[@]}" \
    --fq_reload_path "${transforms}" \
    --fq_skip_stage_b_on_reload \
    "${HW_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    --lora_rank 0 \
    --adc_off_eval_only \
    --skip_model_save
}

run_job() {
  local key="$1" seed="$2"
  local run_name="${key}_${CONVENTION}_seed${seed}${RUN_SUFFIX}"
  local output_dir="${OUTPUT_ROOT}/checkpoints/${run_name}"
  local log_path="${LOG_ROOT}/${run_name}.log"
  if is_completed "${run_name}"; then
    log "skip completed ${run_name}"
  else
    if [[ "${DRY_RUN}" != "1" ]]; then
      mkdir -p "${output_dir}"
    fi
    local attempt status=1
    for (( attempt = 1; attempt <= MAX_ATTEMPTS; attempt++ )); do
      if (( attempt > 1 )); then
        log "attempt ${attempt}/${MAX_ATTEMPTS} of ${run_name}"
      fi
      if run_protocol "${key}" "${seed}" "${run_name}" "${output_dir}" "${log_path}"; then
        status=0
        break
      else
        status=$?
      fi
    done
    if (( status != 0 )); then
      log "GIVING UP on ${run_name} after ${MAX_ATTEMPTS} attempt(s)"
      return "${status}"
    fi
    if [[ "${DRY_RUN}" != "1" ]]; then
      append_headline "${run_name}"
    fi
  fi
  local failed=0
  if [[ "${NO_ADC_EVAL}" == "1" ]]; then
    run_noadc "${key}" "${seed}" "${output_dir}/flat_quant_transforms.pt" || failed=1
  fi
  refresh_summary
  return "${failed}"
}

run_lane() {
  local gpu="$1" jobs="$2"
  local failed=0 job
  if [[ "${DRY_RUN}" != "1" ]] && command -v flock >/dev/null 2>&1; then
    exec 9>"${OUTPUT_ROOT}/.run_gpu${gpu}.lock"
    if ! flock -n 9; then
      log "FAIL  GPU ${gpu}: another run_table2_seeds.sh lane already uses this GPU in ${OUTPUT_ROOT}"
      return 1
    fi
  fi
  for job in ${jobs}; do
    run_job "${job%%:*}" "${job##*:}" || failed=1
  done
  log "lane finished"
  return "${failed}"
}

# ---------------------------------------------------------------------------

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${RESULTS_DIR}" "${LOG_ROOT}"
fi

log "CONVENTION=${CONVENTION} SEEDS=${SEEDS} DATA_SEED=${DATA_SEED} SMOKE=${SMOKE} DRY_RUN=${DRY_RUN} SKIP_COMPLETED=${SKIP_COMPLETED} MAX_ATTEMPTS=${MAX_ATTEMPTS} PRE_LORA_PPL_THRESHOLD=${PRE_LORA_PPL_THRESHOLD:-off} SAVE_MODEL=${SAVE_MODEL} NO_ADC_EVAL=${NO_ADC_EVAL}"
log "GPU 0 jobs: $(printf '%s ' ${GPU0_JOB_LIST})"
log "GPU 1 jobs: $(printf '%s ' ${GPU1_JOB_LIST})"
log "TORCH_DTYPE=${TORCH_DTYPE} LORA_DTYPE=${LORA_DTYPE}; LoRA microbatch ${lora_micro} x ${lora_accum}; teacher on ${LORA_TEACHER_DEVICE}"
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
    lane_jobs_list="${GPU0_JOB_LIST}"
  else
    lane_jobs_list="${GPU1_JOB_LIST}"
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    LANE_LABEL="gpu${gpu}" run_lane "${gpu}" "${lane_jobs_list}" || any_failed=1
  else
    (
      export CUDA_VISIBLE_DEVICES="${gpu}"
      export OMP_NUM_THREADS="${THREADS_PER_LANE}"
      export MKL_NUM_THREADS="${THREADS_PER_LANE}"
      LANE_LABEL="gpu${gpu}"
      run_lane "${gpu}" "${lane_jobs_list}"
    ) &
    lane_pids+=("$!")
  fi
done

for pid in ${lane_pids[@]+"${lane_pids[@]}"}; do
  wait "${pid}" || any_failed=1
done

if [[ "${DRY_RUN}" != "1" ]]; then
  refresh_summary
  if [[ -f "${OUTPUT_ROOT}/summary.md" ]]; then
    cat "${OUTPUT_ROOT}/summary.md"
  fi
  printf 'finished %s; failed=%d\nsummary: %s\nlatex rows: %s\nheadlines: %s\n' \
    "$(date '+%F %T')" "${any_failed}" "${OUTPUT_ROOT}/summary.md" \
    "${OUTPUT_ROOT}/table_rows.tex" "${RESULTS_LOG}" > "${OUTPUT_ROOT}/status.txt"
fi
if (( any_failed )); then
  log "Some runs FAILED; see the FAIL lines above"
fi
log "All lanes finished. Summary: ${OUTPUT_ROOT}/summary.md; LaTeX rows: ${OUTPUT_ROOT}/table_rows.tex; headlines: ${RESULTS_LOG}"
exit "${any_failed}"
