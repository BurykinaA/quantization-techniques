# Multi-architecture ADC transfer

This runner evaluates the same protocol on:

- `meta-llama/Llama-3.2-1B`
- `Qwen/Qwen2.5-1.5B`
- `HuggingFaceTB/SmolLM2-1.7B`
- `TinyLlama/TinyLlama_v1.1`

The historical `run_perplexity_all_models.sh` and
`runs/measure_perplexity.py` remain unchanged.

## Fixed full-run protocol

- BF16 reference
- W4 per-channel symmetric weights
- A4 signed symmetric activations
- signed 8-bit ADC, `k=16`, `M=256`
- FlatQuant Stage A: 1024 samples, 30 epochs, calibration batch size 16,
  MLP diagonal training
- propagated Stage B: 10 epochs, `alpha=0.5`, attention diagonal training
- each layer restores the checkpoint with the best fixed-batch validation
  objective; transform singular values are bounded to prevent late-layer drift
- ADC scale calibration batch size 4
- post-ADC LoRA: rank 4, all seven projections, 5 epochs, CE + KL,
  effective batch size 4. Qwen, SmolLM2, and TinyLlama use microbatch 2 with
  two gradient-accumulation steps.
- WikiText-2 `test` and the existing C4 `test` to `validation` mapping
- context 2048, stride 1024, 1000 C4 samples
- downstream: HellaSwag, MMLU, WinoGrande, ARC-Easy, ARC-Challenge,
  PIQA, OpenBookQA, and BoolQ; MMLU is 5-shot and all other tasks are 0-shot

`--enforce_transfer_quant_config` makes the quantized run fail early if its
signed W4A4/ADC/LoRA settings differ from this protocol.

## Remote environment

From the repository root, activate the configured CUDA environment and make
sure the existing project dependencies plus `pytest` and
`lm-evaluation-harness` are installed. Llama-3.2 also requires an accepted
Hugging Face license and a token available to `transformers`.

Set `PYTHON_BIN` if the remote interpreter is not `python3`.

## Remote checks

Run these only on the configured remote host:

```bash
cd /path/to/quantization-techniques

python3 -m py_compile \
  ADC/llama/core/flat_quant.py \
  ADC/llama/runs/llama_smooth_quant_adc_ptq.py \
  ADC/llama/summarize_multi_arch_transfer.py

bash -n ADC/llama/run_multi_arch_transfer.sh

python3 -m pytest ADC/llama/tests/test_multi_arch_flat_quant.py -q
```

## Smoke batch

Smoke mode runs model loading, one short FlatQuant pass, Stage B, ADC
conversion/calibration, pre-LoRA perplexity, one short LoRA pass, and final
perplexity for every architecture. Perplexity is capped at eight windows per
dataset, the full downstream suite is intentionally skipped, and model weights
are not saved. Logs, compact diagnostics, and the shared JSON are still written.

```bash
SMOKE=1 SKIP_COMPLETED=1 \
  bash ADC/llama/run_multi_arch_transfer.sh
```

To retry one architecture:

```bash
SMOKE=1 ONLY=smollm2_17b SKIP_COMPLETED=0 \
  bash ADC/llama/run_multi_arch_transfer.sh
```

Accepted `ONLY` keys are `llama32_1b`, `qwen25_15b`, `smollm2_17b`, and
`tinyllama_11b`; comma-separated keys and full model IDs are also accepted.

## Full resumable CUDA batch

```bash
SKIP_COMPLETED=1 \
  bash ADC/llama/run_multi_arch_transfer.sh
```

Override artifact locations if needed:

```bash
RESULTS_JSON=/remote/results/multi_arch_transfer.json \
CHECKPOINT_ROOT=/remote/checkpoints/adc_transfer \
LOG_ROOT=/remote/logs/adc_transfer \
SKIP_COMPLETED=1 \
  bash ADC/llama/run_multi_arch_transfer.sh
```

Each model produces one BF16 record and one quantized record. The quantized
record includes bypass, pre-LoRA ADC-PTQ, and post-LoRA ADC metrics. A run is
skipped only when the shared JSON contains a successful record with the same
run name.

To compute the full digital W4A4 PTQ baseline for TinyLlama without repeating
FlatQuant training or running LoRA:

```bash
ONLY=tinyllama_11b INT4_ADC_OFF_ONLY=1 SKIP_COMPLETED=1 \
  bash ADC/llama/run_multi_arch_transfer.sh
```

This loads
`transfer_results/checkpoints/tinyllama_11b/adc_transfer/flat_quant_transforms.pt`,
skips both FlatQuant stages, recalibrates the fixed integer scales, leaves W4A4
weight/activation quantization enabled, and bypasses only the ADC floor/clamp
operation. It then runs complete WikiText-2/C4 sliding-window perplexity and all
eight downstream tasks. LoRA rank is fixed to zero and model serialization is
disabled. The result is stored separately as
`tinyllama_11b_int4_ptq_adc_off`, with `ppl_int4_ptq_wikitext2`,
`ppl_int4_ptq_c4`, and `downstream_int4_ptq` fields.

If the saved transforms are outside the standard checkpoint root:

```bash
ONLY=tinyllama_11b INT4_ADC_OFF_ONLY=1 SKIP_COMPLETED=0 \
INT4_ADC_OFF_TRANSFORMS_PATH=/remote/checkpoints/flat_quant_transforms.pt \
  bash ADC/llama/run_multi_arch_transfer.sh
```

If a run fails after `flat_quant_transforms.pt` was written, rerunning the same
command reloads those final transforms and skips both FlatQuant stages. When the
old log contains the complete pre-LoRA PPL and downstream results, the runner
copies and reuses them instead of repeating the long MMLU evaluation.

Every newly trained Stage A is saved immediately, before Stage B starts:

```text
<output_dir>/flat_quant_transforms_stage_a.pt
```

To restart at Stage B without repeating Stage A:

```bash
ONLY=tinyllama_11b FQ_START_STAGE_B=1 SKIP_COMPLETED=0 \
  bash ADC/llama/run_multi_arch_transfer.sh
```

For a Stage A checkpoint outside the runner's standard output directory:

```bash
ONLY=tinyllama_11b FQ_START_STAGE_B=1 \
FQ_STAGE_A_PATH=/remote/checkpoints/flat_quant_transforms_stage_a.pt \
SKIP_COMPLETED=0 \
  bash ADC/llama/run_multi_arch_transfer.sh
```

Use `FORCE_FQ_RETRAIN=1` to ignore an existing final transform checkpoint and
train a fresh Stage A.

To compare saved Stage A and Stage B checkpoints without retraining, LoRA, or
downstream evaluation, run the diagnostic mode for each checkpoint:

```bash
ONLY=smollm2_17b FQ_DIAGNOSTIC_STAGE=stage_a DIAGNOSTIC_WINDOWS=8 \
  bash ADC/llama/run_multi_arch_transfer.sh

ONLY=smollm2_17b FQ_DIAGNOSTIC_STAGE=stage_b DIAGNOSTIC_WINDOWS=8 \
  bash ADC/llama/run_multi_arch_transfer.sh
```

Each run reports four WikiText-2 checkpoints: post-FlatQuant reparameterization,
post-ADC replacement in full bypass mode, calibrated W4A4 with ADC bypassed, and
calibrated W4A4 with ADC enabled. Diagnostic outputs use separate directories
and never overwrite the saved Stage A or Stage B transforms.

The save-and-resume path can be checked quickly with TinyLlama. The first
command creates a new Stage A checkpoint using two samples; the second command
loads that checkpoint, skips Stage A, and runs the one-epoch smoke Stage B:

```bash
SMOKE=1 ONLY=tinyllama_11b FORCE_FQ_RETRAIN=1 SKIP_COMPLETED=0 \
  bash ADC/llama/run_multi_arch_transfer.sh

SMOKE=1 ONLY=tinyllama_11b FQ_START_STAGE_B=1 SKIP_COMPLETED=0 \
  bash ADC/llama/run_multi_arch_transfer.sh
```

The first Qwen transforms produced before best-epoch selection are intentionally
ignored. Qwen retrains into `adc_transfer_best_epoch_v2`. Before the expensive
downstream suite and LoRA, the full runner aborts if pre-LoRA perplexity exceeds
500, so a collapsed PTQ checkpoint cannot consume another multi-hour evaluation.

The LoRA microbatch can be reduced further while retaining the fixed effective
batch size 4:

```bash
ONLY=qwen25_15b SKIP_COMPLETED=1 \
LORA_MICROBATCH_SIZE=1 LORA_GRADIENT_ACCUMULATION_STEPS=4 \
  bash ADC/llama/run_multi_arch_transfer.sh
```

## Validate and format results

The summarizer fails if any of the expected 16 rows, task metrics, perplexities,
or fixed protocol fields are missing. It recomputes the eight-task mean and
writes copy-ready Markdown and LaTeX rows.

```bash
python3 ADC/llama/summarize_multi_arch_transfer.py \
  ADC/llama/transfer_results/multi_arch_transfer.json \
  --markdown-output ADC/llama/transfer_results/table.md \
  --latex-output ADC/llama/transfer_results/table_rows.tex
```

After the complete JSON and generated rows are available, use them to update
the article. Do not add placeholder values before the remote runs finish.
