# Multi-architecture ADC transfer

This runner evaluates the same protocol on:

- `meta-llama/Llama-3.2-1B`
- `Qwen/Qwen2.5-1.5B`
- `allenai/OLMo-1B-hf`
- `TinyLlama/TinyLlama_v1.1`

The historical `run_perplexity_all_models.sh` and
`runs/measure_perplexity.py` remain unchanged.

## Fixed full-run protocol

- BF16 reference
- W4 per-channel symmetric weights
- A4 signed symmetric activations
- signed 8-bit ADC, `k=16`, `M=256`
- FlatQuant: 1024 samples, 30 epochs
- propagated Stage B: 10 epochs, `alpha=0.5`
- post-ADC LoRA: rank 4, all seven projections, 5 epochs, CE + KL
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
dataset, and the full downstream suite is intentionally skipped.

```bash
SMOKE=1 SKIP_COMPLETED=1 \
  bash ADC/llama/run_multi_arch_transfer.sh
```

To retry one architecture:

```bash
SMOKE=1 ONLY=qwen25_15b SKIP_COMPLETED=0 \
  bash ADC/llama/run_multi_arch_transfer.sh
```

Accepted `ONLY` keys are `llama32_1b`, `qwen25_15b`, `olmo_1b`, and
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

## Validate and format results

The summarizer fails if any of the expected 12 rows, task metrics, perplexities,
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
