# Matched no-ADC control for the post-ADC residual

`run_matched_lora_control.sh` trains the Stage 3 residual of the transfer
protocol (rank 4, alpha 8, all seven projections, CE + 0.5 KL at T=2,
lr 1e-4, 5 epochs, effective batch 4, the same WikiText-2 batches) with only
the ADC floor/clamp bypassed. Tiled W4A4 quantization stays active. Comparing
its gain with the ADC arm of `run_multi_arch_transfer.sh` separates
converter-error correction from generic adapter capacity and distillation.

## Arms

| Arm | Backbone | Converter during adapter training/eval |
| :-- | :-- | :-- |
| `same_backbone` | final Stage 1-2 transforms of the transfer run, reloaded unchanged | off |
| `flatquant` | FlatQuant Stage A recalibrated with `--fq_bypass_adc_floor`, no Stage B | off |

`same_backbone` holds the transforms fixed, so the converter is the only
difference from the ADC arm. For SmolLM2 and TinyLlama, the paper's
"FlatQuant, no ADC" row is this backbone with the ADC off. `flatquant`
follows the protocol of the Llama "FlatQuant, no ADC" row
(`run_int_only_baseline_1b.sh` on `multi-model-paper-comparison`), retrained
on the current FlatQuant code, so its pre-LoRA perplexity can differ slightly
from 14.13.

Default lanes: GPU 0 runs `llama32_1b:same_backbone llama32_1b:flatquant`,
GPU 1 runs `smollm2_17b:same_backbone tinyllama_11b:same_backbone`. Override
them with `GPU0_ARMS` / `GPU1_ARMS`, and keep all arms of one model in one lane.

Each lane runs phase 1 for all of its arms first: LoRA training, perplexity
with the ADC off before and after LoRA, and the same model with the ADC on.
It then runs phase 2, the eight-task downstream suite with the saved adapter
reloaded. Every perplexity result is written before the long lm-eval runs.

## Inputs

For each `same_backbone` model the runner uses the first of:

1. `TRANSFORMS_LLAMA32_1B`, `TRANSFORMS_SMOLLM2_17B`, `TRANSFORMS_TINYLLAMA_11B`
2. `$TRANSFER_CHECKPOINT_ROOT/<key>/adc_transfer/flat_quant_transforms.pt`
   (default `ADC/llama/transfer_results/checkpoints`)
3. a previous extraction under `matched_control_results/extracted_transforms`
4. the single `adc_transfer/flat_quant_transforms.pt` member of `<key>.tar.gz`
   in `ARCHIVE_DIR` or `transfer_results_archives/checkpoints[/checkpoints]`

The summary reads the ADC arm from `TRANSFER_RESULTS_JSON` (default
`ADC/llama/transfer_results/multi_arch_transfer.json`).

## Commands

From the repository root:

```bash
PREFLIGHT=1 bash ADC/llama/run_matched_lora_control.sh   # environment, tests, transforms
SMOKE=1 bash ADC/llama/run_matched_lora_control.sh       # tiny end-to-end run of every arm
bash ADC/llama/run_matched_lora_control.sh               # full run, both GPUs
```

`SMOKE=1` writes to `matched_control_results/smoke`, so it never marks
full runs as completed. Rerunning the full command skips runs that have a
successful record (`SKIP_COMPLETED=1`). An interrupted `flatquant` Stage A
resumes from its layer checkpoint, and completed transforms are reused.
`RUN_DOWNSTREAM=0` stops after phase 1; a later run without it performs only
the missing downstream evaluations. `DRY_RUN=1` prints the commands.

Rebuild the summary at any time:

```bash
python3 ADC/llama/summarize_matched_lora_control.py \
  --transfer-json ADC/llama/transfer_results/multi_arch_transfer.json \
  --control-dir ADC/llama/matched_control_results/results \
  --markdown-output ADC/llama/matched_control_results/summary.md
```

## Outputs

- `results/<key>_noadc_lora_<arm>.json`: `ppl_noadc_prelora_*`,
  `ppl_noadc_lora_*` (the matched control), `ppl_adc_on_prelora_*`, and
  `ppl_adc_on_lora_*` (the ADC-off-trained adapter deployed with the ADC)
- `results/<key>_noadc_lora_<arm>_downstream.json`: `downstream_noadc_lora`
  plus a perplexity re-evaluation that must match phase 1
- `checkpoints/<key>/<arm>/lora_adapter.pt`: adapter tensors and the settings
  they were trained with; a reload rejects any mismatch
- `logs/`, and `summary.md` with the gain table, the cross-deployment table,
  and consistency checks. The `same_backbone` pre-LoRA numbers must reproduce
  `<key>_int4_ptq_adc_off` (ADC off) and the transfer pre-LoRA ADC perplexity
  (ADC on).

## Notes

- `calibrate_adc_lora` stops after `nsamples / fq_cali_bsz` loader batches,
  so with `--lora_nsamples 1024 --fq_cali_bsz 16` both arms train on
  64 batches of 4 sequences (256 sequences of 512 tokens). The log line
  `Training 5 epochs on 64 batches` confirms it.
- Each arm is a single seed (42). The repeated CE configuration of the paper
  varies by about 0.07 WikiText-2 perplexity across seeds.
- The CE+KL teacher runs on the CPU. `THREADS_PER_LANE` (default: all cores
  divided by the active lanes) keeps the two lanes from oversubscribing it.
