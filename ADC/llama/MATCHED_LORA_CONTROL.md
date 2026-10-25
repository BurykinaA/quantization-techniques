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
| `flatquant` | FlatQuant recalibrated from scratch with `--fq_bypass_adc_floor` | off |
| `same_backbone` | final Stage 1-2 transforms of the transfer run, reloaded unchanged | off |

`flatquant` rebuilds the "FlatQuant, no ADC" reference the same way for every
model: Stage A of the transfer protocol (1024 sequences, 30 epochs, batch 16,
MLP diagonals) with the floor/clamp removed from the calibration loss and no
Stage B. The submitted table built this row differently per model (Llama:
`run_int_only_baseline_1b.sh` on `multi-model-paper-comparison`;
SmolLM2/TinyLlama: the ADC-Quant transforms with the ADC off).

`same_backbone` holds the transforms fixed, so the converter is the only
difference from the ADC arm. It needs the transfer checkpoints (see Inputs).

Default lanes, FlatQuant arms first because Stage A dominates the run time:

- GPU 0: `llama32_1b:flatquant tinyllama_11b:flatquant llama32_1b:same_backbone`
- GPU 1: `smollm2_17b:flatquant smollm2_17b:same_backbone tinyllama_11b:same_backbone`

Override them with `GPU0_ARMS` / `GPU1_ARMS`.

Each lane runs phase 1 for all of its arms first: LoRA training, perplexity
with the ADC off before and after LoRA, and the same model with the ADC on.
It then runs phase 2, the eight-task downstream suite with the saved adapter
reloaded, for `flatquant` also before LoRA. Every perplexity result is
written before the long lm-eval runs.

## Inputs

`flatquant` needs nothing beyond the Hugging Face models and datasets. For
each `same_backbone` model the runner uses the first of:

1. `TRANSFORMS_LLAMA32_1B`, `TRANSFORMS_SMOLLM2_17B`, `TRANSFORMS_TINYLLAMA_11B`
2. `$TRANSFER_CHECKPOINT_ROOT/<key>/adc_transfer/flat_quant_transforms.pt`
   (default `ADC/llama/transfer_results/checkpoints`)
3. a previous extraction under `matched_control_results/extracted_transforms`
4. the single `adc_transfer/flat_quant_transforms.pt` member of `<key>.tar.gz`
   in `ARCHIVE_DIR` or `transfer_results_archives/checkpoints[/checkpoints]`

The summary reads the ADC arm from `TRANSFER_RESULTS_JSON` (default
`ADC/llama/transfer_results/multi_arch_transfer.json`).

## Commands

From the repository root, in the environment used for the transfer runs:

```bash
PREFLIGHT=1 bash ADC/llama/run_matched_lora_control.sh   # environment, tests, transforms, GPUs
SMOKE=1 bash ADC/llama/run_matched_lora_control.sh       # tiny end-to-end run of every arm
bash ADC/llama/run_matched_lora_control.sh               # full run, both GPUs
```

`SMOKE=1` writes to `matched_control_results/smoke`, so it never marks
full runs as completed. Rerunning the full command skips runs that have a
successful record (`SKIP_COMPLETED=1`). An interrupted `flatquant` Stage A
resumes from its layer checkpoint, and completed transforms are reused.
`RUN_DOWNSTREAM=0` stops after phase 1; a later run without it performs only
the missing downstream evaluations. `DRY_RUN=1` prints the commands.

Stage A progress: each finished decoder layer logs
`layer N restored best checkpoint`.

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
- `results/<key>_noadc_lora_<arm>_downstream.json`: `downstream_noadc_lora`,
  for `flatquant` also `downstream_noadc_prelora`, plus perplexity
  re-evaluations that must match phase 1
- `checkpoints/<key>/<arm>/lora_adapter.pt`: adapter tensors and the settings
  they were trained with; a reload rejects any mismatch.
  `checkpoints/<key>/flatquant/flat_quant_transforms.pt`: the recalibrated
  FlatQuant transforms
- `logs/`, and `summary.md` with the gain table, the cross-deployment table,
  per-task rows in the Table 2 layout, and consistency checks. The
  `same_backbone` pre-LoRA numbers must reproduce `<key>_int4_ptq_adc_off`
  (ADC off) and the transfer pre-LoRA ADC perplexity (ADC on).

## Notes

- Calibration sequences are the first WikiText-2 train paragraphs longer than
  50 characters, in file order, each truncated or left-padded to 512 tokens.
  FlatQuant captures the first 1024 (`Captured 1024 calibration samples`).
  `calibrate_adc_lora` stops after `nsamples / fq_cali_bsz` loader batches,
  so with `--lora_nsamples 1024 --fq_cali_bsz 16` every Stage 3 run, in both
  arms, trains on the first 64 batches of 4 (256 sequences); the log line
  `Training 5 epochs on 64 batches` confirms it.
- Each arm is a single seed (42). The repeated CE configuration of the paper
  varies by about 0.07 WikiText-2 perplexity across seeds.
- The CE+KL teacher runs on the CPU. `THREADS_PER_LANE` (default: all cores
  divided by the active lanes) keeps the two lanes from oversubscribing it.
