# Matched no-ADC control for the post-ADC residual

`run_matched_lora_control.sh` measures how much the Stage 3 residual gains
when there is no converter. Per model it runs, in one process:

1. FlatQuant calibrated from scratch without the ADC floor/clamp
   (`--fq_bypass_adc_floor`): Stage A of the transfer protocol (1024
   sequences, 30 epochs, batch 16, MLP diagonals), no Stage B
2. perplexity before LoRA with the ADC bypassed
3. the residual of the transfer protocol (rank 4, alpha 8, all seven
   projections, CE + 0.5 KL at T=2, lr 1e-4, 5 epochs, effective batch 4,
   the same WikiText-2 batches) trained with the ADC bypassed
   (`--lora_adc_off_control`)
4. perplexity after LoRA with the ADC bypassed

Tiled W4A4 quantization stays active throughout; only the floor/clamp is
removed. The gain from step 2 to step 4 is compared with the Stage 3 gain of
the ADC arm (`run_multi_arch_transfer.sh`).

Default lanes: GPU 0 runs `llama32_1b tinyllama_11b`, GPU 1 runs
`smollm2_17b`. Override them with `GPU0_MODELS` / `GPU1_MODELS`.

## Backbone

Step 1 must reproduce the model's "FlatQuant, no ADC" row of Table 2, and the
rows were built in two different ways:

- `BACKBONE=flatquant` (default) calibrates FlatQuant from scratch without the
  floor. This is how the Llama row was produced, by
  `run_int_only_baseline_1b.sh` on `multi-model-paper-comparison`.
- `BACKBONE=adcquant` reloads the final ADC-Quant transforms of the transfer
  run and bypasses the ADC, exactly as `run_int4_adc_off` produced the
  SmolLM2/TinyLlama rows (`<key>_int4_ptq_adc_off`). The transforms default to
  `transfer_results/checkpoints/<key>/adc_transfer/flat_quant_transforms.pt`;
  override them with `TRANSFORMS_SMOLLM2_17B` / `TRANSFORMS_TINYLLAMA_11B`.

```bash
BACKBONE=adcquant GPU0_MODELS="tinyllama_11b" GPU1_MODELS="smollm2_17b" \
  bash ADC/llama/run_matched_lora_control.sh
```

## dtype

Run each model in the dtype of the ADC arm it is compared with. The Llama
rows of Table 2 (the `run_overnight_int4_v*` series and the "FlatQuant, no
ADC" row) ran in float16, the default of the runner, because those scripts
never passed `--torch_dtype`. The SmolLM2 and TinyLlama rows come from
`run_multi_arch_transfer.sh`, which always passes bfloat16; its SmolLM2
Stage A also overflowed under float16 autocast.

`TORCH_DTYPE` defaults to `bfloat16`. Any other value writes to
`matched_control_results_<dtype>/`, so a run never reuses transforms or
completed records from another dtype:

```bash
TORCH_DTYPE=float16 GPU0_MODELS="" GPU1_MODELS="llama32_1b" \
  bash ADC/llama/run_matched_lora_control.sh
```

## Commands

From the repository root, in the environment used for the transfer runs:

```bash
python3 -m pytest ADC/llama/tests/test_matched_lora_control.py -q
SMOKE=1 bash ADC/llama/run_matched_lora_control.sh   # tiny end-to-end run
bash ADC/llama/run_matched_lora_control.sh           # full run, both GPUs
```

`SMOKE=1` writes to `matched_control_results/smoke`, so it never marks full
runs as completed. Rerunning the full command skips finished models
(`SKIP_COMPLETED=1`), resumes an interrupted Stage A from its layer
checkpoint, and reuses finished transforms (`FORCE_FQ_RETRAIN=1` starts
over). `DRY_RUN=1` prints the commands. The script exits non-zero when any
run failed, so `SMOKE=1 bash ... && bash ...` starts the full run only after
a clean smoke run.

Stage A progress: each finished decoder layer logs
`layer N restored best checkpoint`.

Rebuild the summary at any time, from one or several result directories:

```bash
python3 ADC/llama/summarize_matched_lora_control.py \
  --results-dir ADC/llama/matched_control_results/results \
                ADC/llama/matched_control_results_float16/results
```

## Outputs

Under `ADC/llama/matched_control_results/`:

- `results/<key>_<backbone>_noadc_lora.json` with `ppl_noadc_prelora_*`
  (backbone, ADC off) and `ppl_noadc_lora_*` (backbone + residual)
- `checkpoints/<key>/flat_quant_transforms.pt`, the recalibrated FlatQuant
  transforms (`flatquant` only)
- `logs/`, and `summary.md`

## Notes

- Calibration sequences are the first WikiText-2 train paragraphs longer than
  50 characters, in file order, each truncated or left-padded to 512 tokens.
  FlatQuant uses the first 1024 (`Captured 1024 calibration samples`).
  `calibrate_adc_lora` stops after `nsamples / fq_cali_bsz` loader batches,
  so with `--lora_nsamples 1024 --fq_cali_bsz 16` Stage 3 trains on the first
  64 batches of 4 (256 sequences), exactly as in the ADC arm; the log line
  `Training 5 epochs on 64 batches` confirms it.
- Each model is a single seed (42).
- The CE+KL teacher runs on the CPU. `THREADS_PER_LANE` (default: all cores
  divided by the active lanes) keeps the two lanes from oversubscribing it.
