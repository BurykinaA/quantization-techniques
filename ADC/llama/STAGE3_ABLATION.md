# Stage 3 without Stages 1–2

Reviewers TuRo and dEPN ask whether Stages 1–2 are still needed once the post-ADC
residual is available: the residual sees the full-precision projection input
and could in principle correct much of the converter error on its own.
`run_stage3_ablation.sh` trains the same Stage 3 residual on backbones with
progressively more pre-ADC calibration:

| Arm (`ARMS`) | Models | Backbone before Stage 3 |
| :-- | :-- | :-- |
| `no_calib` | `GPU0_MODELS`, `GPU1_MODELS` | no learned transforms (`--preprocess_method none`); only the quantizer scales are calibrated |
| `stage1` | Llama | Stage 1 only: clean-input calibration with the converter in the loss, MLP diagonals, no Stage 2 (μ = 0) |
| `stage12` | Llama | Stage 2 (μ = 0.5, attention diagonals) started from the same Stage 1 transforms |

Each model is compared with its Table 2 Stages 1–3 row. The settings follow
the runs behind those rows:

- Llama: `run_overnight_int4_v8.sh` (`r4_all_ce_kl`), float16.
- SmolLM2 and TinyLlama: `run_multi_arch_transfer.sh`, bfloat16.

Common settings: 1024 calibration sequences; Stage 1 30 epochs at 5e-3;
Stage 2 10 epochs; Stage 3 with rank 4, α = 8, all seven projections,
CE + 0.5 KL at T = 2, lr 1e-4, 5 epochs. The converter-on evaluation uses
WikiText-2 and C4, 1000 samples, context 2048, stride 1024.

The LoRA microbatch is 2 with two accumulation steps for every model, as in
the SmolLM2 and TinyLlama transfer runs. Llama's Table 2 row used microbatch
4, which no longer fits in 80 GB with the converter on. Each loader batch of
4 is split into two microbatches, so the samples per optimizer step are
unchanged.

For Llama, the calibration code has changed since v8, for example best-epoch
restoration. `stage12` reruns Stages 1–2 with the current code from the same
Stage 1 transforms as `stage1`. Use it, not 14.03, for the Stage 1 versus
Stages 1–2 comparison.

`no_calib` and Stage 3 read their data budget from `--fq_nsamples 1024
--fq_cali_bsz 16`, the same 64 batches of 4 sequences as in every Stage 3
run.

## Commands

From the repository root:

```bash
SMOKE=1 bash ADC/llama/run_stage3_ablation.sh    # tiny end-to-end run
bash ADC/llama/run_stage3_ablation.sh            # all arms, both GPUs
ARMS="no_calib" bash ADC/llama/run_stage3_ablation.sh
DRY_RUN=1 bash ADC/llama/run_stage3_ablation.sh  # print the commands only
```

Lanes:

- GPU 0: `llama32_1b_stage1` (Stage 1, evaluated, transforms saved) and
  `llama32_1b_stage1_stage3`, then `<key>_no_calib_stage3` for
  `GPU0_MODELS` (default `smollm2_17b`).
- GPU 1: `<key>_no_calib_stage3` for `GPU1_MODELS` (default
  `llama32_1b tinyllama_11b`),
  then `llama32_1b_stage12_stage3`. The latter waits for the Stage 1
  transforms from GPU 0 and is skipped if Stage 1 fails.

Outputs go to `ADC/llama/stage3_ablation_results/`. Finished runs are skipped
(`SKIP_COMPLETED=1`), and the script exits non-zero when any run failed.
`stage1` and `stage12` can run later on their own; `stage12` needs a finished
`stage1`.

Rebuild the summary at any time:

```bash
python3 ADC/llama/summarize_stage3_ablation.py \
  --results-dir ADC/llama/stage3_ablation_results/results
```

## Notes

- No pre-Stage-3 perplexity threshold is set: the uncalibrated backbones are
  expected to be collapsed, and Stage 3 is trained on them regardless.
- Each run is a single seed (42).
