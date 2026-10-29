# Stage 3 without Stages 1–2

Reviewers TuRo and dEPN ask whether Stages 1–2 are still needed once the post-ADC
residual is available: the residual sees the full-precision projection input
and could in principle correct much of the converter error on its own.
`run_stage3_ablation.sh` trains the same Stage 3 residual on Llama-3.2-1B
backbones with progressively more pre-ADC calibration:

| Arm (`ARMS`) | Backbone before Stage 3 |
| :-- | :-- |
| `no_calib` | no learned transforms (`--preprocess_method none`); only the quantizer scales are calibrated |
| `stage1` | Stage 1 only: clean-input calibration with the converter in the loss, MLP diagonals, no Stage 2 (μ = 0) |
| `stage12` | Stage 2 (μ = 0.5, attention diagonals) started from the same Stage 1 transforms |

Settings follow `run_overnight_int4_v8.sh` (`r4_all_ce_kl`), which produced
the Table 2 row 26.66 → 14.03 / 45.62 → 23.20: float16, 1024 calibration
sequences, Stage 1 30 epochs at 5e-3, Stage 2 10 epochs, and Stage 3 with
rank 4, α = 8, all seven projections, CE + 0.5 KL at T = 2, lr 1e-4, 5 epochs.
The converter-on evaluation uses WikiText-2 and C4, 1000 samples, context 2048,
stride 1024.

Two things differ from the Table 2 row:

- The LoRA microbatch is 2 with two accumulation steps instead of 4. Batch 4
  no longer fits in 80 GB with the converter on. Each loader batch of 4 is split
  into two microbatches, so the samples per optimizer step are unchanged.
- The calibration code has changed since v8, for example best-epoch
  restoration. `stage12` reruns Stages 1–2 with the current code from the same
  Stage 1 transforms as `stage1`. Use it, not 14.03, for the Stage 1 versus
  Stages 1–2 comparison; it also shows how far the current code is from the
  Table 2 row.

`no_calib` and Stage 3 read their data budget from `--fq_nsamples 1024
--fq_cali_bsz 16`, the same 64 batches of 4 sequences as in every Stage 3
run.

## Commands

From the repository root:

```bash
SMOKE=1 bash ADC/llama/run_stage3_ablation.sh    # tiny end-to-end run
bash ADC/llama/run_stage3_ablation.sh            # all three arms, both GPUs
ARMS="no_calib stage1" bash ADC/llama/run_stage3_ablation.sh
DRY_RUN=1 bash ADC/llama/run_stage3_ablation.sh  # print the commands only
```

Lanes:

- GPU 0: `llama32_1b_stage1` (Stage 1, evaluated, transforms saved), then
  `llama32_1b_stage1_stage3`.
- GPU 1: `llama32_1b_no_calib_stage3`, then `llama32_1b_stage12_stage3`. The
  latter waits for the Stage 1 transforms from GPU 0 and is skipped if Stage 1
  fails.

Outputs go to `ADC/llama/stage3_ablation_results/`. Finished runs are skipped
(`SKIP_COMPLETED=1`), and the script exits non-zero when any run failed.
`stage12` can also run later on its own (`ARMS=stage12`) once `stage1` has
finished.

Rebuild the summary at any time:

```bash
python3 ADC/llama/summarize_stage3_ablation.py \
  --results-dir ADC/llama/stage3_ablation_results/results
```

## Notes

- Before Stage 3, the `no_calib` perplexity is compared with the Table 2
  "ADC, no calib." row (2402.19 / 2614.93). The supplement lists the same
  2402.19 for a 128-sequence calibrated baseline (Table `int4v1`, no-ADC 15.41),
  so the two may differ.
- No pre-Stage-3 perplexity threshold is set: the uncalibrated backbone is
  expected to be collapsed, and Stage 3 is trained on it regardless.
- Each run is a single seed (42).
