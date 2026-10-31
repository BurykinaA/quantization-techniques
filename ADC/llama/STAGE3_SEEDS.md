# Stage 3 seed variance

Reviewers dEPN and AI ask how stable the Stage 3 numbers are across seeds.
The question matters most for TinyLlama, whose Table 2 downstream gain from
Stage 3 is +0.39 points (41.81 → 42.20). `run_stage3_seeds.sh` retrains the
Table 2 Stage 3 residual (rank 4, α = 8, all seven projections, CE + 0.5 KL
at T = 2, lr 1e-4, 5 epochs, 256 calibration sequences) with several seeds
on the Table 2 Stages 1–2 backbone and measures converter-on perplexity and
the paper's eight downstream tasks per seed.

What a seed changes:

- `--seed <s>`: the adapter initialization (`lora_A`, Kaiming uniform). The
  backbone, its quantizer scales and the evaluation do not depend on it.
- `--lora_data_seed <s>` (new runner flag, passed with `DATA_SEED=1`, the
  default): the residual's calibration pool is permuted once with this seed,
  so the 256 training sequences and their order change too, while every
  epoch still sees the same sequences. Without it the residual trains on the
  first 64 batches of the pool in corpus order, as in Table 2. `DATA_SEED=0`
  keeps that data for every seed.

The backbone is the transfer run's final Stage 1–2 transforms,
`transfer_results/checkpoints/<key>/adc_transfer/flat_quant_transforms.pt`
(override with `TRANSFORMS_<KEY>`), reloaded with `--fq_reload_path` as in
`run_adc_midpoint.sh`. For SmolLM2 and TinyLlama these are exactly the
Table 2 transforms; the run with `PRE_LORA_LM_EVAL_SEED` (default: the first
seed) also measures the backbone's downstream before Stage 3, which should
reproduce the Table 2 Stages 1–2 row and is the baseline for the gain. Llama
has no saved Table 2 transforms; set `TRANSFORMS_LLAMA32_1B` to the v6
transforms used for the midpoint runs (Stages 1–2 floor 26.99/47.65 against
26.66/45.62 in Table 2) and compare within that backbone.

Deviations from the Table 2 runs, all shared by every seed: the CE+KL teacher
runs on the GPU (`LORA_TEACHER_DEVICE=cuda`), the LoRA microbatch is 2 × 2
for every model (Llama's 4 × 1 no longer fits with the converter on), and
lm-evaluation-harness runs at batch 16 (`LM_EVAL_BATCH_SIZE`; batch 1 takes
about a day per model). Adapters are saved to
`checkpoints/<run>/lora_weights.pt`; `REUSE_LORA=1` reloads them instead of
training.

## Commands

From the repository root:

```bash
SMOKE=1 bash ADC/llama/run_stage3_seeds.sh                       # tiny end-to-end run
DRY_RUN=1 bash ADC/llama/run_stage3_seeds.sh                     # print the commands
bash ADC/llama/run_stage3_seeds.sh                               # TinyLlama, seeds 42 43 44, GPU 1
GPU0_MODELS="tinyllama_11b" GPU0_SEEDS="44" GPU1_MODELS="" \
  bash ADC/llama/run_stage3_seeds.sh                             # one more seed on GPU 0
TRANSFORMS_LLAMA32_1B=ADC/llama/checkpoints/overnight_int4_v6_20260406/lora_r4_down_20260406/flat_quant_transforms.pt \
  GPU0_MODELS="llama32_1b" GPU1_MODELS="" bash ADC/llama/run_stage3_seeds.sh
```

Lanes run in parallel: GPU 0 runs `GPU0_MODELS` × `GPU0_SEEDS`, GPU 1 runs
`GPU1_MODELS` × `GPU1_SEEDS` (both seed lists default to `SEEDS`). Each lane
holds its own lock, so a second invocation can use the other GPU with the
same output directory; keep the seed lists disjoint. Outputs go to
`ADC/llama/stage3_seeds_results/` (`_<dtype>` and `_lora_<dtype>` suffixes as
in the Stage 3 ablation), run names are `<key>_stage3_seed<s>`. Finished runs
are skipped (`SKIP_COMPLETED=1`); with `SKIP_COMPLETED=0` the record is
replaced and the old log kept with a timestamp.

Per seed on one A100: about 20 minutes of calibration and perplexity, 15–20
minutes of Stage 3, and about 2 hours of downstream at batch 16; the
`PRE_LORA_LM_EVAL_SEED` run adds a second downstream pass on the backbone.

## Summary

```bash
python3 ADC/llama/summarize_stage3_seeds.py \
  --results-dir ADC/llama/stage3_seeds_results/results
```

Per model: a row per seed with perplexity, the eight accuracies, their mean
and the gain over the Stages 1–2 backbone; a `mean ± std` row (sample
standard deviation); the Table 2 row for comparison; and the mean gain with
its 95% t confidence interval. The checks flag seeds without downstream
results, a backbone whose downstream was not measured (the gain then uses
the Table 2 row) and a pre-Stage 3 perplexity that differs between seeds.
