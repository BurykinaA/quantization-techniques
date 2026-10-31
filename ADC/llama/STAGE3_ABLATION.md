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

Outputs go to `ADC/llama/stage3_ablation_results/`. `TORCH_DTYPE` runs every
model in one dtype instead of its Table 2 dtype and writes to
`stage3_ablation_results_<dtype>/`, for example Llama in bfloat16 like the
other two models:

```bash
TORCH_DTYPE=bfloat16 ARMS="no_calib" GPU0_MODELS="llama32_1b" GPU1_MODELS="" \
  bash ADC/llama/run_stage3_ablation.sh
```

`LORA_DTYPE` (default `float32`) sets the dtype of the Stage 3 residual: its
weights, arithmetic and AdamW state. Other values append `_lora_<dtype>` to
the directory. The converter itself is always simulated in float32.

`LM_EVAL=1` adds the paper's eight downstream tasks (lm-evaluation-harness,
MMLU 5-shot, the rest 0-shot, batch size `LM_EVAL_BATCH_SIZE`, default 1 as
in the transfer runs) to every run. With Stage 3 they are measured after the
residual only: the uncalibrated backbones are collapsed before it, and the
paper reports no downstream numbers for collapsed rows. The summary gets a
downstream table next to the Table 2 rows.

`LORA_TEACHER_DEVICE` (default `cuda`) places the CE+KL teacher. The Table 2
runs kept it on the CPU, where a 1B teacher costs tens of seconds per
microbatch and dominates the Stage 3 time; on the GPU it needs about 5 GB
(fp16 weights plus one microbatch of logits) next to the student and runs
before the student's forward, so its activations are gone at the memory
peak. Use `cpu` if Stage 3 runs out of memory.

Every Stage 3 run saves its adapters to
`checkpoints/<run_name>/lora_weights.pt` (about 11 MB for Llama). The runner
reloads them with `--lora_reload_path <file>` in place of training, so an
evaluation can be repeated or extended without the hours of Stage 3; the
model, LoRA flags and `--lora_dtype` must match the saved run. In the script,
`REUSE_LORA=1` does this for every run whose file exists (runs without one
train as usual), for example to add downstream or change
`LM_EVAL_BATCH_SIZE` afterwards. lm-evaluation-harness at batch 1 is bound by
the kernel launches of the tiled simulation (about 2 requests/s, a day per
model for the eight tasks); batch 16 is roughly ten times faster and changes
the scores only by padding arithmetic.

Finished runs are skipped (`SKIP_COMPLETED=1`), and the script exits non-zero
when any run failed. `SKIP_COMPLETED=0` reruns them: the results record with
the same run name is replaced and the previous log is kept with a timestamp
suffix. `stage1` and `stage12` can run later on their own; `stage12` needs a
finished `stage1`.

Downstream for the finished `no_calib` arm (the records only had perplexity
and the adapters were not saved), Llama in bfloat16 with a bfloat16 residual
and the other two in their Table 2 setting, in two tmux windows:

```bash
LM_EVAL=1 SKIP_COMPLETED=0 ARMS="no_calib" TORCH_DTYPE=bfloat16 LORA_DTYPE=bfloat16 \
  GPU0_MODELS="" GPU1_MODELS="llama32_1b" bash ADC/llama/run_stage3_ablation.sh
LM_EVAL=1 SKIP_COMPLETED=0 ARMS="no_calib" \
  GPU0_MODELS="smollm2_17b tinyllama_11b" GPU1_MODELS="" bash ADC/llama/run_stage3_ablation.sh
```

Rebuild the summary at any time:

```bash
python3 ADC/llama/summarize_stage3_ablation.py \
  --results-dir ADC/llama/stage3_ablation_results/results
```

## Notes

- No pre-Stage-3 perplexity threshold is set: the uncalibrated backbones are
  expected to be collapsed, and Stage 3 is trained on them regardless.
- Each run is a single seed (42).
