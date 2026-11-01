# Table 2 with seeds and midpoint reading

Reviewers asked for the variance of the main results, and two of them noted
that the floor reading of the converter leaves a one-sided error of about
Δ/2 per conversion that Stage 3 partly learns. `run_table2_seeds.sh`
retrains the two ADC-Quant rows of Table 2 from scratch for several seeds,
with every converter in the run reading non-saturated codes at the bin
centre, (q + 0.5)·Δ: the FlatQuant Stage A/B calibration loss, Stage 3 and
the evaluation (`--adc_midpoint`; `ADC_MIDPOINT=0` keeps the paper's floor
reading). The ADC scale calibration uses full-precision statistics and does
not depend on the reading.

One run (model, seed) is the Table 2 protocol of `run_multi_arch_transfer.sh`
(`run_adc_transfer`):

- FlatQuant Stage A (1024 sequences, 30 epochs, batch 16, MLP diagonals) and
  the converter-aware Stage B (10 epochs, α = 0.5, attention diagonals);
- ADC scale calibration (100 batches of 4 sequences);
- converter-on WikiText-2 / C4 perplexity and the eight downstream tasks of
  the Stages 1–2 backbone;
- the Stage 3 residual (rank 4, α = 8, all seven projections, CE + 0.5 KL at
  T = 2, lr 1e-4, 5 epochs, 256 sequences), then perplexity and the eight
  tasks again.

What a seed changes: `--seed` sets the random orthogonal initialization of
the Kronecker transforms and the adapter initialization; `--lora_data_seed`
(`DATA_SEED=1`, default) permutes the residual's calibration pool, as in the
Stage 3 seed runs (`STAGE3_SEEDS.md`). The FlatQuant calibration data is the
first 1024 sequences in corpus order for every seed. The BF16 and "FlatQuant,
no ADC" rows are not rerun; the LaTeX output copies them from Table 2. The
"ADC, no calib." row is not part of the run.

Deviations from the Table 2 runs, shared by every seed: midpoint reading;
bfloat16 for every model (Table 2's Llama rows are float16); LoRA microbatch
2 × 2 for every model (Llama's 4 × 1 no longer fits with the converter on);
the CE+KL teacher on the GPU; lm-evaluation-harness at batch 16.

## Commands

From the repository root, in the server environment:

```bash
python3 -m pytest ADC/llama/tests/test_table2_seeds.py ADC/llama/tests/test_adc_midpoint.py -q
bash -n ADC/llama/run_table2_seeds.sh
DRY_RUN=1 bash ADC/llama/run_table2_seeds.sh      # print every command
SMOKE=1 bash ADC/llama/run_table2_seeds.sh        # tiny end-to-end run on both GPUs
```

The full run, detached, with the console in a file:

```bash
tmux new -d -s table2 'cd ~/project && source .venv/bin/activate && \
  bash ADC/llama/run_table2_seeds.sh > ADC/llama/table2_seeds.out 2>&1'
```

Defaults: seeds `42 43 44`; GPU 0 runs SmolLM2, GPU 1 runs Llama and
TinyLlama; each lane goes seed-major (seed 42 of every model, then 43, then
44), so a complete one-seed table exists early and the std fills in later.
`GPU0_JOBS` / `GPU1_JOBS` give a lane an explicit `key:seed` order, for
example to move one TinyLlama seed to the SmolLM2 lane:

```bash
GPU0_JOBS="smollm2_17b:42 smollm2_17b:43 smollm2_17b:44 tinyllama_11b:44" \
GPU1_JOBS="llama32_1b:42 tinyllama_11b:42 llama32_1b:43 tinyllama_11b:43 llama32_1b:44" \
  bash ADC/llama/run_table2_seeds.sh
```

Other knobs: `SEEDS`, `GPU0_MODELS` / `GPU1_MODELS`, `GPU0_SEEDS` /
`GPU1_SEEDS`, `ADC_MIDPOINT` (1), `DATA_SEED` (1), `TORCH_DTYPE` (bfloat16),
`LORA_DTYPE` (float32), `LM_EVAL` (1), `LM_EVAL_BATCH_SIZE` (16),
`LORA_TEACHER_DEVICE` (cuda), `LORA_MICROBATCH_SIZE` ×
`LORA_GRADIENT_ACCUMULATION_STEPS` (2 × 2), `PRE_LORA_PPL_THRESHOLD` (500,
empty disables the guard), `MAX_ATTEMPTS` (2), `SAVE_MODEL` (0; 1 also saves
the merged model with `save_pretrained`), `NO_ADC_EVAL` (0; 1 adds a
perplexity-only evaluation of each seed's transforms with the converter
bypassed, the "FlatQuant, no ADC" number of the same transforms), `PYTHON_BIN`,
`TABLE2_ROOT`.

### Unattended operation

- A failed run is retried `MAX_ATTEMPTS` times in a row and then skipped;
  the lane moves on to the next job and the script exits non-zero at the end.
- Rerunning the same command after an interruption resumes every unfinished
  run: Stage A and Stage B from their per-layer progress checkpoints, Stage B
  from saved Stage A transforms, Stage 3 from saved Stages 1–2 transforms
  (the pre-Stage 3 perplexity and downstream are read from the old log), and
  the evaluation from saved adapters. Finished runs are skipped
  (`SKIP_COMPLETED=1`). To retrain a run from scratch, delete its
  `checkpoints/<run>/` directory.
- Each lane holds a lock per GPU, so a second invocation can use a free GPU
  with the same output directory.

Progress while it runs:

```bash
tail -f ADC/llama/table2_seeds.out                       # START / DONE / FAIL / RESULT lines
cat ADC/llama/table2_seeds_results/results.log           # one line of final numbers per finished run
cat ADC/llama/table2_seeds_results/summary.md            # rebuilt after every run
```

## Outputs

Under `ADC/llama/table2_seeds_results/` (`_floor`, `_<dtype>` and
`_lora_<dtype>` suffixes when the defaults change; `smoke/` for `SMOKE=1`):

- `checkpoints/<key>_midpoint_seed<s>/`: `flat_quant_transforms_stage_a.pt`,
  `flat_quant_transforms.pt` (final Stages 1–2), `lora_weights.pt` (the
  residual), progress checkpoints while a stage runs, and the merged model
  with `SAVE_MODEL=1`.
- `results/<run>.json`: the runner's record with `ppl_adc_prelora_*`,
  `downstream_adc_ptq` (Stages 1–2), `ppl_adc_*`, `downstream_adc_lora`
  (Stages 1–3) and the full configuration.
- `logs/<run>.log`: the runner's console output (an older attempt is kept
  with a timestamp).
- `results.log`: one line per finished run with the perplexities and the
  downstream means of both rows.
- `summary.md`: per model, both Table 2 rows for every seed, `mean ± std`,
  the Table 2 (floor) row, the Stage 3 gain with a 95% t confidence interval,
  and checks (missing seeds or downstream results, a Stage 3 that raised
  perplexity).
- `table_rows.tex`: the rows of Table 2 with `\pmcell{mean}{std}` cells
  (`\newcommand{\pmcell}[2]{#1\,{\scriptsize$\pm$#2}}`), the digital
  reference rows copied from Table 2, bold on the better converter-enabled
  row.
- `status.txt`: written when every lane has finished.

Rebuild the summary at any time:

```bash
python3 ADC/llama/summarize_table2_seeds.py \
  --results-dir ADC/llama/table2_seeds_results/results \
  --markdown-output ADC/llama/table2_seeds_results/summary.md \
  --latex-output ADC/llama/table2_seeds_results/table_rows.tex
```

## Time

Per model and seed on one A100, from the transfer runs and the Stage 3 seed
runs: Stage A about 8 hours for SmolLM2 and less for the 1B models, Stage B
a few hours, ADC calibration and perplexity about 20 minutes, each downstream
pass about 2 hours at batch 16 (two passes per run), Stage 3 about 20
minutes: roughly 10–16 hours per run. With the default lanes the first seed
of every model is done after about a day and all three seeds after about
2.5 days; GPU 0 (SmolLM2 only) finishes before GPU 1.

## Code change

`--adc_midpoint` now also reaches the FlatQuant calibration loss: the
converter simulated inside `FlatQuantLinear._train_forward_adc`
(`adc_config["midpoint"]`) reads non-saturated codes at (q + 0.5)·Δ, as
`TiledLinearADC` does after `set_adc_midpoint(True)`. Transforms trained
with the flag are therefore calibrated against the converter they run with.
Reloaded transforms are unaffected, so the midpoint runs of `ADC_MIDPOINT.md`
(floor-calibrated transforms, read at the midpoint) keep their meaning.
