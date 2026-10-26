# Midpoint ADC reconstruction

The simulated converter floors: a code q covers inputs in [q, q + 1) steps,
and dequantization returns its lower edge q·Δ. That leaves an error of about
Δ/2 per conversion with the same sign in every tile, so it does not average
out across the tiles of a projection (supplement, Appendix A). Midpoint
reconstruction (`--adc_midpoint`) returns (q + 0.5)·Δ for every non-saturated
code. Saturated edge codes keep their edge value. It is a constant digital
offset, free in hardware, and is applied only at dequantization; calibration
and the integer MVM are unchanged.

`run_adc_midpoint.sh` reloads the final Stage 1-2 transforms of the transfer
run (`transfer_results/checkpoints/<key>/adc_transfer/flat_quant_transforms.pt`,
override with `TRANSFORMS_<KEY>`) and keeps the converter on:

- `MODE=eval` (default) evaluates WikiText-2/C4 perplexity with floor, which
  must reproduce the model's Stage 1-2 row, and with midpoint. No training.
- `MODE=stage3` retrains the transfer-protocol residual (rank 4, all seven
  projections, CE + 0.5 KL, 5 epochs) with midpoint on and evaluates it
  before and after. Worth running only if `MODE=eval` shows a large effect.

The summary reports the share of the ADC arm's Stage 3 gain that midpoint
alone recovers: (floor − midpoint) / (floor − Stage 1-3 with floor), with
Stage 1-3 taken from `<key>_adc_transfer` in `TRANSFER_RESULTS_JSON`.

## Commands

From the repository root:

```bash
python3 -m pytest ADC/llama/tests/test_adc_midpoint.py -q
SMOKE=1 bash ADC/llama/run_adc_midpoint.sh        # tiny end-to-end run
bash ADC/llama/run_adc_midpoint.sh                # MODE=eval, both GPUs
MODE=stage3 bash ADC/llama/run_adc_midpoint.sh    # optional retraining
```

Lanes default to GPU 0: `llama32_1b tinyllama_11b`, GPU 1: `smollm2_17b`
(`GPU0_MODELS` / `GPU1_MODELS`). Outputs go to
`ADC/llama/adc_midpoint_results/`; other values of `TORCH_DTYPE` write to
`adc_midpoint_results_<dtype>/`. Finished runs are skipped
(`SKIP_COMPLETED=1`), and the script exits non-zero when any run failed.

Rebuild the summary at any time:

```bash
python3 ADC/llama/summarize_adc_midpoint.py \
  --results-dir ADC/llama/adc_midpoint_results/results \
  --transfer-json ADC/llama/transfer_results/multi_arch_transfer.json
```

## Notes

- The transfer transforms are bfloat16. For Llama, the Table 2 Stage 1-2 row
  (26.66) comes from the older float16 series, so the floor run reproduces
  the transfer record rather than 26.66. The floor/midpoint comparison itself
  is unaffected.
- Each run is a single seed (42).
