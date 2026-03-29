# FlatQuant + ADC PTQ for Llama

## Setup

**Model:** Llama-3.2-1B
**Task:** Post-Training Quantization (PTQ) with Analog-Digital Compute (ADC)
**Calibration:** 128 samples × 2048 tokens, WikiText2
**Eval:** WikiText2 test, sliding window (ctx=2048, stride=1024)
**Branch for experiments:** `llama-flatquant-adc` (early baseline), `llama-flatquant-adc-pact` (current)

### Quantization config

| Parameter | Value |
|-----------|-------|
| Weight bits (`bw`) | 8 |
| Activation bits (`bx`) | 8 |
| ADC bits (`ba`) | 8 |
| Tile size (`mvm_limit`) | 256 |
| ADC parallelism (`k`) | 16 |
| ADC delta (hardware constant) | ≈ 2016 |
| FlatQuant epochs | 30 |
| FlatQuant LR | 0.005 |
| Calibration samples | 128 |

**ADC delta formula:** `delta = 2 * tile_in * 127 * 127 / (2^ba * k) = 2 * 256 * 127 * 127 / (256 * 16) ≈ 2016`

### Key metrics

- **PPL (bypass)** — WikiText2 PPL with ADC disabled (only tiling + INT8 quant). Shows FlatQuant transform quality.
- **PPL (ADC)** — WikiText2 PPL with full ADC pipeline.
- **dead_rate** — fraction of output channels where `|y_int| < delta` (ADC outputs 0). Lower is better.
- **clip_rate** — fraction of ADC outputs saturated at `|z| > 127`.
- **bin_usage** — fraction of ADC output bins used. Higher is better.
- **reconstruction_rel** — relative MSE of ADC output vs FP on calibration data.

---

## ADC Quality Problem

**Current status (pact branch, new baseline):** bypass PPL = 10.0, ADC PPL = 28.86, dead_rate mean = **10.3%**.

The dead_rate of 81% seen in early experiments (branch `llama-flatquant-adc`) was a FlatQuant transform quality issue, not a fundamental hardware constraint. The pact branch produces better transforms (bypass 21.6 → 10.0) with much lower dead_rate (81% → 10.3%).

**Remaining gap:** bypass PPL 10.0 → ADC PPL 28.86. With dead_rate at 10.3% and reconstruction_rel at 0.22%, the ~3x PPL degradation comes primarily from **ADC quantization resolution** — delta=2016 gives only ~15–30 discrete output levels in the typical z range, not dead zone.

**Delta is a hardware constant** — it cannot be reduced by training.

---

## Experiments

### Branch `llama-flatquant-adc` (early)

| Experiment | Description | PPL bypass | PPL ADC | dead_rate (mean) | Notes |
|------------|-------------|------------|---------|-----------------|-------|
| **FP baseline** | BF16, no quantization | — | ~10.5 | — | Reference |
| **E2 baseline** | FlatQuant w8a8, per-token ADC | ~21.6 | 28.99 | 81% | Transforms suboptimal |
| **E8 mid** | E2 + dead penalty λ=0.1 | — | 864 | 80.7% | LWC clip factors → delta≈0, catastrophic |
| **E8 weak+freeze** | E2 + dead penalty λ=0.01, freeze_clip=True | — | 1637 | 81.2% | Penalty didn't fix dead zone |
| **add_diag=True** | E2 + diagonal scaling transforms | — | ~9000 | — | Confirmed bad |
| **L1 loss** | FlatQuant with L1 reconstruction loss | — | 52.19 | 81.3% | Much worse than MSE |

**Key finding:** With poor transforms (bypass=21.6), dead zone is structural and cannot be fixed by penalties or loss changes.

---

### Branch `llama-flatquant-adc-pact` (current)

#### PACT experiments

**Idea (from advisor):** Replace per-token amax with a learned per-projection fixed threshold (PACT-style). With a smaller alpha, typical features get non-zero codes, making `y_int >> delta` for most channels.

| Experiment | Description | PPL bypass | PPL ADC | dead_rate (mean) | Notes |
|------------|-------------|------------|---------|-----------------|-------|
| **PACT-1** | Single alpha/proj, init=p99, alpha learned | 21.6 | 38945 | 81% | alpha=5 → still sparse codes |
| **PACT-2** | Single alpha/proj, init=p50, alpha learned | 71.1 | 34463 | 10.5% | Bypass PPL degraded: transforms adapted to PACT loss |
| **PACT-3** | Per-tile alpha, init=p50, alpha learned | 757 | 5768 | 10.4% | Optimizer changed alpha during transform training |
| **PACT-4** | Per-tile alpha frozen at 100.0, post-training calib | 397960 | 1545294 | 9.9% | alpha=100 → s_xi=0.787 >> per-token → codes≈0 |
| **PACT-5 (buggy)** | Per-token training, PACT inference (bypass bug) | 26926 | 77183 | — | Bug: bypass used PACT s_x for dequant |
| **PACT-5 (fixed bypass, PACT inference)** | Bypass fix applied, PACT alpha at inference | 10.0 | 75448 | 10.3% (diag) | PACT p50 clipping error worse than dead zone |

**PACT conclusion:** PACT at p50 doesn't work. Clipping 50% of activations introduces reconstruction error worse than the dead zone benefit. PACT-2/5 both give ADC PPL ~34k–75k. Approach abandoned.

#### New baseline (per-token inference, improved transforms)

| Experiment | Description | PPL bypass | PPL ADC | dead_rate (mean) | reconstruction_rel | Notes |
|------------|-------------|------------|---------|-----------------|-------------------|-------|
| **New baseline** | FlatQuant w8a8, per-token ADC (no PACT) | **10.01** | **28.86** | **10.3%** | 0.22% | Current best |

**Key finding:** Improved transforms on pact branch (bypass 21.6 → 10.0) also reduce dead_rate (81% → 10.3%). ADC PPL essentially unchanged (28.99 → 28.86) despite much lower dead_rate — confirming the ~3x gap (10 → 29) is **ADC resolution**, not dead zone.

---

## Run commands

```bash
# New baseline (per-token inference, improved transforms)
bash ADC/llama/run_flat_quant_adc_ptq_e7e8e9.sh baseline

# With PACT at inference (experimental, generally worse)
bash ADC/llama/run_flat_quant_adc_ptq_e7e8e9.sh baseline --pact_inference

# E7: clip penalty
bash ADC/llama/run_flat_quant_adc_ptq_e7e8e9.sh e7

# E8: dead-zone penalty
bash ADC/llama/run_flat_quant_adc_ptq_e7e8e9.sh e8

# E9: combined penalties
bash ADC/llama/run_flat_quant_adc_ptq_e7e8e9.sh e9
```

**WandB project:** `llama-flat-quant-adc-ptq-blocks`

---

## Files

| File | Description |
|------|-------------|
| `core/flat_quant.py` | FlatQuant calibration, FlatQuantLinear, PACT implementation |
| `core/adc_layers.py` | TiledLinearADC, QATLinearADC (inference ADC layers) |
| `runs/llama_smooth_quant_adc_ptq.py` | Main PTQ script |
| `run_flat_quant_adc_ptq_e7e8e9.sh` | Experiment launcher (baseline/e7/e8/e9/l1/huber) |
