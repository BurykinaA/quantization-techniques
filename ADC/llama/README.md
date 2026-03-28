# FlatQuant + ADC PTQ for Llama

## Setup

**Model:** Llama-3.2-1B
**Task:** Post-Training Quantization (PTQ) with Analog-Digital Compute (ADC)
**Calibration:** 128 samples × 2048 tokens, WikiText2
**Eval:** WikiText2 test, sliding window (ctx=2048, stride=1024)
**Branch for experiments:** `llama-flatquant-adc` (baseline), `llama-flatquant-adc-pact` (PACT experiments)

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

## Dead Zone Problem

The root cause of bad ADC quality: **81% of output channels are "dead"** (`y_int < delta`).

**Why it happens:**
Per-token amax activation scaling (`s_xi = amax(xi_tile) / 127`) concentrates all quantization range on the outlier feature within each tile. Typical features get code ≈ 0. The dot product `y_int = code_xi @ code_wi` is then dominated by a single sparse term, which for most output channels is too small to cross the delta threshold.

**Delta is a hardware constant** — it cannot be reduced by training.

---

## Experiments

### Baseline & ablations (branch `llama-flatquant-adc`)

| Experiment | Description | PPL bypass | PPL ADC | dead_rate (mean) | Notes |
|------------|-------------|------------|---------|-----------------|-------|
| **FP baseline** | BF16, no quantization | — | ~10.5 | — | Reference |
| **E2 baseline** | FlatQuant w8a8, per-token ADC | ~21.6 | **28.99** | 81% | Best ADC PPL so far |
| **E8 mid** | E2 + dead penalty λ=0.1 | — | 864 | 80.7% | LWC clip factors → delta≈0, catastrophic |
| **E8 weak+freeze** | E2 + dead penalty λ=0.01, freeze_clip=True | — | 1637 | 81.2% | Penalty didn't fix dead zone |
| **add_diag=True** | E2 + diagonal scaling transforms | — | ~9000 | — | Already tested, confirmed bad |
| **L1 loss** | FlatQuant with L1 reconstruction loss | — | 52.19 | 81.3% | Much worse than MSE |

**Key finding from branch `llama-flatquant-adc`:** P=I (MSE baseline E2) is near-optimal for this setup. Dead zone is structural — cannot be fixed by output-aware penalties or loss changes when using per-token scaling.

---

### PACT experiments (branch `llama-flatquant-adc-pact`)

**Idea (from advisor):** Replace per-token amax with a learned per-projection fixed threshold (PACT-style). With a smaller alpha, typical features get non-zero codes, making `y_int >> delta` for most channels.

| Experiment | Description | PPL bypass | PPL ADC | dead_rate (mean) | clip_rate | Notes |
|------------|-------------|------------|---------|-----------------|-----------|-------|
| **PACT-1** | Single alpha/proj, init=p99, alpha learned | 21.6 | 38945 | 81% | alpha=5 → still sparse codes |
| **PACT-2** | Single alpha/proj, init=p50, alpha learned | 71.1 | 34463 | 10.5% | Bypass PPL degraded: transforms adapted to PACT loss |
| **PACT-3** | Per-tile alpha, init=p50, alpha learned | 757 | 5768 | 10.4% | Worse: optimizer changed alpha during transform training |
| **PACT-4** | Per-tile alpha frozen at 100.0, post-training calib from xi | 397960 | 1545294 | 9.9% | alpha=100 → s_xi=0.787 (worse than per-token!) → transforms learn on dead codes |
| **PACT-5** | Per-token in train forward (E2-identical), PACT only at inference via TiledLinearADC | TBD | TBD | TBD | Current run |

#### Why PACT-1,2,3 failed

**PACT-1** (single alpha, large p99): alpha=5 → s_xi=0.039 → most features get code≈0 → dead zone unchanged.

**PACT-2** (single alpha, p50 init): dead zone fixed (10.5%), but **bypass PPL 21→71**. Root cause: optimizer drives alpha jointly with Kronecker transforms. PACT clips ~50% of activations → transforms adapt to the PACT loss landscape, becoming suboptimal for standard INT8 bypass.

**PACT-3** (per-tile alpha, p50 init): single alpha per projection caused tile-level catastrophe — the tile containing outlier features had ALL codes saturated at ±127 (alpha too small), giving y_int = random sign sum = wrong output. Per-tile alpha fixed this. But bypass PPL = 757 (even worse) because optimizer still trained alpha jointly with transforms across more parameters.

**PACT-4** (failed): alpha=100 is NOT equivalent to per-token.
- Per-token: `s_xi = amax(xi_tile) / 127` adapts per-token (for typical tile max≈5: s_xi=0.039)
- PACT alpha=100: `s_xi = 100/127 = 0.787` — fixed and huge
- Most codes ≈ 0 during training → transforms learn to compensate → extreme weight distributions → bypass PPL catastrophic

**PACT-5** (current): PACT only at inference, E2-identical training.
- `_train_forward_adc` reverted to per-token amax — PACT code removed entirely from training
- Post-training d5 calibration runs no-grad forward to capture per-tile xi values
- Sets `raw_alpha_adc` to p50 per tile from actual transformed activations
- `propagate_alpha_adc_to_tiled` applies PACT to TiledLinearADC at inference

Expected: bypass PPL ≈ 21.6 (transforms = E2), dead_rate ≈ 10% (PACT p50 at inference), ADC PPL < 28.99.

---

## Run commands

```bash
# Baseline (E2-equivalent on pact branch)
bash ADC/llama/run_flat_quant_adc_ptq_e7e8e9.sh baseline

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
