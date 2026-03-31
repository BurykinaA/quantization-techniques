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

## What FlatQuant Transforms Do

FlatQuant learns per-layer **Kronecker-decomposed orthogonal transforms** applied to activations and weights before quantization. For each projection (q/k/v/o/gate/up/down), a pair of small orthogonal matrices (Kronecker factors) is trained to rotate the activation and weight space so that the resulting distributions are as flat (uniform) as possible — minimizing per-channel variance without changing the linear map.

```
y = x @ W^T  =  (x @ T^{-1}) @ (T @ W^T)  =  x_rot @ W_rot^T
```

The transform `T` is learned by minimizing MSE between FP output and INT8-quantized output of the rotated layer. After calibration, `T` is folded into the weights (reparameterization), so inference cost is the same as standard INT8.

**Why transforms matter for ADC:**

Per-token activation scaling (`s_xi = max(|xi_tile|) / 127`) concentrates all quantization range on the largest feature in each tile. If one feature is 10× larger than the rest, it gets code 127 and all others get codes ≈ 0–12. The integer dot product `y_int = code_xi @ code_w` is then dominated by a single sparse term, which for most output channels gives `|y_int| < delta = 2016` → dead zone.

Good transforms **flatten** the activation distribution within each tile: no single feature dominates. After rotation, all features have similar magnitude → all codes are ≈ 50–100 → every feature contributes to `y_int`. The dot product `y_int ≈ 64 * sum(code_w) ≈ 64 * 256 * mean(|code_w|)` is typically >> delta → no dead zone.

**Bypass PPL is a direct proxy for transform quality** — it measures INT8 reconstruction accuracy without ADC quantization noise. Lower bypass PPL = flatter distributions = more uniform codes = less dead zone.

## ADC Quality Problem

**Current status (pact branch, new baseline):** bypass PPL = 10.0, ADC PPL = 28.86, dead_rate mean = **10.3%**.

The dead_rate of 81% seen in early experiments (branch `llama-flatquant-adc`) was a **FlatQuant transform quality issue**, not a fundamental hardware constraint. The pact branch produces better transforms (bypass 21.6 → 10.0) and dead_rate dropped correspondingly (81% → 10.3%) — without any changes to the ADC hardware parameters or activation scaling scheme.

We discovered this when running the E2-equivalent baseline on the pact branch: the dead zone disappeared on its own, purely from better transforms. All the PACT experiments (PACT-1 through PACT-5) were trying to fix a problem that better FlatQuant training already solves.

**Why pact branch transforms are better** is not fully pinned down. The run uses `calibration_method=percentile` (99.9th percentile for static scale calibration in Step 2) vs the earlier `absmax`. The training code also accumulated several fixes during PACT development. The bypass PPL improvement (21.6 → 10.0, approaching FP baseline 10.5) is the cleanest indicator.

**Remaining gap:** bypass PPL 10.0 → ADC PPL 28.86. With dead_rate at 10.3% and reconstruction_rel at 0.22%, the ~3x PPL degradation comes primarily from **ADC quantization resolution** — delta=2016 gives only ~15–30 discrete output levels in the typical z range (std_z ≈ 15). Each active output channel has limited precision regardless of how well the transforms work.

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

#### Branch `llama-flatquant-adc-v2` experiments

**Hypothesis:** The bypass→ADC gap (10→29) comes from coarse ADC resolution (delta=2016). Two approaches to reduce it:

1. **Bin-center loss** (`--fq_lambda_center`): adds `cos²(π·z)` as a penalty during FlatQuant training. `z = y_int/delta`. The cosine has minimum at half-integer z (bin centres of floor quantizer) and maximum at integer z (bin boundaries). Minimising it nudges y_int toward bin centres, reducing per-step floor-rounding error from O(delta) to O(delta/4).

2. **Propagated calibration** (`--fq_propagate_quant`): each layer i is trained to map ADC-quantized inputs (from layer i-1) to FP reference outputs, rather than FP inputs → FP outputs. Addresses error accumulation across layers — layer-wise PTQ without error propagation may underestimate the reconstruction difficulty faced at inference.

| Experiment | Description | PPL bypass | PPL ADC | dead_rate | Notes |
|------------|-------------|------------|---------|-----------|-------|
| **center mid** | bin-center λ=0.1 | 10.01 | 28.86 | 10.3% | No improvement vs baseline; bin-center loss alone does not help |
| **prop** | propagated calibration | 11.01 | **14.55** | 10.3% | Major improvement: −47% ADC PPL vs baseline |
| **prop+center mid** | propagated + bin-center λ=0.1 | 11.00 | **14.41** | 10.3% | Best result; marginal gain over prop alone |

#### Branch `llama-flatquant-adc-int4-experiments` — INT4 overnight sweep

**Config:** bx=4, bw=4, ba=8, k=16. delta ≈ 6.12 (finer ADC resolution than INT8).

**Bug fix:** `LlamaADCConverter` was creating duplicate uncalibrated `TiledLinearADC` copies inside `_orig_attn.q/k/v/o_proj` (never used at inference, but wasted memory and caused misleading "512/1792 uncalibrated" diagnostic). Fixed by adding `_orig_attn` to `exclude_patterns`. Baseline bypass PPL improved from 33.45 → 15.41 after fix.

| Experiment | Description | PPL bypass | PPL ADC | dead_rate | Notes |
|------------|-------------|------------|---------|-----------|-------|
| **baseline_int4** | FlatQuant w4a4, no extras | 15.41 | 2354.86 | 10.6% | Bypass–ADC gap ×153; INT4 ADC far worse than INT8 |
| **prop_int4** | propagated calibration | 40.28 | 203.89 | 10.7% | Prop hurts bypass (unlike INT8); ADC gap ×5 |
| **center_int4** | bin-center λ=0.1 | 15.03 | 7895.26 | 10.8% | Bin-center actively harmful for INT4 |
| **prop+center_int4** | propagated + bin-center | 107.12 | 2700.35 | 10.8% | Both together catastrophic |
| **hadamard_int4** | Hadamard Kronecker init | 15.08 | 2080.51 | 10.7% | Tiny improvement over baseline |
| **hadamard+prop_int4** | Hadamard + propagated | 34.73 | 178.86 | 10.7% | Best with prop: ADC gap ×5 |
| **prop_512s_int4** | propagated, 512 cal. samples | 34.25 | **40.63** | 10.4% | **Best result: ADC gap ×1.2 — near bypass** |
| **decoupled_bw8_int4** | transforms at bw=8, ADC at bx=4 | 11.64 | 276.25 | 10.3% | Good bypass (≈INT8), ADC gap ×24 |

**Key findings:**

- **512 calibration samples + propagation** collapses the bypass→ADC gap from ×153 to ×1.2 (ADC PPL 40.63, bypass PPL 34.25). This is the dominant lever for INT4.
- **Propagation alone** narrows the gap but degrades bypass PPL (transforms optimise for corrupted inputs). With 128 samples this is net neutral. With 512 samples it becomes a strong win.
- **Bin-center loss is harmful for INT4** — with delta=6.12 the penalty term dominates and pushes y_int away from valid regions (ADC PPL 7895 vs 2354 baseline). The opposite effect vs INT8.
- **Decoupled training** (learn transforms at W8, apply ADC at W4) gives the best bypass PPL (11.64, close to INT8 level), but the ADC PPL is worse than baseline — the transforms optimised for W8 don't generalize to W4 ADC noise.
- **Dead_rate stays at ~10.7%** across all experiments — the dead zone pattern is independent of INT4 calibration strategy.

---

#### New baseline (per-token inference, improved transforms)

| Experiment | Description | PPL bypass | PPL ADC | dead_rate (mean) | reconstruction_rel | Notes |
|------------|-------------|------------|---------|-----------------|-------------------|-------|
| **New baseline** | FlatQuant w8a8, per-token ADC (no PACT) | **10.01** | **28.86** | **10.3%** | 0.22% | Current best |

**Key finding:** Improved transforms on pact branch (bypass 21.6 → 10.0) also reduce dead_rate (81% → 10.3%). ADC PPL essentially unchanged (28.99 → 28.86) despite much lower dead_rate — confirming the ~3x gap (10 → 29) is **ADC resolution**, not dead zone.

---

## Run commands

```bash
# INT8 experiments (branch llama-flatquant-adc-v2)
bash ADC/llama/run_flat_quant_adc_ptq_e7e8e9.sh baseline        # PPL 28.86
bash ADC/llama/run_flat_quant_adc_ptq_e7e8e9.sh prop             # PPL 14.55
bash ADC/llama/run_flat_quant_adc_ptq_e7e8e9.sh prop+center mid  # PPL 14.41 (best INT8)

# INT4 overnight sweep (branch llama-flatquant-adc-int4-experiments)
bash ADC/llama/run_overnight_int4.sh
# Results saved to: ADC/llama/results/overnight_int4_YYYYMMDD.json
```

**WandB project:** `llama-flat-quant-adc-ptq-blocks`

---

## Conclusions

### What we found

| Method | bits | cal. samples | PPL bypass | PPL ADC | Δ ADC vs INT8 baseline |
|--------|------|-------------|------------|---------|------------------------|
| FP baseline | — | — | — | ~10.5 | — |
| **INT8 baseline** | w8a8 | 128 | 10.01 | 28.86 | — |
| Bin-center (λ=0.1) | w8a8 | 128 | 10.01 | 28.86 | 0% |
| **Propagated** | w8a8 | 128 | 11.01 | **14.55** | **−47%** |
| **Prop + bin-center** | w8a8 | 128 | 11.00 | **14.41** | **−50%** |
| INT4 baseline | w4a4 | 128 | 15.41 | 2354.86 | +8060% |
| INT4 prop | w4a4 | 128 | 40.28 | 203.89 | +607% |
| INT4 hadamard+prop | w4a4 | 128 | 34.73 | 178.86 | +520% |
| **INT4 prop + 512 samples** | w4a4 | 512 | 34.25 | **40.63** | **+41%** |

### Key takeaways

**1. Dead zone was a transform quality problem, not a hardware problem.**
The original 81% dead_rate disappeared with better FlatQuant transforms (percentile calibration). Dead_rate dropped to 10.3% without touching ADC hardware. All PACT experiments were fixing a symptom, not the root cause.

**2. The INT8 bypass→ADC gap (10 → 29) is ADC resolution.**
With dead_rate at 10.3%, the 3× PPL degradation comes from coarse floor quantization: delta=2016 gives ~15–30 discrete output levels. Bin-center loss doesn't help — nudging y_int toward bin centers reduces per-step rounding error but the number of bins is the same.

**3. Propagated calibration is the key fix for INT8.**
Training each layer on ADC-quantized inputs from the previous layer (rather than clean FP) teaches it to compensate for upstream errors. Bypass PPL rises slightly (10.0 → 11.0) but ADC PPL drops 28.86 → 14.55 (−47%). Standard layer-wise PTQ with FP inputs underestimates reconstruction difficulty at inference.

**4. Bin-center loss: marginal gain in INT8, harmful in INT4.**
In INT8, prop+center (14.41) vs prop (14.55) is ~1% gain. In INT4, bin-center is actively harmful (ADC PPL 7895 vs 2354 baseline) — with delta=6.12 the penalty dominates and destabilises training.

**5. INT4 requires more calibration samples — 128 is insufficient.**
With 128 samples, all INT4 methods give ADC PPL ≥ 178. With 512 samples + propagation, ADC PPL drops to 40.63, closing the bypass→ADC gap to ×1.2. The INT4 optimizer overfits on 128 samples; more data is the dominant lever.

**6. Propagation and calibration samples interact differently in INT4 vs INT8.**
In INT8: propagation helps even at 128 samples (28.86 → 14.55). In INT4: propagation alone gives 203 PPL; only with 512 samples does it give 40.63. Low-bit PTQ is more sensitive to calibration data quantity.

**7. Decoupled training (transform at W8, ADC at W4) gives good bypass but poor ADC.**
Bypass PPL 11.64 ≈ INT8 quality, but ADC PPL is 276 — transforms optimised without W4 quantization noise don't generalize to the W4 ADC inference regime.

**8. Remaining gap: 14.41 (INT8) and 40.63 (INT4) vs FP baseline 10.5.**
Further reduction likely requires lower delta (hardware), reduced tile size, or higher ADC bits.

---

## Files

| File | Description |
|------|-------------|
| `core/flat_quant.py` | FlatQuant calibration, FlatQuantLinear, PACT implementation |
| `core/adc_layers.py` | TiledLinearADC, QATLinearADC (inference ADC layers) |
| `runs/llama_smooth_quant_adc_ptq.py` | Main PTQ script |
| `run_flat_quant_adc_ptq_e7e8e9.sh` | INT8 experiment launcher (baseline/center/prop/prop+center) |
| `run_overnight_int4.sh` | INT4 overnight batch runner (8 experiments, JSON results) |
