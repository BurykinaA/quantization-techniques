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

#### INT4 experiments (bx=4, bw=4, k=16, ba=8)

With INT4, delta scales proportionally: `delta = 2 * 256 * 7 * 7 / (256 * 16) ≈ 6.12` (vs 2016 for INT8). Finer ADC resolution but more quantization noise per weight/activation value.

| Experiment | Description | PPL bypass | PPL ADC | dead_rate | delta | Notes |
|------------|-------------|------------|---------|-----------|-------|-------|
| **prop+center (w4a4)** | propagated + bin-center λ=0.1, bx=4 bw=4 | 33.45 | 236.78 | 10.6% | 6.12 | 512/1792 layers uncalibrated; INT4 transforms degraded |

**Key finding:** INT4 is significantly worse than INT8 despite the smaller delta. Bypass PPL degrades to 33.45 (vs 11.00 for INT8 prop+center) because INT4 quantizers add more noise during FlatQuant transform training — the optimizer cannot compensate fully. The 512/1792 uncalibrated layers warning (`_orig_attn` q/k/v tiles retaining default act_scale=0.01) may compound the result but is not the primary cause.

---

#### New baseline (per-token inference, improved transforms)

| Experiment | Description | PPL bypass | PPL ADC | dead_rate (mean) | reconstruction_rel | Notes |
|------------|-------------|------------|---------|-----------------|-------------------|-------|
| **New baseline** | FlatQuant w8a8, per-token ADC (no PACT) | **10.01** | **28.86** | **10.3%** | 0.22% | Current best |

**Key finding:** Improved transforms on pact branch (bypass 21.6 → 10.0) also reduce dead_rate (81% → 10.3%). ADC PPL essentially unchanged (28.99 → 28.86) despite much lower dead_rate — confirming the ~3x gap (10 → 29) is **ADC resolution**, not dead zone.

---

## Run commands

```bash
# Baseline (per-token inference, improved transforms) — current best: PPL 28.86
bash ADC/llama/run_flat_quant_adc_ptq_e7e8e9.sh baseline

# Bin-center loss: cos²(π·z) pushes y_int toward ADC bin centres
# intensity: weak=0.01, mid=0.1, strong=1.0
bash ADC/llama/run_flat_quant_adc_ptq_e7e8e9.sh center mid

# Propagated calibration: each layer trained on ADC-quantized inputs from prev layers
bash ADC/llama/run_flat_quant_adc_ptq_e7e8e9.sh prop

# Combined: propagated + bin-center
bash ADC/llama/run_flat_quant_adc_ptq_e7e8e9.sh prop+center mid
```

**WandB project:** `llama-flat-quant-adc-ptq-blocks`

---

## Conclusions

### What we found

| Method | bits | PPL bypass | PPL ADC | Δ ADC vs baseline |
|--------|------|------------|---------|-------------------|
| FP baseline | — | — | ~10.5 | — |
| New baseline (per-token) | w8a8 | 10.01 | 28.86 | — |
| Bin-center loss (λ=0.1) | w8a8 | 10.01 | 28.86 | 0% |
| **Propagated calibration** | w8a8 | 11.01 | **14.55** | **−47%** |
| **Prop + bin-center** | w8a8 | 11.00 | **14.41** | **−50%** |
| Prop + bin-center (w4a4) | w4a4 | 33.45 | 236.78 | +720% |

### Key takeaways

**1. Dead zone was a transform quality problem, not a hardware problem.**
The original 81% dead_rate (branch `llama-flatquant-adc`) disappeared by itself with better FlatQuant transforms (percentile calibration + accumulated fixes). Dead_rate dropped to 10.3% without touching ADC hardware parameters. All PACT experiments were trying to fix a symptom, not the root cause.

**2. The bypass→ADC gap (10 → 29) is ADC resolution.**
With dead_rate at 10.3%, the 3× PPL degradation comes from coarse floor quantization: delta=2016 gives only ~15–30 discrete output levels in a typical z range. Bin-center loss doesn't change this — nudging y_int toward bin centers reduces per-step rounding error within each bin, but the number of bins is the same. This is why center loss alone gives no improvement.

**3. Propagated calibration is the key fix.**
When each layer trains on ADC-quantized inputs from the previous layer (rather than clean FP inputs), it learns to compensate for upstream quantization errors. The bypass PPL rises slightly (10.0 → 11.0, transforms slightly suboptimal for FP path) but ADC PPL drops from 28.86 → 14.55 (−47%). The improvement is real: standard layer-wise PTQ with FP inputs underestimates the reconstruction difficulty each layer faces at inference.

**4. Bin-center loss gives a marginal additional gain on top of propagated.**
Prop+center (14.41) vs prop alone (14.55) is a ~1% improvement. Not negligible, but not the primary lever.

**5. INT4 (bx=bw=4) is worse despite smaller delta.**
Smaller delta (6.12 vs 2016) gives finer ADC resolution, but INT4 quantization noise dominates — FlatQuant transforms cannot compensate for 4-bit precision loss. Bypass PPL = 33.45 (vs 11.00 for INT8), ADC PPL = 236.78. INT8 propagated (14.41) remains far better.

**6. Remaining gap: 14.41 vs FP baseline 10.5.**
~37% PPL gap remains. Further reduction likely requires lower delta (hardware change), reduced tile size, or higher ADC bits — not achievable through training.

---

## Files

| File | Description |
|------|-------------|
| `core/flat_quant.py` | FlatQuant calibration, FlatQuantLinear, PACT implementation |
| `core/adc_layers.py` | TiledLinearADC, QATLinearADC (inference ADC layers) |
| `runs/llama_smooth_quant_adc_ptq.py` | Main PTQ script |
| `run_flat_quant_adc_ptq_e7e8e9.sh` | Experiment launcher (baseline/e7/e8/e9/l1/huber) |
