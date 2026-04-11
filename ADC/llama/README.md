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

#### Branch `llama-flatquant-adc-int4-experiments` — INT4 v1 overnight sweep (128–512 samples)

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

#### Branch `llama-flatquant-adc-int4-v2` — partial propagation sweep (1024 samples)

**New feature:** `propagate_quant_alpha` α ∈ [0,1] — dual-forward mixing:
`loss = (1−α)·MSE(layer(fp_inp), ref) + α·MSE(layer(quant_inp), ref)`
α=1.0 = full propagation (v1), α=0.0 = no propagation (baseline), α=0.5 = equal mix.

**2-stage:** stage A = 30 epochs no prop (good transforms) → stage B = 10 epochs α=0.5 (robustness fine-tune, lr×0.1).

| Experiment | Description | PPL bypass | PPL ADC | gap (ADC/bypass) | dead_rate |
|------------|-------------|------------|---------|-----------------|-----------|
| **baseline_1024** | no prop, 1024 samples | 19.80 | 56.83 | ×2.87 | 10.4% |
| **prop_full_1024** | α=1.0, 1024 samples | 30.32 | 40.20 | ×1.32 | 10.4% |
| **propalpha_075_1024** | α=0.75, 1024 samples | 27.46 | 35.92 | ×1.31 | 10.4% |
| **propalpha_05_1024** | α=0.5, 1024 samples | 24.24 | **31.22** | **×1.29** | 10.4% |
| **propalpha_025_1024** | α=0.25, 1024 samples | 21.65 | 32.35 | ×1.49 | 10.4% |
| **hadamard_propalpha05_1024** | Hadamard + α=0.5 | 24.63 | 33.12 | ×1.34 | 10.4% |
| **2stage_sb10_pa05** | stage A no prop → stage B α=0.5 | 22.82 | 32.21 | ×1.41 | 10.3% |
| **add_diag_propalpha05_1024** | bounded LET + α=0.5 | **20.23** | **27.56** | **×1.36** | 10.4% |

**Key findings:**

- **add_diag + α=0.5 gives best absolute INT4 PPL (27.56)** — diagonal scaling (bounded LET) improves both bypass (24.24→20.23) and ADC (31.22→27.56). The bypass/ADC gap is slightly wider (×1.36 vs ×1.29 for α=0.5 alone), but absolute numbers are best across all INT4 experiments.
- **α=0.5 without diag is the best gap ratio (×1.29)** — diagonal scaling adds expressive power but also slightly over-specialises transforms.
- **Below α=0.5, ADC gets worse** — at α=0.25, ADC PPL rises to 32.35 despite better bypass (21.65). When FP input dominates the loss, transforms don't adapt sufficiently to ADC-corrupted inference inputs.
- **Hadamard init gives no benefit** — hadamard+prop_α05 (33.12) is worse than random+prop_α05 (31.22) at the same α. Result consistent across v1 and v2.
- **2-stage is competitive** — bypass=22.82, ADC=32.21, gap ×1.41. Better bypass than α=0.5 but slightly worse ADC.
- **1024 vs 512 samples**: prop_full_1024 (ADC=40.20) vs prop_512s (ADC=40.63) — marginal. The α-mixing (→31.22) matters more than extra samples beyond 512.

---

#### Branch `llama-flatquant-adc-int4-v3` — layer-wise alpha + selective diag (1024 samples)

**New features:**
- `--fq_prop_alpha_early` / `--fq_prop_late_start` — use different α for early layers (0..N-1) vs late layers
- `--fq_diag_attn` / `--fq_diag_mlp` — train `diag_scale` only for attention or MLP blocks respectively

**Layer-wise α only (no diag, clean from first run):**

| Experiment | Description | PPL bypass | PPL ADC | gap | dead_rate |
|------------|-------------|------------|---------|-----|-----------|
| **propalpha_early025_late05** | α: 0.25 early / 0.5 late | 23.26 | 32.98 | ×1.42 | 10.4% |
| **propalpha_early05_late075** | α: 0.5 early / 0.75 late | 28.14 | 33.49 | ×1.19 | 10.4% |

**Selective diag + layer-wise α (fixed run):**

| Experiment | Description | PPL bypass | PPL ADC | gap | dead_rate |
|------------|-------------|------------|---------|-----|-----------|
| **repro_diagboth_alpha05** | both diag + flat α=0.5 (control) | 19.69 | **28.46** | ×1.45 | 10.4% |
| **diagattn_propalpha05** | attn diag only + flat α=0.5 | 24.98 | 29.41 | ×1.18 | 10.4% |
| **diagmlp_propalpha05** | MLP diag only + flat α=0.5 | **19.37** | 31.65 | ×1.63 | 10.4% |
| **diagmlp_early025_late05** | MLP diag + α 0.25/0.5 | **19.00** | 28.66 | ×1.51 | 10.4% |
| **diagattn_early025_late05** | attn diag + α 0.25/0.5 | 22.71 | 30.53 | ×1.34 | 10.4% |
| **diagboth_early025_late05** | both diag + α 0.25/0.5 | 19.39 | **28.55** | ×1.47 | 10.4% |

**Key observations:**
- **Control reproduced v2:** repro gives bypass=19.69, ADC=28.46 vs v2's 20.23/27.56 — within run-to-run variance. Bug fix confirmed.
- **MLP diag alone has a split personality:** best bypass of all (19.37) but worst ADC (31.65). MLP diag flattens weight distributions well (low bypass PPL) but without attn diag the ADC gap widens.
- **Attn diag alone is worse than both:** bypass=24.98 (much worse than both=19.69). Attn diag does not drive the bypass improvement — that comes from MLP diag.
- **Layer-wise α rescues MLP diag for ADC:** diagmlp alone → ADC=31.65; diagmlp + early025/late05 → ADC=28.66. Adding less propagation in early layers compensates for the attn-diag absence.
- **diagmlp_early025_late05 has best bypass (19.00) and ADC competitive with both-diag (28.66 vs 28.46)** — with half the diag parameters trained.
- **diagboth + layer-wise α ≈ flat α:** 28.55 vs 28.46 — no benefit from layer-wise α when both diag are active.
- **No experiment beats v2 best (27.56)** — all results cluster at ADC=28.4–31.6. The v2 `add_diag + α=0.5` result may have benefited from a favorable random seed.

---

#### Branch `llama-flatquant-adc-int4-v4` — MLP diag split + staged selective diag (1024 samples)

**Motivation:** v3 showed MLP diag improves bypass, attn diag closes ADC gap. Two questions:
1. Inside MLP diag: is it `up_gate_trans` or `down_trans` that drives the effect?
2. Is staged training (MLP diag first, then attn diag) better than training both simultaneously?

**New features:**
- `--fq_diag_mlp_up` / `--fq_diag_mlp_down` — split MLP diag into up_gate_trans vs down_trans
- `--fq_stage_b_diag_attn` / `--fq_stage_b_diag_mlp` — Stage B can use different diag than Stage A

| Experiment | Description | PPL bypass | PPL ADC | gap | dead_rate |
|------------|-------------|------------|---------|-----|-----------|
| Experiment | Description | PPL bypass | PPL ADC | gap | dead_rate |
|------------|-------------|------------|---------|-----|-----------|
| **diag_up_propalpha05** | up_gate_trans diag only + α=0.5 | 23.22 | 39.59 | ×1.70 | 10.3% |
| **diag_down_propalpha05** | down_trans diag only + α=0.5 | 20.42 | 31.26 | ×1.53 | 10.4% |
| **staged_mlpdiag_then_attn** | Stage A: MLP diag no prop → Stage B: attn diag + α=0.5, 10ep | **18.50** | **27.60** | **×1.49** | 10.4% |

**Key observations:**
- **`down_trans` diag is the key driver, not `up_gate_trans`:** `diag_down` gives bypass=20.42, ADC=31.26 — close to full MLP diag (19.37/31.65 from v3). `diag_up` gives bypass=23.22, ADC=39.59 — much worse. `down_proj` sits right before the ADC-quantized accumulation; its scale directly affects the z=y_int/delta distribution.
- **Staged training is the new best:** `staged_mlpdiag_then_attn` achieves bypass=18.50, ADC=27.60 — best bypass across all INT4 experiments, and ADC PPL matching v2 best (27.56). Two-phase training works: Stage A (no prop) learns clean MLP-diag transforms without ADC-noise interference; Stage B (attn diag + prop) adds robustness without destroying Stage A's gains.
- **Stage A `fq_propagate_quant=false` confirmed in JSON** — bypass=18.50 is the best INT4 bypass seen, consistent with clean FP training in Stage A.

---

#### Branch `llama-flatquant-adc-int4-v5` — stochastic propagation (1024 samples)

**Motivation:** deterministic α=0.5 mixes FP and quant losses in fixed proportion every batch. Stochastic propagation (QDrop-style) randomly varies the mix:
- **Bernoulli**: each batch randomly uses fp_inp OR quant_inp (single forward, 50/50). Cost = 1 forward vs 2 for deterministic dual.
- **Beta(β,β)**: each batch samples α ~ Beta(β,β), dual forward. β=2 → concentrated near 0.5; β=1 → uniform[0,1].

**New params:** `--fq_stochastic_prop`, `--fq_stochastic_mode {bernoulli,beta}`, `--fq_beta_param`

| Experiment | Description | PPL bypass | PPL ADC | gap | dead_rate |
|------------|-------------|------------|---------|-----|-----------|
| Experiment | Description | PPL bypass | PPL ADC | gap | dead_rate |
|------------|-------------|------------|---------|-----|-----------|
| **stoch_bern_propalpha05** | Bernoulli stochastic, flat, no diag | 19.25 | 35.38 | ×1.84 | 10.4% |
| **stoch_beta2_propalpha05** | Beta(2,2) stochastic, flat, no diag | 22.90 | 31.40 | ×1.37 | 10.4% |
| **staged_stoch_bern** | staged (MLP diag → attn diag) + Bernoulli Stage B | **16.94** | 32.44 | ×1.92 | 10.4% |
| **staged_stoch_beta2** | staged (MLP diag → attn diag) + Beta(2,2) Stage B | 17.16 | **27.82** | **×1.62** | 10.3% |

**Key observations:**
- **Beta(2,2) > Bernoulli across the board.** Bernoulli gives excellent bypass (19.25 flat / 16.94 staged) but poor ADC — the hard 0/1 switching is too noisy for ADC-path training. Beta samples near 0.5 every batch → smooth interpolation → better ADC at the cost of bypass.
- **`staged_stoch_beta2` is the new best ADC PPL: 27.82** — marginal improvement over v4 staged (27.60) and v2 best (27.56). Bypass=17.16 is also among the best seen.
- **Bernoulli staged explodes the gap (×1.92):** bypass=16.94 (best ever) but ADC=32.44 — Stage B Bernoulli batches with fp_inp don't carry ADC signal at all; the attn diag ends up learning for FP distribution, not ADC-corrupted inputs.
- **Flat stoch_bern vs stoch_beta2:** Bernoulli bypass=19.25 is better than Beta bypass=22.90 but ADC is much worse (35.38 vs 31.40). Consistent with staged results.
- **Pattern:** Beta(2,2) ≈ deterministic α=0.5 in expectation, with extra randomness → slightly better ADC (31.40 vs 31.22 for flat; 27.82 vs 27.60 for staged). The stochasticity is mildly helpful but not a large effect.

---

#### Branch `llama-flatquant-adc-int4-v6` — Residual post-ADC LoRA (1024 samples)

**Motivation:** Pure PTQ approaches a plateau (~27.5–28.5 ADC PPL across v3–v5). Next step: apply low-rank residual correction on top of the frozen PTQ checkpoint.

**Design (residual / QLoRA-style):**
```
y = frozen_TiledLinearADC(x)              # exact calibrated ADC path, frozen
y += scaling * lora_B(lora_A(x.float()))  # FP32 residual, added AFTER ADC output
```
- LoRA params always in float32; base model frozen in fp16
- Gradients never touch `round_ste` or ADC clamp → stable training
- Trained via LM cross-entropy, 5 epochs, lr=1e-4

**New params:** `--lora_rank`, `--lora_alpha`, `--lora_target_modules`, `--lora_epochs`, `--lora_lr`

**Base PTQ:** `staged_mlpdiag_then_attn` → bypass=18.50, ADC=27.60

| Experiment | LoRA targets | rank | PPL bypass | PPL ADC | gap | dead_rate |
|------------|-------------|------|------------|---------|-----|-----------|
| *staged_mlpdiag_then_attn (base PTQ)* | — | — | 18.50 | 27.60 | ×1.49 | 10.4% |
| **lora_r4_down** | down_proj | 4 | 19.33 | **16.32** | **×0.84** | 10.4% |
| **lora_r8_down** | down_proj | 8 | 19.17 | **15.79** | **×0.82** | 10.3% |
| **lora_r4_down_o** | down_proj + o_proj | 4 | 18.68 | **15.63** | **×0.84** | 10.3% |
| **lora_r4_all** | all 7 projections | 4 | **17.73** | 16.53 | ×0.93 | 10.3% |

**Analysis:**

- **Residual LoRA works extremely well.** ADC PPL drops from 27.60 → 15.63 (best), a ×1.77 improvement over the base PTQ plateau. This breaks through the ~27.5–28.5 wall that pure PTQ couldn't cross.
- **ADC PPL < bypass PPL across all runs** (gap < 1.0). The LoRA residual specifically learns to compensate ADC quantization error — it is a net benefit in ADC mode and slightly hurts bypass (19.33 vs 18.50) because the correction "overshoots" for the clean signal. This is expected and desirable: the adapter has learned to model the ADC noise.
- **`lora_r4_down_o` is the best ADC checkpoint** (15.63): adding `o_proj` alongside `down_proj` helps because attn output also feeds into the residual stream and carries ADC distortion.
- **Rank 8 vs rank 4** (`lora_r8_down` vs `lora_r4_down`): modest improvement (15.79 vs 16.32) — most of the gain is already captured at rank=4.
- **`lora_r4_all`** gives the best bypass (17.73) but slightly worse ADC (16.53 vs 15.63) than `lora_r4_down_o` — covering all projections spreads capacity across both paths instead of concentrating on ADC correction.
- **dead_rate unchanged** (~10.3–10.4%): LoRA doesn't touch the base weights, so the ADC dead-zone structure is preserved.

---

#### Branch `llama-flatquant-adc-int4-v7` — ADC-LoRA ablation study

**Motivation:** v6 residual LoRA broke the PTQ plateau (ADC 27.60 → 15.63). Now we need to validate and characterise the result before drawing conclusions.

**Six ablation groups:**

**Notes on design:** Each run re-trains PTQ + LoRA from scratch. `base_staged_no_lora` provides a within-v7 control. Groups B/C/D/E/F use `--seed 42` to reduce PTQ variance while keeping runs independent.

| Group | Question | Experiments |
|-------|---------|-------------|
| **CTRL** | Is PTQ base stable in this branch? | `base_staged_no_lora` |
| **A. Seed reproducibility** | Is ADC=15.63 stable end-to-end? | seed1/2/3 × r4_down_o |
| **B. Rank sweep** | Does correction saturate at r=4? | r∈{1,2,4,8} × down+o, α=2r |
| **C. Loss: CE vs CE+KL** | Does teacher KL improve over plain CE? | r4_down_o, ce_kl |
| **E. Pre-ADC vs Post-ADC** | Before or after ADC clamp is better? | pre_adc r4_down_o |
| **D. Layer-selective** | Is error in later layers? | last-8 vs first-8 × down+o |
| **F. Full coverage** | CE then CE+KL on all 7 projections | r4_all_ce, r4_all_ce_kl |

**New CLI params:** `--lora_mode {residual,pre_adc}`, `--lora_layer_indices`, `--lora_loss {ce,ce_kl}`, `--lora_kl_weight`, `--lora_kl_temperature`

##### Group A — Seed reproducibility

| Experiment | seed | PPL bypass | PPL ADC | gap | Notes |
|------------|------|------------|---------|-----|-------|
| seed1_r4_down_o | 1 | — | **15.79** | — | |
| seed2_r4_down_o | 2 | — | **15.88** | — | |
| seed3_r4_down_o | 3 | — | **15.93** | — | σ≈0.07 across 3 seeds |

##### Group B — Rank sweep (down_proj + o_proj)

| Experiment | rank | PPL bypass | PPL ADC | gap | Notes |
|------------|------|------------|---------|-----|-------|
| rank1_down_o | 1 | — | 15.90 | — | |
| rank2_down_o | 2 | — | 15.60 | — | |
| rank4_down_o | 4 | — | **15.57** | — | |
| rank8_down_o | 8 | — | **15.24** | — | marginal gain r4→r8 |

##### Group C — Loss ablation

| Experiment | loss | PPL bypass | PPL ADC | gap | Notes |
|------------|------|------------|---------|-----|-------|
| r4_down_o (v6 reference) | CE | 18.68 | 15.63 | ×0.84 | v6 best |
| r4_down_o_ce_kl | CE+KL | 19.54 | **14.33** | **×0.73** | +1.3 PPL improvement over CE |

##### Group D — Layer-selective LoRA (r=4, down+o)

| Experiment | layers | PPL bypass | PPL ADC | gap | Notes |
|------------|--------|------------|---------|-----|-------|
| r4_down_o (all layers) | 0–15 | 18.68 | 15.63 | ×0.84 | v6 best |
| r4_down_o_last8 | 8–15 | 18.62 | 20.08 | ×1.08 | worse than all-layers |
| r4_down_o_first8 | 0–7 | 18.42 | **16.64** | **×0.90** | first 8 carry more ADC error |

##### Group E — Pre-ADC vs Post-ADC LoRA (r=4, down+o)

| Experiment | mode | PPL bypass | PPL ADC | gap | Notes |
|------------|------|------------|---------|-----|-------|
| lora_r4_down_o (residual) | post-ADC | 18.68 | 15.63 | ×0.84 | v6 best |
| pre_adc_r4_down_o | pre-ADC (RAOQ-style) | 123.34 | 105.63 | ×0.86 | **catastrophic divergence** |

##### Group F — Full coverage (CE then CE+KL)

| Experiment | targets | loss | PPL bypass | PPL ADC | gap | Notes |
|------------|---------|------|------------|---------|-----|-------|
| lora_r4_all (v6, CE) | all 7 | CE | 17.73 | 16.53 | ×0.93 | v6 reference |
| r4_all_ce | all 7 | CE | 17.62 | 16.68 | ×0.95 | |
| r4_all_ce_kl | all 7 | CE+KL | 19.13 | **14.03** | **×0.73** | **new best** |

##### Control

| Experiment | LoRA | PPL bypass | PPL ADC | Notes |
|------------|------|------------|---------|-------|
| base_staged_no_lora | none | 17.89 | 27.55 | consistent with v6 (27.60) |

**Analysis:**

- **Best result: `r4_all_ce_kl` ADC=14.03** — new overall best, beating v6's 15.63 by 1.6 PPL. CE+KL loss with all 7 projections is the winning combination.
- **CE+KL consistently beats CE** (+1.3 PPL on down+o: 15.57→14.33; +2.65 PPL on all targets: 16.68→14.03). The FP teacher signal provides information that CE on the quantized model alone cannot.
- **Rank saturation at r=1** (15.90 vs 15.24 at r=8 — only 0.66 PPL difference across all ranks). Most of the correction capacity is captured at rank=1; higher ranks give diminishing returns.
- **Seeds stable** (σ≈0.07 across seeds 1/2/3, all ~15.8 ADC PPL). The end-to-end pipeline (PTQ + LoRA) is reproducible.
- **Pre-ADC LoRA fails** (bypass=123.34, ADC=105.63 — catastrophic divergence). Even with fp32 parameters, gradients through `round_ste(Qw(W+ΔW))` and the ADC clamp are too noisy for CE loss from step 0 without MSE warmup. This confirms residual post-ADC is the correct architecture.
- **First 8 layers beat last 8** (16.64 vs 20.08 ADC PPL). ADC distortion is stronger in early layers — they accumulate errors that propagate through the rest of the network. Concentrating LoRA on layers 8–15 misses the main source of error.
- **PTQ control validates v7 branch** (bypass=17.89, ADC=27.55 vs v6 baseline 18.50/27.60 — within expected PTQ variance).

---

#### Branch `llama-flatquant-adc-int4-v8` — C4 generalisation check (best-3 LoRA configs)

**Motivation:** v7 validated LoRA configs on wikitext2. v8 re-runs the top-3 with both wikitext2 and C4 evaluation to check whether LoRA correction generalises to out-of-domain web text.

| Experiment | targets | loss | rank | wiki bypass | wiki ADC | C4 bypass | C4 ADC | Notes |
|------------|---------|------|------|-------------|----------|-----------|--------|-------|
| base_no_lora | none | — | — | *pending* | *pending* | *pending* | *pending* | PTQ control |
| r4_all_ce_kl | all 7 | CE+KL | 4 | *pending* | *pending* | *pending* | *pending* | v7 best (wiki ADC=14.03) |
| r4_down_o_ce_kl | down+o | CE+KL | 4 | *pending* | *pending* | *pending* | *pending* | v7 2nd (wiki ADC=14.33) |
| rank8_down_o | down+o | CE | 8 | *pending* | *pending* | *pending* | *pending* | v7 3rd (wiki ADC=15.24) |

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

# INT4 v1 sweep (branch llama-flatquant-adc-int4-experiments)
bash ADC/llama/run_overnight_int4.sh

# INT4 v2 sweep: partial propagation α-sweep + 2-stage + bounded LET
bash ADC/llama/run_overnight_int4_v2.sh
# Results: ADC/llama/results/overnight_int4_v2_YYYYMMDD.json

# INT4 v3 sweep: layer-wise alpha + selective diag (run after bug fix)
bash ADC/llama/run_overnight_int4_v3.sh
# Results: ADC/llama/results/overnight_int4_v3_YYYYMMDD.json

# INT4 v4 sweep: MLP diag split (up vs down) + staged selective diag
bash ADC/llama/run_overnight_int4_v4.sh
# Results: ADC/llama/results/overnight_int4_v4_YYYYMMDD.json

# INT4 v5 sweep: stochastic propagation
bash ADC/llama/run_overnight_int4_v5.sh
# Results: ADC/llama/results/overnight_int4_v5_YYYYMMDD.json

# INT4 v6 sweep: ADC-LoRA post-correction
bash ADC/llama/run_overnight_int4_v6.sh
# Results: ADC/llama/results/overnight_int4_v6_YYYYMMDD.json

# INT4 v7 sweep: ADC-LoRA ablation (seeds, rank, loss, layers, pre vs post ADC)
bash ADC/llama/run_overnight_int4_v7.sh
# Results: ADC/llama/results/overnight_int4_v7_YYYYMMDD.json

# INT4 v8 sweep: best-3 LoRA configs, wikitext2 + C4 eval
bash ADC/llama/run_overnight_int4_v8.sh
# Results: ADC/llama/results/overnight_int4_v8_YYYYMMDD.json
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
| INT4 baseline (128s) | w4a4 | 128 | 15.41 | 2354.86 | +8060% |
| INT4 prop (128s) | w4a4 | 128 | 40.28 | 203.89 | +607% |
| INT4 prop (512s) | w4a4 | 512 | 34.25 | 40.63 | +41% |
| INT4 baseline (1024s) | w4a4 | 1024 | 19.80 | 56.83 | +97% |
| INT4 prop α=1.0 (1024s) | w4a4 | 1024 | 30.32 | 40.20 | +39% |
| INT4 prop α=0.75 (1024s) | w4a4 | 1024 | 27.46 | 35.92 | +24% |
| **INT4 prop α=0.5 (1024s)** | w4a4 | 1024 | 24.24 | **31.22** | **+8%** |
| INT4 2-stage α=0.5 (1024s) | w4a4 | 1024 | 22.82 | 32.21 | +12% |
| **INT4 add_diag + α=0.5 (1024s)** | w4a4 | 1024 | 20.23 | 27.56 | −5% |
| **INT4 staged: MLP diag → attn diag + α=0.5** | w4a4 | 1024 | **18.50** | **27.60** | **−4%** |

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
With 128 samples, all INT4 methods give ADC PPL ≥ 178. With 512–1024 samples + propagation, ADC PPL drops to 31–40, closing the gap substantially. The INT4 optimizer overfits on 128 samples; more data is the dominant lever.

**6. Partial propagation (α=0.5) is strictly better than full propagation for INT4.**
α-sweep (0.25–1.0) shows α=0.5 achieves the best ADC PPL (31.22) AND the best bypass/ADC gap (×1.29). Full propagation (α=1.0) overshoots: transforms over-specialise for corrupted inputs and lose bypass quality. Below α=0.5, FP input dominates the loss and transforms under-adapt to inference conditions.

**7. Hadamard init does not help.**
At the same α=0.5 and 1024 samples, Hadamard init (ADC=33.12) is consistently worse than random init (ADC=31.22). Result is consistent across v1 and v2. Not worth pursuing further.

**8. 2-stage training (no-prop → prop fine-tune) gives a good bypass/ADC trade-off.**
Stage A (no prop, good transforms) → Stage B (10ep α=0.5, robustness) gives bypass=22.82, ADC=32.21. Best bypass among α-sweep methods, competitive ADC. Useful if clean FP performance matters.

**9. Diagonal scaling (bounded LET) + α=0.5 gives best absolute INT4 result.**
add_diag + α=0.5 achieves bypass=20.23, ADC=27.56 — best absolute numbers across all INT4 experiments. The diagonal per-channel scale (bounded [1e-4, 10]) gives transforms more expressive power to adapt to W4A4 noise. Combined with α=0.5 partial propagation this is the strongest purely PTQ result.

**10. Remaining gap: 14.41 (INT8) and 27.56 (INT4) vs FP baseline 10.5.**
INT4 ADC PPL 27.56 is now close to INT8 baseline (28.86) — the extra complexity of INT4 has been largely compensated by better calibration. Further reduction requires hardware changes (lower delta, higher ba).

---

## Files

| File | Description |
|------|-------------|
| `core/flat_quant.py` | FlatQuant calibration, FlatQuantLinear, PACT implementation |
| `core/adc_layers.py` | TiledLinearADC, QATLinearADC (inference ADC layers) |
| `runs/llama_smooth_quant_adc_ptq.py` | Main PTQ script |
| `run_flat_quant_adc_ptq_e7e8e9.sh` | INT8 experiment launcher (baseline/center/prop/prop+center) |
| `run_overnight_int4.sh` | INT4 overnight batch runner (8 experiments, JSON results) |
| `run_overnight_int4_v2.sh` | INT4 v2: partial propagation α-sweep + 2-stage + bounded LET |
| `run_overnight_int4_v3.sh` | INT4 v3: layer-wise alpha + selective diag_scale per block type |
| `run_overnight_int4_v4.sh` | INT4 v4: MLP diag split (up_gate vs down) + staged selective diag |
| `run_overnight_int4_v5.sh` | INT4 v5: stochastic propagation (Bernoulli / Beta) |
| `run_overnight_int4_v6.sh` | INT4 v6: ADC-LoRA post-correction (rank 4/8, various targets) |
| `run_overnight_int4_v7.sh` | INT4 v7: ADC-LoRA ablation (seeds, rank, loss, layers, pre vs post ADC) |
| `run_overnight_int4_v8.sh` | INT4 v8: best-3 LoRA configs, wikitext2 + C4 generalisation eval |
| `core/adc_lora.py` | ResidualLoRATiledLinearADC, PreADCLoRATiledLinearADC, apply_adc_lora, calibrate_adc_lora |
