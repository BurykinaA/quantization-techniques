# ADC-Aware INT4 Quantization of Llama — Best Configuration

Clean reference implementation of the best-performing quantization pipeline for
Llama-3.2-1B on analog hardware with Analog-to-Digital Converters (ADCs).

The four configurations below tell the full story: how much PPL each component
of the pipeline costs, and how post-ADC LoRA correction recovers it.

---

## Results (Llama-3.2-1B)

### Unipolar ADC — 4-quadrant decomposition (this branch)

| Config | Description | Wiki PPL | C4 PPL | Latency (ms) | Tok/s |
|---|---|---|---|---|---|
| `fp` | Full precision (FP16) | **8.68** | **13.13** | 11.4 | 44888 |
| `int4_no_adc` | INT4 FlatQuant, no ADC floor | 12.56 | 20.13 | 74.8 | 6847 |
| `best_ptq` | INT4 FlatQuant + unipolar ADC | **18.08** | **28.41** | 444.0 | 1153 |
| `best_lora` | INT4 + unipolar ADC + LoRA | TBD | TBD | — | — |

Key observations:
- **INT4 cost alone** (no ADC): FP 8.68 → INT4 12.56 (+3.9 PPL)
- **Unipolar ADC overhead**: INT4 12.56 → INT4+ADC **18.08** (+5.5 PPL)
- **LoRA correction**: pending

### Bipolar ADC — previous baseline

| Config | Wiki PPL | C4 PPL |
|---|---|---|
| `best_ptq` | 26.78 | 45.40 |
| `best_lora` | **14.01** | **23.14** |

### Why unipolar PTQ is better: +8.7 PPL on WikiText2

The improvement from 26.78 → 18.08 comes from better ADC resolution per branch.

**Bipolar model (old):**
```
y_int ∈ [−M, +M],   M = tile_in · q_x · q_w = 256 · 7 · 7 = 12544
δ = 2M / (2^ba · k) = 25088 / (256 · 16) ≈ 6.12
```
The ADC must cover the full signed range [−12544, +12544] with 256 bins.
Typical dot products in a well-calibrated layer have σ ≈ 30–50, so most outputs
fall within a few dozen ADC levels — only ~0.5% of the range is actually used.

**Unipolar 4-quadrant (new):**

Signed inputs are split into positive and negative parts:
```
x = x⁺ − x⁻,   x⁺ = max(code_x, 0),   x⁻ = max(−code_x, 0)
w = w⁺ − w⁻,   w⁺ = max(code_w, 0),   w⁻ = max(−code_w, 0)
```

Four non-negative MVMs replace the single signed one:
```
y = x⁺·w⁺ + x⁻·w⁻ − x⁺·w⁻ − x⁻·w⁺
```

Each branch is guaranteed ≥ 0, so the ADC fits its range exactly:
```
δ_branch = tile_in · qmax_a · qmax_b / ((2^ba − 1) · k)
         = 256 · 7 · 7 / (255 · 16) ≈ 3.06     (2× finer than bipolar)
```

There is no DC offset: the four branch outputs cancel the zero-point bias
exactly in the digital domain, so no ADC bins are wasted on a constant baseline.
Each branch's full 256-level range is used for signal, not offset.

Combined effect: effective ADC resolution doubles across all four branches,
which directly reduces the dead-zone fraction and reconstruction error.

---

## How to Run

```bash
# Run all four configs (takes several hours — each config trains from scratch)
bash run.sh

# Run only specific configs
bash run.sh --configs best_lora

# Custom output directory
bash run.sh --output_dir /data/results
```

Results are printed as a table and saved to `outputs/results.json`.

---

## Hardware Model

The target hardware uses a matrix-vector multiply (MVM) unit that performs
the dot product in integer arithmetic and reads the result through an ADC.

### Tiled Integer MVM

A weight matrix **W** ∈ ℝ^{out × in} is split into tiles of width `mvm_limit`
(number of columns the MVM unit can process in one shot):

```
tile width  = mvm_limit = 256 columns
tile height = out_features (all output channels in one tile)
```

For each activation vector **x** ∈ ℝ^{in}:

1. **Quantize activations:**  
   `x_int = clamp(round(x / s_x), −q_x, q_x)`  
   where `s_x` is the per-token scale and `q_x = 2^{b_x−1} − 1` (e.g. 7 for INT4)

2. **Quantize weights:**  
   `W_int = clamp(round(W / s_w), −q_w, q_w)` per channel  
   where `q_w = 2^{b_w−1} − 1`

3. **Integer dot product per tile:**  
   `z_int = x_int_tile · W_int_tile^T`  
   Result range: `|z_int| ≤ tile_in · q_x · q_w`

4. **ADC floor quantization (bipolar model):**
```
z = clamp(floor(z_int / δ),  na, pa)     na = −2^(ba−1),  pa = 2^(ba−1) − 1
```
e.g. for ba=8: na=−128, pa=127  →  z ∈ [−128, 127]

5. **Unipolar ADC (optical hardware, this branch):**

The physical device accepts only **non-negative** weights and activations and
reads non-negative codes `[0, 2^ba − 1]`.  We use a zero-point shift: shift
both codes to non-negative before the MVM, then subtract correction terms
digitally after the ADC:

```
q_x = 2^(b_x−1) − 1        # e.g. 7 for INT4 (magnitude of qmin)
q_w = 2^(b_w−1) − 1

x_pos = code_x + q_x        # [-q_x, q_x]  →  [0, 2·q_x]
W_pos = code_w + q_w        # [-q_w, q_w]  →  [0, 2·q_w]

y_pos = x_pos · W_pos^T     # ∈ [0, tile_in · (2q_x) · (2q_w)]  — always ≥ 0 ✓
```

ADC reads `y_pos` with `δ_uni = 2·δ` (range is 2× wider on the positive side):
```
z_pos = clamp(floor(y_pos / δ_uni), 0, 2^ba − 1)   # unipolar ADC output [0, 255]
```

Digital correction (subtracted after ADC, no hardware cost):
```
correction = q_x · Σ_j W_int_j   (per output channel, precomputed)
           + q_w · Σ_j x_int_j   (per token, cheap)
           + q_x · q_w · tile_in  (scalar constant)

y_int ≈ z_pos · δ_uni − correction   →   dequantize as usual
```

### Delta (ADC Resolution)

Delta is a fixed hardware constant that determines how many integer values the
ADC can distinguish:

```
δ = 2 · tile_in · q_x · q_w / (2^{b_a} · k)
```

| Config | b_x | b_w | b_a | k | tile_in | δ | Useful bins in ±σ_z |
|--------|-----|-----|-----|---|---------|---|---------------------|
| INT8   | 8   | 8   | 8   | 16 | 256 | **≈ 2016** | 15–30 |
| INT4   | 4   | 4   | 8   | 16 | 256 | **≈ 6.12** | ~1 |

With INT4, δ ≈ 6.12 means that the floating-point range of the dot product is
compressed into very few ADC levels. This is the root cause of the large gap
between `int4_no_adc` and `best_ptq`.

### Dead Zone

When `|z_int| < δ`, the floor quantization maps it to 0 — the ADC contribution
is completely lost. This is the **dead zone**. The fraction of outputs in the
dead zone is called `dead_rate`.

With poor calibration (early experiments): `dead_rate ≈ 81%`.  
After better FlatQuant calibration (percentile method): `dead_rate ≈ 10%`.

---

## FlatQuant Transforms

### Motivation

Per-token L∞ scaling (`x_int = round(x / amax(x))`) concentrates all the
quantization range on the largest feature in each token. All other features
are quantized with many fewer levels or fall into the dead zone.

**FlatQuant** learns invertible linear transforms that redistribute the
activation energy uniformly across channels before quantization.

### Kronecker Decomposition

Each transform is factored as a Kronecker product:

```
T = kron(L, R) ≈ L ⊗ R
```

where L ∈ ℝ^{√d × √d} and R ∈ ℝ^{√d × √d} are approximately orthogonal
matrices parameterized via a Cayley transform. The Kronecker structure
reduces the parameter count from O(d²) to O(d) while preserving expressiveness.

**Diagonal scaling** adds a learnable per-channel scale d ∈ [10⁻⁴, 10]:

```
T_diag = diag(d) · kron(L, R)
```

### Calibration Loss

For each transformer block, the transforms are trained to minimize the
reconstruction error between the block's FP output and its quantized output:

```
L_calib = MSE( block_fp(x), block_quant(T·x) )
```

where `block_quant` includes the integer dot product and ADC floor quantization.

### Staged Training (Two-Phase)

**Stage A (30 epochs):**
- Train Kronecker transforms for MLP blocks only
- No propagation — each layer sees clean FP16 inputs from upstream
- Establishes good bypass PPL (transform quality)

**Stage B (10 epochs, lr × 0.1):**
- Adds diagonal scaling for attention blocks
- Enables **partial propagation** with α = 0.5:

```
x_cal = α · x_adc + (1 − α) · x_fp
```

This dual-forward trains each layer on a mix of FP inputs and ADC-quantized
inputs from the previous layer. α = 0.5 is the sweet spot — α = 1.0
over-specializes transforms and degrades bypass quality; α = 0 ignores ADC.

---

## Post-ADC LoRA Correction

### Motivation

After the best PTQ (best_ptq ≈ 27.6 PPL), further FlatQuant improvements yield
< 0.5 PPL gain. The root cause is that **4-bit weight quantization introduces a
systematic residual error** that transform calibration cannot recover — transforms
can redistribute the signal but cannot add back information lost to INT4 rounding.

### Architecture

For each target linear layer, LoRA wraps the TiledLinearADC with a residual:

```
y = TiledLinearADC(x) + (α / r) · B(A(x.float()))
```

- **A** ∈ ℝ^{r × in} and **B** ∈ ℝ^{out × r} are FP32 parameters  
- **r = 4** (rank), **α = 8.0** → scaling = 2.0  
- **B** is initialized to zeros → initial output equals the frozen ADC model  
- Correction is added **after** the ADC floor, so gradients never flow through
  the non-differentiable floor() — training is numerically stable

### Training Loss

```
L = L_CE + λ · KL(student ‖ teacher_fp)
```

where:
- **L_CE** = cross-entropy on the next-token prediction task  
- **KL** = KL divergence between student softmax and FP teacher softmax  
- **λ = 0.5**, temperature **T = 2.0**  
- **teacher_fp** is a frozen copy of the original FP16 model

The KL term acts as a teacher signal that pulls the student's output distribution
toward the FP baseline, providing smoother gradients than CE alone (+1.3 PPL).

### Targets

All 7 projections are corrected:
- **MLP:** `down_proj`, `up_proj`, `gate_proj`
- **Attention:** `q_proj`, `k_proj`, `v_proj`, `o_proj`

Training: 5 epochs, lr = 1e-4, AdamW.

---

## Code Structure

```
ADC/best/
├── README.md          ← this file
├── run.sh             ← entry point: bash run.sh
├── configs.py         ← 4 config presets as dataclasses
├── pipeline.py        ← clean ~400-line pipeline
├── eval.py            ← PPL (sliding window) + latency
└── core/
    ├── adc_layers.py  ← TiledLinearADC: tiling, INT quant, ADC floor
    ├── adc_lora.py    ← ResidualLoRATiledLinearADC + calibrate_adc_lora
    ├── flat_quant.py  ← FlatQuant: Kronecker transforms, calibration loop
    └── grad_functions.py  ← round_ste, floor_ste (straight-through estimators)
```

### Key Classes

**`TiledLinearADC`** (`core/adc_layers.py`)  
Implements the hardware model: tiled INT4/INT8 MVM + ADC floor quantization.
Supports `set_bypass_adc(True)` to disable the floor for ablation.

**`ResidualLoRATiledLinearADC`** (`core/adc_lora.py`)  
Wraps TiledLinearADC with a FP32 LoRA residual. Post-ADC placement ensures
stable training. Pre-ADC placement (inside quantization) was tried and
catastrophically diverged (bypass PPL > 100).

**`FlatQuantLinear`** (`core/flat_quant.py`)  
Wrapper that applies learnable Kronecker transforms during calibration, then
`reparameterize_model()` bakes the transforms into the weight matrices so
inference has no overhead.

---

## References

- FlatQuant: [arxiv.org/abs/2410.09426](https://arxiv.org/abs/2410.09426)
- QLoRA: [arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314)  
- GPTQ perplexity evaluation methodology: [arxiv.org/abs/2210.17323](https://arxiv.org/abs/2210.17323)
