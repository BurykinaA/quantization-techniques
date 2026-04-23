# ADC-Aware INT4 Quantization of Llama — Best Configuration

Clean reference implementation of the best-performing quantization pipeline for
Llama-3.2-1B on analog hardware with Analog-to-Digital Converters (ADCs).

The four configurations below tell the full story: how much PPL each component
of the pipeline costs, and how post-ADC LoRA correction recovers it.

---

## Results (Llama-3.2-1B)

### Unipolar ADC — unsigned shift-subtract (this branch)

| Config | Description | Wiki PPL | C4 PPL |
|---|---|---|---|
| `fp` | Full precision (FP16) | **8.68** | **13.13** |
| `int4_no_adc` | INT4 FlatQuant, no ADC floor | TBD | TBD |
| `best_ptq` | INT4 FlatQuant + unsigned ADC, k=16 | TBD | TBD |
| `best_ptq_k` | best_ptq + per-layer k search | TBD | TBD |
| `best_lora` | best_ptq + post-ADC LoRA correction | TBD | TBD |
| `best_lora_k` | best_lora + per-layer k search | TBD | TBD |

### Previous results

| Approach | Config | Wiki PPL | C4 PPL |
|---|---|---|---|
| Bipolar | `best_ptq` | 26.78 | 45.40 |
| Bipolar | `best_lora` | **14.01** | **23.14** |
| 4-quadrant unipolar | `best_ptq` | 18.08 | 28.41 |
| Unsigned unipolar, bipolar FQ cache | `best_ptq` | 94274.75 | 89773.24 |
| Unsigned unipolar, bipolar FQ cache | `best_ptq_k` | 16.78 | 27.59 |
| Unsigned unipolar, bipolar FQ cache | `best_lora` | 19.27 | 34.40 |
| Unsigned unipolar, bipolar FQ cache | `best_lora_k` | **13.42** | **21.94** |
| Unsigned unipolar, unipolar FQ (δ=14.06) | `best_ptq` | 39.00 | 67.78 |
| Unsigned unipolar, unipolar FQ (δ=14.06) | `best_ptq_k` | 21.64 | 33.79 |
| Unsigned unipolar, unipolar FQ (δ=14.06) | `best_lora` | 17.29 | 29.26 |
| Unsigned unipolar, unipolar FQ (δ=14.06) | `best_lora_k` | **14.56** | **23.78** |

### Unsigned shift-subtract vs alternatives

**Bipolar (original):**
```
y_int ∈ [−M, +M],   M = tile_in · q_x · q_w = 256 · 7 · 7 = 12544
δ = 2M / (2^ba · k) = 25088 / (256 · 16) ≈ 6.12
```
Range covers [−12544, +12544] but typical outputs have σ ≈ 30–50 — only ~0.5% of bins used.

**4-quadrant unipolar (previous):**
Split codes into pos/neg parts, 4 separate non-negative MVMs:
```
δ_branch = 256 · 7 · 7 / (255 · 16) ≈ 3.06   (finer δ, but 4× the compute)
```

**Unsigned shift-subtract (current):**
Shift both codes to non-negative range with zero-point, single MVM:
```
zp_x = 2^(bx−1) = 8,  zp_w = 2^(bw−1) = 8
code_x_u = code_x + zp_x   ∈ [0, 15]
code_w_u = code_w + zp_w   ∈ [0, 15]
y_uint   = code_x_u · code_w_u^T   ∈ [0, 256 · 15 · 15 = 57600]

δ_uni = tile_in · (2^bx−1) · (2^bw−1) / ((2^ba−1) · k)
      = 256 · 15 · 15 / (255 · 16) ≈ 14.1
```

Digital correction (exact, no approximation):
```
y_int = y_uint − zp_x · Σ_j code_w_u_j − zp_w · Σ_i code_x_u_i + tile_in · zp_x · zp_w
```

The unsigned approach has a coarser δ (14.1 vs 3.06) but is 4× faster — one MVM instead of four.
Per-layer k search compensates: layers with narrow output distributions can use larger k (finer δ).

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

5. **Unipolar ADC — unsigned shift-subtract (this branch):**

The physical device accepts only non-negative inputs. Shift both quantized codes
to the non-negative range with zero-points, then do a single unsigned MVM:

```
zp_x = 2^(b_x−1)         # e.g. 8 for INT4
zp_w = 2^(b_w−1)         # e.g. 8 for INT4

code_x_u = code_x + zp_x   # [−zp_x, zp_x−1]  →  [0, 2^bx − 1]
code_w_u = code_w + zp_w   # [−zp_w, zp_w−1]  →  [0, 2^bw − 1]

y_uint = code_x_u · code_w_u^T   # ∈ [0, tile_in · (2^bx−1) · (2^bw−1)] ≥ 0 ✓
```

ADC reads `y_uint` with resolution:
```
δ_uni = tile_in · (2^bx−1) · (2^bw−1) / ((2^ba−1) · k)
```

For INT4, k=16: `δ ≈ 256 · 15 · 15 / (255 · 16) ≈ 14.1`

```
z_uni = clamp(floor(y_uint / δ_uni), 0, 2^ba − 1)   # [0, 255]
```

Digital correction (exact; weight term precomputable, activation term per-token):
```
correction = zp_x · Σ_j code_w_u_j          (per output channel)
           + zp_w · Σ_i code_x_u_i          (per token)
           − tile_in · zp_x · zp_w          (scalar constant)

y_int = z_uni · δ_uni − correction   →   dequantize as usual
```

### Delta (ADC Resolution)

```
δ_uni = tile_in · (2^bx−1) · (2^bw−1) / ((2^ba−1) · k)
```

| Config | b_x | b_w | b_a | k | tile_in | δ |
|--------|-----|-----|-----|---|---------|---|
| INT8   | 8   | 8   | 8   | 16 | 256 | **≈ 2016** |
| INT4   | 4   | 4   | 8   | 16 | 256 | **≈ 14.1** |
| INT4   | 4   | 4   | 8   | 64 | 256 | **≈ 3.5** |

Per-layer k search uses MSE minimisation to find the largest k that still
minimises `E[(quantise(y_uint, δ(k)) − y_uint)²]` on calibration data.

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
