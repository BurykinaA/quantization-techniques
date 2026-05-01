# ADC-Aware INT4 Quantization of Llama — Best Configuration

Clean reference implementation of the best-performing quantization pipeline for
Llama-3.2-1B on analog hardware with Analog-to-Digital Converters (ADCs).

Each configuration tells one part of the story: how much PPL each pipeline
component costs, and how per-layer k search and post-ADC LoRA correction recover it.

---

## Results (Llama-3.2-1B)

### Current setup — unsigned shift-subtract, unipolar FQ (δ = 14.0625, k=16 global or per-layer)

FlatQuant trained with unipolar delta formula: `δ = tile_in · (2^bx−1) · (2^bw−1) / (2^ba · k)`.

| Config | Description | Wiki PPL | C4 PPL |
|---|---|---|---|
| `fp` | Full precision (FP16) | **8.68** | **13.13** |
| `int4_no_adc` | INT4 FlatQuant, no ADC floor | 12.56 | 20.13 |
| `best_ptq` | INT4 FlatQuant + unsigned ADC, k=16 (global) | 39.00 | 67.78 |
| `best_ptq_k` | best_ptq + per-layer k search (no FQ recal) | 21.64 | 33.79 |
| `best_ptq_k_recal` | best_ptq_k + FlatQuant retrained with per-layer k | 22.19 | 36.94 |
| `best_lora` | best_ptq + post-ADC LoRA correction | 17.29 | 29.26 |
| `best_lora_k` | best_lora + per-layer k search | **14.56** | **23.78** |

`int4_no_adc` result from bipolar FQ training (ADC not involved — FlatQuant quality only).

---

## Algorithm: Unsigned Shift-Subtract ADC

The physical device accepts only non-negative inputs. The algorithm replaces one
sign-extended bipolar MVM with one non-negative MVM followed by exact digital correction.

All steps below assume INT4 weights and activations (`bx = bw = 4`),
tile width `tile_in = 256`, ADC bits `ba = 8`, parallelism `k = 16`.

**Step 1 — Shift to unsigned:**
```
code_x_u = code_x + 8   →  [0, 15]
code_w_u = code_w + 8   →  [0, 15]
```
`code_x ∈ [−8, 7]` and `code_w ∈ [−8, 7]` are the INT4 activation and weight codes.
Adding the zero-point `zp = 2^(b−1) = 8` maps both to `[0, 2^b − 1] = [0, 15]`.

**Step 2 — MVM on hardware (one ADC read):**
```
y_uint = code_x_u · code_w_u^T   ∈ [0, 57600]
```
`y_uint ≥ 0` always, so a single unsigned MVM suffices — no 4-quadrant split needed.
Maximum per tile: `tile_in · (2^bx − 1) · (2^bw − 1) = 256 · 15 · 15 = 57600`.

**Step 3 — ADC:**
```
adc_out = floor(y_uint / δ) · δ
δ = tile_in · (2^bx − 1) · (2^bw − 1) / (2^ba · k)
  = 256 · 15 · 15 / (256 · 16)
  = 57600 / 4096
  = 14.0625  for k=16
```
`adc_out` quantizes the hardware-summed result to multiples of δ.
Increasing k reduces δ (finer ADC resolution) at the cost of more parallel ADC reads.

**Step 4 — Digital correction (exact, no approximation):**
```
y_int = adc_out
      − 8 · Σ_j code_w_u_j        ← per output channel  (precompute at load time)
      − 8 · Σ_i code_x_u_i        ← per token           (compute once per token)
      + 256 · 8 · 8                ← scalar constant     (= tile_in · zp_x · zp_w)
```
This recovers the true signed inner product from the shifted unsigned one.
The per-channel term `Σ_j code_w_u_j` is a weight-only sum — precomputed once.
The per-token term `Σ_i code_x_u_i` is a single sum over the tile — O(tile_in) per token.

**Step 5 — Dequantization:**
```
y_real = y_int · s_x · s_w
```
`s_x` — per-token activation scale, `s_w` — per-channel weight scale.

### Derivation

The correction follows from expanding the unsigned product:
```
code_x_u · code_w_u = (code_x + zp_x) · (code_w + zp_w)
                    = code_x · code_w
                    + zp_x · code_w
                    + zp_w · code_x
                    + zp_x · zp_w

⟹  code_x · code_w = code_x_u · code_w_u
                    − zp_x · Σ code_w_u    (because Σ code_w = Σ code_w_u − tile_in · zp_w)
                    − zp_w · Σ code_x_u
                    + tile_in · zp_x · zp_w
```

### Why not 4-quadrant?

The previous branch (`clean-optical-settup`) split codes into positive/negative parts
and ran four separate non-negative MVMs (x⁺·w⁺, x⁻·w⁻, x⁺·w⁻, x⁻·w⁺).
That gives a finer δ_branch ≈ 3.06 but costs 4× the ADC reads.

Unsigned shift-subtract uses **one** MVM, with δ ≈ 14.06. Per-layer k search
compensates: layers with narrow output distributions get a larger k (finer δ).

### ADC resolution (δ) vs k

| k | δ | ADC reads per tile |
|---|---|---|
| 4  | 56.25 | 4  |
| 8  | 28.13 | 8  |
| 16 | **14.0625** | 16 |
| 32 | 7.03  | 32 |
| 64 | 3.52  | 64 |

Per-layer k search finds the optimal k for each projection layer independently,
trading ADC resolution against hardware cost per layer.

---

## Experiment History

All experiments on Llama-3.2-1B, WikiText2 / C4 perplexity (lower is better).

### 1. Bipolar ADC (baseline)

FlatQuant trained with bipolar delta: `δ = 2 · tile_in · q_x · q_w / (2^ba · k)`.  
Range: `y_int ∈ [−M, +M]`, `M = 256 · 7 · 7 = 12544`, `δ ≈ 6.12` for k=16.

| Config | Wiki PPL | C4 PPL |
|---|---|---|
| `best_ptq` | 26.78 | 45.40 |
| `best_lora` | **14.01** | **23.14** |

### 2. 4-quadrant unipolar (branch `clean-optical-settup`)

Signed codes split into positive/negative parts, four separate non-negative MVMs.  
`δ_branch ≈ 3.06` but 4× hardware cost. FlatQuant trained with bipolar delta.

| Config | Wiki PPL | C4 PPL |
|---|---|---|
| `best_ptq` | 18.08 | 28.41 |

### 3. Unsigned shift-subtract — bipolar FQ cache (wrong setup, historical)

Unsigned ADC evaluated with FlatQuant checkpoints trained for bipolar delta (δ_fq ≈ 6.12).
The FlatQuant transforms were optimized for the wrong δ — mismatch blows up `best_ptq`,
but per-layer k search partially recovers by finding small k that keeps δ consistent.

| Config | Wiki PPL | C4 PPL | Notes |
|---|---|---|---|
| `best_ptq` | 94274.75 | 89773.24 | δ mismatch: FQ trained for δ≈6, eval uses δ≈14 |
| `best_ptq_k` | 16.78 | 27.59 | k search works around mismatch |
| `best_lora` | 19.27 | 34.40 | |
| `best_lora_k` | **13.42** | **21.94** | best overall, but incorrect setup |

### 4. Unsigned shift-subtract — unipolar FQ (δ = 14.0625) — current setup

FlatQuant retrained with the correct unipolar delta formula.
FQ transforms optimised for δ=14.06 throughout calibration.

| Config | Wiki PPL | C4 PPL | Notes |
|---|---|---|---|
| `best_ptq` | 39.00 | 67.78 | coarse δ=14.06, transforms not fully adapted |
| `best_ptq_k` | 21.64 | 33.79 | per-layer k, FQ unchanged |
| `best_ptq_k_recal` | 22.19 | 36.94 | FQ retrained with per-layer k (2× FQ cost) |
| `best_lora` | 17.29 | 29.26 | LoRA on top of global k=16 |
| `best_lora_k` | **14.56** | **23.78** | LoRA on top of per-layer k — best correct result |

`best_ptq_k_recal` (FQ retrained with per-layer k) is worse than `best_ptq_k` (in-place k),
suggesting that re-training FQ on the found k over-specializes transforms and hurts generalization.

### Iterative k-Search During LoRA Training (branch `best-iterative-k-lora`)

**Hypothesis:** Start with k=4 (coarse ADC, δ≈56), let LoRA adapt to that noise level,
then re-search k per-layer to tighten resolution where distributions allow.

k-search uses the **unipolar range criterion**: R = 99.9th percentile of y\_uint,
picks largest k from {4, 8, 16, 32, 64} such that the ADC covers R without saturation.

| Config | k\_init | k\_search | epochs | Wiki PPL | C4 PPL |
|---|---|---|---|---|---|
| `best_lora_k` *(baseline)* | 16 | before training | 5 | **14.56** | **23.78** |
| `iter_lora_k_pre` | 4 | before training | 5 | **14.56** | **23.78** |
| `iter_lora_k_i2`  | 4 | every 2 epochs | 6 | 14.90 | 24.22 |
| `iter_lora_k_i1`  | 4 | every 1 epoch  | 5 | 14.96 | 24.18 |

**Conclusion:** Mid-training k updates hurt (14.90–14.96 vs 14.56). LoRA adapts to the initial ADC noise level; changing k mid-training disrupts that adaptation. Pre-training k-search = per-layer baseline exactly.

### Per-tile k Search (branch `best-iterative-k-lora`)

**Hypothesis:** Individual tiles within the same layer (e.g. `q_proj.tiles.0` vs `.tiles.1`) have
different y\_uint distributions — per-tile k may outperform per-layer k aggregated across all tiles.

| Config | k granularity | LoRA | Wiki PPL | C4 PPL |
|---|---|---|---|---|
| `best_ptq_k` *(baseline)* | per-layer | no | 21.64 | 33.79 |
| `best_ptq_k_tile` | per-tile | no | — | — |
| `best_lora_k` *(baseline)* | per-layer | yes | 14.56 | 23.78 |
| `best_lora_k_tile` | per-tile | yes | — | — |

### Outlier-aware Tiling (branch `best-iterative-k-lora`)

**Hypothesis:** Consecutive tiling puts outlier and normal channels in the same tile, inflating
the per-token scale s\_x so normal channels lose INT4 resolution. Sorting by mean |activation|
puts all outlier channels in tile 0, letting tile 1 use a tight scale.

| Config | tiling | LoRA | Wiki PPL | C4 PPL |
|---|---|---|---|---|
| `best_ptq_k` *(baseline)* | consecutive | no | 21.64 | 33.79 |
| `outlier_tile_ptq` | outlier-aware | no | — | — |
| `best_lora_k` *(baseline)* | consecutive | yes | 14.56 | 23.78 |
| `outlier_tile_lora` | outlier-aware | yes | — | — |

---

## Per-Layer k Search

After FlatQuant calibration with a global k, the search finds the best k per layer by
capturing pre-ADC values `y_uint` on the calibration set and selecting k to minimize
reconstruction MSE (or, for unipolar, to ensure the ADC range covers the actual signal).

**Unipolar criterion:**
```
R = quantile(|y_uint|, 0.999)          # observed output range
k_min = M_uint / (2 · R)               # smallest k where ADC covers the signal
k* = smallest candidate ≥ k_min
```
where `M_uint = tile_in · (2^bx−1) · (2^bw−1)` is the theoretical maximum.

Layers with narrow output distributions (small R) get a small k (coarser δ, less
ADC reads). Layers with wide distributions get a large k (finer δ, more ADC reads).

Candidates: `k ∈ {4, 8, 16, 32, 64}`.

---

## How to Run

```bash
# Default: best_ptq + best_ptq_k + best_lora + best_lora_k
bash run.sh

# PTQ only, no LoRA
bash run.sh --configs best_ptq best_ptq_k

# With FlatQuant recalibration
bash run.sh --configs best_ptq best_ptq_k_recal

# Custom output directory
bash run.sh --output_dir /data/results
```

Results are printed as a table and saved to `outputs/results.json`.

---

## Hardware Model

The target hardware performs a tiled integer MVM and reads the result through an ADC.

### Tiled Integer MVM

Weight matrix **W** ∈ ℝ^{out × in} is split into tiles of width `mvm_limit = 256`:

```
tile width  = mvm_limit = 256 columns
tile height = out_features
```

For each activation vector **x** ∈ ℝ^{in}:

1. **Quantize activations:**
   `x_int = clamp(round(x / s_x), −q_x, q_x)`,  `q_x = 2^{bx−1} − 1 = 7`

2. **Quantize weights:**
   `W_int = clamp(round(W / s_w), −q_w, q_w)`,  `q_w = 2^{bw−1} − 1 = 7`

3. **Unsigned shift per tile (Step 1 above):**
   `code_x_u = code_x + 8`,  `code_w_u = code_w + 8`

4. **Unsigned MVM on hardware (Step 2 above):**
   `y_uint = code_x_u · code_w_u^T   ∈ [0, 57600]`

5. **ADC floor (Step 3 above):**
   `adc_out = floor(y_uint / δ) · δ`,  `δ = 14.0625` for k=16

6. **Digital correction (Step 4 above):**
   `y_int = adc_out − 8·Σ_j code_w_u_j − 8·Σ_i code_x_u_i + 256·64`

7. **Dequantize (Step 5 above):**
   `y_real = y_int · s_x · s_w`

---

## FlatQuant Transforms

### Motivation

Per-token L∞ scaling concentrates all quantization range on the largest feature in
each token. Other features are quantized with few levels or fall into the dead zone.

**FlatQuant** learns invertible linear transforms that redistribute activation
energy uniformly across channels before quantization.

### Kronecker Decomposition

Each transform is factored as a Kronecker product:

```
T = kron(L, R) ≈ L ⊗ R
```

where L ∈ ℝ^{√d × √d} and R ∈ ℝ^{√d × √d} are approximately orthogonal
matrices parameterized via a Cayley transform. **Diagonal scaling** adds a
learnable per-channel scale d ∈ [10⁻⁴, 10]: `T_diag = diag(d) · kron(L, R)`.

### Calibration Loss

For each transformer block:
```
L_calib = MSE( block_fp(x), block_quant(T·x) )
```
where `block_quant` includes the integer dot product and ADC floor quantization
(using the correct unipolar δ = 14.0625 for this branch).

### Staged Training (Two-Phase)

**Stage A (30 epochs):**
- Train Kronecker transforms for MLP blocks only
- No propagation — each layer sees clean FP16 inputs from upstream

**Stage B (10 epochs, lr × 0.1):**
- Adds diagonal scaling for attention blocks
- Partial propagation with α = 0.5:

```
x_cal = α · x_adc + (1 − α) · x_fp
```

α = 0.5 is the sweet spot — α = 1.0 over-specializes transforms; α = 0 ignores ADC.

---

## Post-ADC LoRA Correction

### Architecture

For each target linear layer:
```
y = TiledLinearADC(x) + (α / r) · B(A(x.float()))
```
- **A** ∈ ℝ^{r × in}, **B** ∈ ℝ^{out × r}, FP32
- **r = 4**, **α = 8.0** → scaling = 2.0
- **B** initialized to zeros → initial output equals the frozen ADC model
- Correction added **after** the ADC floor — gradients never flow through floor()

### Training Loss

```
L = L_CE + λ · KL(student ‖ teacher_fp)
```
- **λ = 0.5**, temperature **T = 2.0**
- Teacher is a frozen FP16 copy of the original model
- KL term provides smoother gradients than CE alone (+1.3 PPL if removed)

### Targets

All 7 projections: `down_proj`, `up_proj`, `gate_proj`, `q_proj`, `k_proj`, `v_proj`, `o_proj`.  
Training: 5 epochs, lr = 1e-4, AdamW.

---

## Code Structure

```
ADC/best/
├── README.md          ← this file
├── run.sh             ← entry point: bash run.sh
├── configs.py         ← config presets as dataclasses
├── pipeline.py        ← pipeline: FlatQuant → k search → LoRA → eval
├── eval.py            ← PPL (sliding window) + latency
└── core/
    ├── adc_layers.py  ← TiledLinearADC / QATLinearADC: tiling, quant, ADC floor
    ├── adc_lora.py    ← ResidualLoRATiledLinearADC + calibrate_adc_lora
    ├── flat_quant.py  ← FlatQuant: Kronecker transforms, calibration loop
    └── grad_functions.py  ← round_ste, floor_ste (straight-through estimators)
```

### Key Classes

**`TiledLinearADC`** (`core/adc_layers.py`)  
Hardware model: tiled INT4/INT8 MVM + unsigned shift-subtract ADC floor.
`set_k(k)` updates ADC parallelism in-place and recomputes δ.
`set_bypass_adc(True)` disables the ADC floor for ablation.

**`QATLinearADC`** (`core/adc_layers.py`)  
Single tile. Implements Steps 1–4 from the algorithm above.
`_capturing = True` enables y_uint capture for per-layer k search.

**`ResidualLoRATiledLinearADC`** (`core/adc_lora.py`)  
Wraps TiledLinearADC with a FP32 LoRA residual. Post-ADC placement ensures
stable training. Pre-ADC placement (inside quantization) was tried and
catastrophically diverged (bypass PPL > 100).

**`FlatQuantLinear`** (`core/flat_quant.py`)  
Applies learnable Kronecker transforms during calibration, then
`reparameterize_model()` bakes the transforms into weight matrices so
inference has no overhead. Uses unipolar δ formula when `fq_unipolar_delta=True`.

---

## References

- FlatQuant: [arxiv.org/abs/2410.09426](https://arxiv.org/abs/2410.09426)
- QLoRA: [arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314)
- GPTQ perplexity evaluation methodology: [arxiv.org/abs/2210.17323](https://arxiv.org/abs/2210.17323)
