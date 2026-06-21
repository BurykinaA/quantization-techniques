# ADC-Aware INT4 Quantization of Llama

Clean reference implementation of an ADC-aware INT4 quantization pipeline for
Llama-3.2-1B on analog hardware with Analog-to-Digital Converters (ADCs).

The four configs tell the full story: how much perplexity (PPL) each stage of the
pipeline costs, and how post-ADC LoRA correction recovers it. The ADC model is the
symmetric (bipolar) one throughout.

---

## Configs

| Config        | Description                                                  |
|---------------|--------------------------------------------------------------|
| `fp`          | Full precision (FP16) baseline                               |
| `int4_no_adc` | INT4 FlatQuant transforms, evaluated WITHOUT the ADC floor   |
| `best_ptq`    | INT4 FlatQuant + symmetric ADC hardware model (best pure PTQ)|
| `best_lora`   | `best_ptq` + post-ADC LoRA residual correction               |

`best_ptq` and `best_lora` share identical FlatQuant training; they differ only in
whether the LoRA correction is applied.

---

## How to Run

```bash
# Run all four configs (each INT4 config trains FlatQuant from scratch — slow)
bash run.sh

# Run a subset
bash run.sh --configs best_ptq best_lora

# Custom output directory
bash run.sh --output_dir /data/results
```

Results are printed as a table and saved to `outputs/results.json`. The
FlatQuant+ADC model is cached under `outputs/fq_cache/`, so re-runs skip training.

## Tests

```bash
python -m pytest tests        # from ADC/best_clean/
```

The tests are CPU-only and do not download the model.

---

## Hardware Model

The target hardware uses a matrix-vector multiply (MVM) unit that performs the dot
product in integer arithmetic and reads the result through a symmetric ADC.

### Tiled Integer MVM

A weight matrix **W** of shape `out x in` is split into tiles of width `mvm_limit`
(the number of columns the MVM unit processes at once). For each activation vector
**x**:

1. **Quantize activations** (per token, symmetric):
   `x_int = clamp(round(x / s_x), -q_x, q_x)`, `q_x = 2^(bx-1) - 1`
2. **Quantize weights** (per output channel, symmetric):
   `W_int = clamp(round(W / s_w), -q_w, q_w)`, `q_w = 2^(bw-1) - 1`
3. **Integer dot product per tile:** `z_int = x_int_tile . W_int_tile^T`
4. **Symmetric ADC floor quantization:**
   ```
   z = clamp(floor(z_int / delta), na, pa) * delta
   na = -2^(ba-1),  pa = 2^(ba-1) - 1
   ```

### Delta (ADC resolution)

```
delta = 2 * tile_in * q_x * q_w / (2^ba * k)
```

| Config | bx | bw | ba | k  | tile_in | delta    |
|--------|----|----|----|----|---------|----------|
| INT8   | 8  | 8  | 8  | 16 | 256     | ~= 2016  |
| INT4   | 4  | 4  | 8  | 16 | 256     | ~= 6.125 |

`k` is the ADC parallelism (columns per converter); larger `k` gives a finer step.

---

## FlatQuant Transforms

Per-token L-infinity scaling concentrates the quantization range on the largest
feature in each token. **FlatQuant** learns invertible linear transforms that
redistribute activation energy uniformly across channels before quantization.

Each transform is a Kronecker product `T = kron(L, R)` with approximately
orthogonal `L`, `R` (Cayley parameterization), plus an optional learnable
per-channel diagonal scale `d in [1e-4, 10]`. The Kronecker structure cuts the
parameter count from O(d^2) to O(d).

### Calibration Loss

For each transformer block the transforms minimize the reconstruction error
between the FP output and the quantized (ADC) output:

```
L_calib = MSE( block_fp(x), block_quant(T . x) )
```

### Staged Training

**Stage A (30 epochs):** train Kronecker transforms for MLP blocks, no propagation
(each layer sees clean FP16 inputs). Establishes good transform quality.

**Stage B (10 epochs, lr x 0.1):** add diagonal scaling for attention blocks and
enable partial propagation with `alpha = 0.5`:

```
x_cal = alpha * x_adc + (1 - alpha) * x_fp
```

This trains each layer on a mix of FP and ADC-quantized inputs from the previous
layer. `alpha = 0.5` is the sweet spot (`alpha = 1.0` over-specializes; `alpha = 0`
ignores the ADC).

After calibration, `reparameterize_model()` bakes the transforms into the weight
matrices so inference has no transform overhead.

---

## Post-ADC LoRA Correction

INT4 weight quantization introduces a systematic residual error that transform
calibration cannot recover. A small post-ADC LoRA residual corrects it.

### Architecture

```
y = TiledLinearADC(x) + (alpha / r) * B(A(x.float()))
```

- `A` (r x in) and `B` (out x r) are FP32 parameters
- `r = 4`, `alpha = 8.0` -> scaling = 2.0
- `B` is zero-initialized -> initial output equals the frozen ADC model
- The correction is added **after** the ADC floor, so gradients never flow through
  the non-differentiable `floor()` — training is numerically stable

### Training Loss

```
L = L_CE + lambda * KL(student || teacher_fp),   lambda = 0.5,  T = 2.0
```

`teacher_fp` is a frozen copy of the original FP16 model. The KL term pulls the
student distribution toward the FP baseline (smoother gradients than CE alone).

### Targets

All 7 projections: MLP (`down_proj`, `up_proj`, `gate_proj`) and attention
(`q_proj`, `k_proj`, `v_proj`, `o_proj`). Training: 5 epochs, lr = 1e-4, AdamW.

---

## Code Structure

```
ADC/best_clean/
├── README.md          ← this file
├── run.sh             ← entry point: bash run.sh
├── configs.py         ← 4 config presets as dataclasses
├── pipeline.py        ← end-to-end pipeline + orchestration
├── eval.py            ← perplexity (sliding window) + latency
├── core/
│   ├── adc_layers.py     ← TiledLinearADC: tiling, INT quant, symmetric ADC floor
│   ├── adc_lora.py       ← ResidualLoRATiledLinearADC + calibrate_adc_lora
│   ├── flat_quant.py     ← FlatQuant: Kronecker transforms, calibration loop
│   ├── grad_functions.py ← round_ste, floor_ste, learnable-quantizer autograd
│   └── utils.py          ← kurtosis (W-reshape) loss
└── tests/             ← CPU unit tests (pytest)
```

### Key Classes

**`TiledLinearADC`** (`core/adc_layers.py`) — the hardware model: tiled INT4/INT8
MVM + symmetric ADC floor quantization. `set_bypass_adc(True)` disables the floor
for ablation.

**`ResidualLoRATiledLinearADC`** (`core/adc_lora.py`) — wraps `TiledLinearADC` with
an FP32 post-ADC LoRA residual.

**`FlatQuantLinear`** (`core/flat_quant.py`) — applies learnable Kronecker
transforms during calibration; `reparameterize_model()` then bakes them into the
weights so inference has no overhead.

---

## References

- FlatQuant: [arxiv.org/abs/2410.09426](https://arxiv.org/abs/2410.09426)
- QLoRA: [arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314)
- GPTQ perplexity methodology: [arxiv.org/abs/2210.17323](https://arxiv.org/abs/2210.17323)
