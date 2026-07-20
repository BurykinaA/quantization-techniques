# ADC-aware INT4 quantization for Vision Transformers

Ports the FlatQuant + unipolar-ADC method (thesis §5.5.4, originally Llama-3.2-1B)
to timm Vision Transformers, evaluated on ImageNet top-1/top-5.

The architecture-agnostic machinery in `../core/` is reused **unchanged**:
`adc_layers.py` (`QATLinearADC`/`TiledLinearADC`), `adc_lora.py`
(`ResidualLoRATiledLinearADC`/`apply_adc_lora`), and `flat_quant.py`
(`FlatQuantLinear`, `KroneckerTransform`). Only the ViT-specific glue lives here.

## Files

| File | Role |
|---|---|
| `vit_pipeline.py` | Orchestrator + CLI (load → FlatQuant → ADC → LoRA → eval) |
| `vit_flat_quant.py` | `FlatQuantViTAttention`/`FlatQuantViTMlp`, `apply_flatquant_to_vit`, `reparameterize_vit`, `calibrate_flat_quant_vit` |
| `vit_configs.py` | The four config presets + `--smoke` scaling |
| `vit_data.py` | ImageNet loaders (timm-derived transforms) |
| `vit_eval.py` | top-1/top-5 validation + latency |
| `vit_lora.py` | ViT post-ADC LoRA training (image CE + KL to FP teacher) |
| `run.sh` | env wrapper (see below) |

## Configurations (thesis §5.5.4, ported)

1. **`fp`** — full-precision timm model, no quantization.
2. **`int4_no_adc`** — INT4 FlatQuant transforms, ADC floor bypassed. Isolates INT4 error.
3. **`unsigned_ptq`** — INT4 FlatQuant + unipolar (unsigned shift-subtract) ADC. `k` configurable.
4. **`unsigned_ptq_lora`** — (3) + post-ADC residual LoRA (rank **r=4**, α=8, all projections,
   CE + λ·KL to the FP teacher, λ=0.5, T=2.0).

Quantized layers per block: `attn.qkv`, `attn.proj`, `mlp.fc1`, `mlp.fc2`
(4 × N_blocks). The patch-embed Conv2d and the classifier `head` are **not** quantized.

`k` is a CLI flag (**default 4** per request; the thesis used a global **k=16**). Larger `k`
→ smaller ADC step δ → finer resolution.

## Environment (important)

The project `PYTHONPATH` shadows torch and breaks `import torch`. Always launch through
`run.sh`, which does `cd /tmp && env -u PYTHONPATH IMAGENET_ROOT=... python vit_pipeline.py`.
Verified stack: torch 2.2.2+cu121, timm 0.9.2, torchvision 0.16 on an A100 80GB.
Needs only torch + timm + torchvision (no `transformers`/`datasets`).

`IMAGENET_ROOT` defaults to `/home/coder/project/imagenet/data` (layout `{train,val}/<class>/*.JPEG`).

## Running

```bash
cd /home/coder/project/quantization_techniques/ADC/best/vit

# 1) Smoke check — fast correctness (2/1 FlatQuant epochs, 128 calib imgs, 5% val subset):
./run.sh --model vit_tiny_patch16_224 --configs fp int4_no_adc --smoke --val_portion 0.05

# 2) Full unsigned PTQ (k=4) on vit_tiny:
./run.sh --model vit_tiny_patch16_224 --configs unsigned_ptq --k 4

# 3) Add post-ADC LoRA (rank 4):
./run.sh --model vit_tiny_patch16_224 --configs unsigned_ptq_lora --k 4

# 4) All four on vit_base:
./run.sh --model vit_base_patch16_224 \
    --configs fp int4_no_adc unsigned_ptq unsigned_ptq_lora --k 4
```

Useful flags: `--k {4,8,16}`, `--val_portion 0.05` (strided subset for speed),
`--smoke`, `--no_fq_cache`. FlatQuant results are cached under
`outputs_vit/fq_cache/` keyed by (model, bits, k, epochs, …), so re-running a config
that shares FlatQuant training skips the expensive calibration.

Results are printed as a table and written to `outputs_vit/results_vit.json`.

## Expected sanity behavior

There is **no thesis ViT target number** — this is a port to a new modality. The correctness
signal is:

- `fp` top-1 ≈ **76%** (vit_tiny) / **~85%** (vit_base) — matches timm pretrained.
- Monotonic degradation: `fp ≥ int4_no_adc ≥ unsigned_ptq` in top-1.
- LoRA recovers accuracy: `unsigned_ptq_lora > unsigned_ptq`.
- `--k 16` vs `--k 4` changes the logged δ and moves accuracy (proves the ADC path is engaged).
- LoRA CE+KL loss decreases across epochs (logged).
```
