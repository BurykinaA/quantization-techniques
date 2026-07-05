# Experiments needed to close all visualization placeholders

Этот файл — детальный план: какой эксперимент, что именно запускать,
какой скрипт менять, какие данные уже есть, а каких не хватает.

---

## Фигуры, которые можно сделать прямо сейчас (данные в README)

### `plot_int8_main_results.pdf` — слайд 05_int8

**Данные уже есть:**
| method         | bypass | ADC PPL |
|----------------|--------|---------|
| baseline       | 10.01  | 28.86   |
| center         | 10.01  | 28.86   |
| prop           | 11.01  | 14.55   |
| prop+center    | 11.00  | 14.41   |

**Что рисовать:** grouped bar chart. Две группы баров на каждый метод:
bypass (светло-синий) и ADC (оранжевый). Добавить dashed линии: FP=10.5, INT8-prop+center=14.41.

**Скрипт:** написать `slides/scripts/plot_int8.py` с данными хардкодом (4 точки).

---

### `plot_int4_ptq_progression.pdf` — слайд 06

**Данные уже есть (выбранные ключевые точки):**
| step | label                 | bypass | ADC PPL |
|------|-----------------------|--------|---------|
| 1    | baseline (128s)       | 15.41  | 2354.86 |
| 2    | prop (128s)           | 40.28  | 203.89  |
| 3    | prop (512s)           | 34.25  | 40.63   |
| 4    | α=0.5 (1024s)         | 24.24  | 31.22   |
| 5    | add_diag + α=0.5      | 20.23  | 27.56   |
| 6    | staged MLP→attn       | 18.50  | 27.60   |

**Что рисовать:** step plot с log y-axis. Две линии: bypass и ADC PPL.
Аннотировать каждый шаг с key change. FP=10.5, INT8 prop+center=14.41 как dashed.

---

### `plot_diag_roles.pdf` — слайд 07

**Данные уже есть (v3/v4):**
| config          | bypass | ADC PPL |
|-----------------|--------|---------|
| diag MLP only   | 19.37  | 31.65   |
| diag attn only  | 24.98  | 29.41   |
| diag both       | 19.69  | 28.46   |
| staged          | 18.50  | 27.60   |

---

### `plot_mlp_split_up_down.pdf` — слайд 07

**Данные уже есть (v4):**
| config          | bypass | ADC PPL |
|-----------------|--------|---------|
| diag up_gate    | 23.22  | 39.59   |
| diag down_trans | 20.42  | 31.26   |
| diag both MLP   | 19.37  | 31.65   |

---

### `plot_lora_ablations_grid.pdf` — слайд 11

**2×2 сетка (все данные из v7):**

Top-left — CE vs CE+KL:
| config              | ADC PPL |
|---------------------|---------|
| down+o, CE, r4      | 15.57   |
| down+o, CE+KL, r4   | 14.33   |
| all7, CE, r4        | 16.68   |
| all7, CE+KL, r4     | 14.03   |

Top-right — rank sweep (down+o, CE+KL):
| rank | ADC PPL |
|------|---------|
| 1    | 15.90   |
| 2    | 15.60   |
| 4    | 15.57   |
| 8    | 15.24   |

Bottom-left — layer-selective (all: 15.63, first8: 16.64, last8: 20.08)

Bottom-right — post vs pre ADC (15.63 vs 105.63 — log scale!)

---

### `plot_final_results_wiki_c4.pdf` — слайд 12

**Данные уже есть (v8):**
| method             | Wiki ADC | C4 ADC |
|--------------------|----------|--------|
| INT8 baseline      | 28.86    | —      |
| INT8 prop+center   | 14.41    | —      |
| INT4 staged PTQ    | 27.60    | 46.40  |
| INT4 r4_all_ce_kl  | 13.96    | 23.30  |

---

### `plot_ptq_plateau.pdf` — слайд 09

**Данные:** взять лучшие ADC PPL из каждой версии v1–v5:
| version | best method | ADC PPL |
|---------|-------------|---------|
| v1 128s | hadamard+prop | 178.86 |
| v1 512s | prop_512s    | 40.63   |
| v2 α-sweep | add_diag+α0.5 | 27.56 |
| v3 diag | staged-like  | 28.46   |
| v4 staged | staged_mlp_attn | 27.60 |
| v5 stoch | staged+beta2 | 27.82  |

Показать "PTQ plateau" как серую горизонтальную полосу 27–28.5.

---

## Фигуры, которые требуют схем (TikZ / Inkscape)

### `fig_digital_vs_analog_path.pdf` — слайд 01

Две колонки:
- Левая: digital matmul → output
- Правая: analog weights (crossbar), analog MVM → ADC → output

Можно сделать в TikZ, уже подключён.

### `fig_tile_adc_pipeline.pdf` — слайд 02

Уже встроен в TikZ прямо в слайде (02_problem_setup.tex).
Если хочешь вынести как отдельный PDF — перенести tikzpicture в standalone.

### `fig_flatquant_intuition.pdf` — слайд 03

Два histogram plots:
- До трансформа: один высокий spike (feature 0), остальные маленькие
- После трансформа: равномерное распределение

**Скрипт:** `slides/scripts/plot_flatquant_intuition.py`
Можно сделать с синтетическими данными (не нужен настоящий Llama):
```python
import numpy as np, matplotlib.pyplot as plt

np.random.seed(0)
# before: one outlier
before = np.abs(np.random.randn(256)) * 0.1
before[0] = 12.7  # outlier

# after: flat (orthogonal rotation squeezes outlier)
after = np.random.uniform(4, 9, 256)

# codes
codes_before = (before / before.max() * 127).astype(int)
codes_after  = (after  / after.max()  * 127).astype(int)

# y_int = dot(codes_x, codes_w); plot histogram of codes
```

### `fig_adc_lora_post_vs_pre.pdf` — слайд 10

TikZ схема:
```
x ─────┬──────────────────────────────────────────→ Qx → MVM → ADC floor → y_base ─┐
       │                                                                              + → y
       └──────────────────────────────→ lora_A → lora_B → scaling ────────────────┘

[Red box]: PRE-ADC PATH: x → Qx → MVM + ΔW → ADC floor → (FAILED)
```

### `fig_contributions_summary.pdf` — слайд 13

Flow diagram с 5 шагами в TikZ (arrows между boxes).

---

## Эксперимент, которого ещё НЕТ: Loss Landscape (слайд 08)

### Что нужно запустить

Это ключевой эксперимент, которого нет в готовых данных.

**Цель:** визуализировать, как меняется форма loss landscape при обучении FlatQuant
с разными значениями α (0, 0.5, 1.0).

Метод: Li et al. ["Visualizing the Loss Landscape of Neural Nets"](https://arxiv.org/abs/1712.09913),
filter-normalised random directions.

---

### Шаг 1: Обучить 3 чекпоинта

Использовать **один слой** или **небольшой subset** (например, только layer 0–3),
чтобы эксперимент был быстрым.

```bash
# α=0: pure FP loss (no ADC in loss)
python ADC/llama/runs/llama_smooth_quant_adc_ptq.py \
  --bw 4 --bx 4 --ba 8 \
  --fq_prop_alpha 0.0 \
  --n_calibration_samples 256 \
  --fq_epochs 30 \
  --save_checkpoint checkpoints/landscape_alpha0.pt

# α=0.5: mixed
python ADC/llama/runs/llama_smooth_quant_adc_ptq.py \
  --bw 4 --bx 4 --ba 8 \
  --fq_prop_alpha 0.5 \
  --n_calibration_samples 256 \
  --fq_epochs 30 \
  --save_checkpoint checkpoints/landscape_alpha05.pt

# α=1.0: full ADC loss
python ADC/llama/runs/llama_smooth_quant_adc_ptq.py \
  --bw 4 --bx 4 --ba 8 \
  --fq_prop_alpha 1.0 \
  --n_calibration_samples 256 \
  --fq_epochs 30 \
  --save_checkpoint checkpoints/landscape_alpha1.pt
```

---

### Шаг 2: Написать скрипт визуализации

Создать файл `slides/scripts/plot_loss_landscape.py`:

```python
"""
Loss landscape visualisation for FlatQuant α-mixing.
Method: Li et al. 2018, filter-normalised random directions.

Usage:
  python plot_loss_landscape.py \
      --checkpoint_alpha0   checkpoints/landscape_alpha0.pt \
      --checkpoint_alpha05  checkpoints/landscape_alpha05.pt \
      --checkpoint_alpha1   checkpoints/landscape_alpha1.pt \
      --output              figs/plot_loss_landscape_alpha.pdf
"""

import argparse, torch, numpy as np, matplotlib.pyplot as plt
from copy import deepcopy

# ── Filter-normalised direction ──────────────────────────────────────────────
def filter_normalize(v: dict) -> dict:
    """Normalize each weight tensor of v by the norm of the corresponding
    filter (row for Linear, or full tensor for bias/scalar).
    Returns dict with same structure but normalised values."""
    out = {}
    for k, w in v.items():
        if w.dim() >= 2:
            # per-output-channel normalisation
            norms = w.view(w.shape[0], -1).norm(dim=1, keepdim=True)
            norms = norms.view(w.shape[0], *([1]*(w.dim()-1))).clamp(min=1e-8)
            out[k] = w / norms
        else:
            out[k] = w / (w.norm().clamp(min=1e-8))
    return out


def random_direction(params: dict) -> dict:
    """Random filter-normalised direction in parameter space."""
    d = {k: torch.randn_like(v) for k, v in params.items()}
    return filter_normalize(d)


def get_flatquant_params(checkpoint) -> dict:
    """Extract only the learnable FlatQuant transform parameters."""
    # Adjust key prefix to match your checkpoint structure
    return {k: v.float() for k, v in checkpoint.items()
            if 'flatquant' in k or 'transform' in k or 'diag' in k}


def perturb_params(base: dict, d1: dict, d2: dict, a: float, b: float) -> dict:
    return {k: base[k] + a * d1[k] + b * d2[k] for k in base}


def eval_adc_ppl(params: dict, model, calibration_data, device) -> float:
    """
    Load params into model, run evaluation with ADC enabled, return PPL.
    Adapt this function to your model loading / eval pipeline.
    """
    # TODO: load params into model, run eval
    raise NotImplementedError("Implement with your eval pipeline")


def compute_landscape(base_params, d1, d2, model, cal_data, device,
                      grid_range=(-1.0, 1.0), n=21):
    xs = np.linspace(*grid_range, n)
    ys = np.linspace(*grid_range, n)
    Z = np.zeros((n, n))
    for i, a in enumerate(xs):
        for j, b in enumerate(ys):
            p = perturb_params(base_params, d1, d2, float(a), float(b))
            Z[i, j] = eval_adc_ppl(p, model, cal_data, device)
    return xs, ys, Z


def plot_landscapes(landscapes: list, alphas: list, output_path: str):
    """landscapes: list of (xs, ys, Z) tuples."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    vmin = min(Z.min() for _, _, Z in landscapes)
    vmax = min(np.percentile(Z, 95) for _, _, Z in landscapes)  # clip outliers

    for ax, (xs, ys, Z), alpha in zip(axes, landscapes, alphas):
        im = ax.contourf(xs, ys, Z.T, levels=30, cmap='RdYlBu_r',
                         vmin=vmin, vmax=vmax)
        ax.contour(xs, ys, Z.T, levels=15, colors='k', linewidths=0.3, alpha=0.4)
        ax.set_title(f'$\\alpha = {alpha}$', fontsize=13)
        ax.set_xlabel('direction $d_1$')
        ax.set_ylabel('direction $d_2$')
        plt.colorbar(im, ax=ax, label='ADC PPL')

    fig.suptitle('Loss landscape: FlatQuant INT4, varying $\\alpha$', fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"Saved to {output_path}")


if __name__ == '__main__':
    # Placeholder: actual argument parsing + loading goes here
    print("TODO: wire up checkpoint loading and eval pipeline")
```

---

### Шаг 3: Что именно должно быть видно на картинке

**α=0 (pure FP loss):**
- FlatQuant трансформы хорошо оптимизированы для FP loss → острый узкий минимум
  вдоль FP direction, но широкий (flat) вдоль ADC direction (ADC loss не обучался)
- На ADC PPL landscape: широкая неоптимальная долина (не converged по ADC axis)

**α=0.5 (mixed):**
- Минимум умеренно широкий и по FP, и по ADC directions
- Loss surface относительно гладкая и пологая (flat basin) в обоих направлениях
- Это соответствует "flatter minima = better generalisation" story

**α=1.0 (pure ADC loss):**
- Очень острый, глубокий минимум в ADC direction
- Bypass PPL far from optimal → но это и не оптимизировалось
- Интерпретация: переобученность под конкретную ADC noise realization

**Ожидаемый вывод для слайда:**
> "α=0.5 produces a flat wide minimum in both FP and ADC loss directions,
>  avoiding the sharp narrow minima of single-objective training."

---

### Альтернативный быстрый вариант: proxy loss landscape

Если запускать полный eval тяжело, можно сделать **proxy**:
вместо PPL использовать calibration MSE на фиксированном batch из 16 samples.

```python
def eval_proxy_loss(params, layer, cal_batch):
    """
    Compute MSE(layer_fp_output, layer_quantized_output) on cal_batch.
    Fast: single forward pass, no full model eval needed.
    """
    load_params_into_layer(params, layer)
    with torch.no_grad():
        y_fp   = layer.forward_fp(cal_batch)
        y_adc  = layer.forward_adc(cal_batch)
    return F.mse_loss(y_adc, y_fp).item()
```

Это намного быстрее (секунды vs часы) и даёт тот же качественный результат для визуализации.

---

## Summary: что запустить и когда

| # | Figure | Status | Action needed |
|---|--------|--------|---------------|
| 1 | `plot_int8_main_results` | ✅ data ready | write plot script |
| 2 | `plot_int4_ptq_progression` | ✅ data ready | write plot script |
| 3 | `plot_diag_roles` | ✅ data ready | write plot script |
| 4 | `plot_mlp_split_up_down` | ✅ data ready | write plot script |
| 5 | `plot_ptq_plateau` | ✅ data ready | write plot script |
| 6 | `plot_lora_ablations_grid` | ✅ data ready | write plot script |
| 7 | `plot_final_results_wiki_c4` | ✅ data ready | write plot script |
| 8 | `plot_deadzone_not_fundamental` | ✅ data ready | write plot script |
| 9 | `plot_loss_landscape_alpha` | ❌ NEW EXP | run 3 checkpoints + plot script |
| 10 | `fig_flatquant_intuition` | ❌ synthetic | write synthetic script |
| 11 | `fig_digital_vs_analog_path` | ❌ TikZ | draw in TikZ |
| 12 | `fig_adc_lora_post_vs_pre` | ❌ TikZ | draw in TikZ |
| 13 | `fig_contributions_summary` | ❌ TikZ | draw in TikZ |

**Только один реально новый эксперимент:** #9 (loss landscape).
Остальные — это скрипты на имеющихся данных и TikZ-схемы.
