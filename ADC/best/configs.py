"""
Configuration presets for ADC-aware quantization configurations.

Each config represents one step in the quantization pipeline:
  fp             → full precision, no quantization
  int4_no_adc    → INT4 FlatQuant transforms, evaluated WITHOUT ADC floor
  best_ptq       → INT4 FlatQuant transforms + ADC hardware model  (global k)
  best_lora      → best_ptq + post-ADC LoRA correction
  best_ptq_k     → best_ptq + automatic per-layer k search (no FlatQuant recal)
  best_ptq_k_recal → best_ptq_k + FlatQuant recalibrated with per-layer k

Configs 2–4 use IDENTICAL FlatQuant training (staged + diagonal + propagation α=0.5).
The difference is evaluation mode and whether LoRA correction is applied.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class BaseConfig:
    # Model
    model_name: str = "meta-llama/Llama-3.2-1B"
    torch_dtype: str = "float16"
    seed: int = 42
    output_dir: str = "./outputs"

    # ADC hardware constants
    # δ = 2·tile_in·q_x·q_w / (2^{b_a}·k)
    # For INT4 (q=7): δ ≈ 2·256·7·7 / (256·16) ≈ 6.12  → very few useful bins
    # For INT8 (q=127): δ ≈ 2·256·127·127 / (256·16) ≈ 2016
    bx: int = 4           # activation bits
    bw: int = 4           # weight bits
    ba: int = 8           # ADC output bits
    k: int = 16           # ADC parallelism (# columns per ADC converter)
    mvm_limit: int = 256  # tile size (columns per matrix-vector multiply unit)

    # Calibration
    calibration_dataset: str = "wikitext2"
    calibration_method: str = "percentile"  # percentile (99.9th) >> absmax for INT4
    calibration_max_length: int = 512
    calibration_batch_size: int = 4
    num_calibration_batches: int = 100

    # Evaluation
    eval_datasets: Tuple[str, ...] = ("wikitext2", "c4")
    eval_split: str = "test"
    max_length: int = 2048
    stride: int = 1024       # 50% overlap for sliding window PPL
    max_eval_samples: int = 1000  # C4 only (WikiText2 uses full test set)


@dataclass
class FPConfig(BaseConfig):
    """
    Full precision baseline — model loaded as-is, no quantization.
    Sets the upper bound for model quality (~PPL 10.5 on WikiText2).
    """
    name: str = "fp"


@dataclass
class _SharedFlatQuantConfig(BaseConfig):
    """
    Shared FlatQuant training settings used by INT4NoADCConfig, BestPTQConfig,
    and BestLoRAConfig. Do not instantiate directly.

    Training recipe:
      Stage A: 30 epochs, MLP diagonal scaling only, no propagation
      Stage B: 10 epochs, lr×0.1, adds attention diagonal, propagation α=0.5
    """
    # FlatQuant calibration
    fq_epochs: int = 30
    fq_lr: float = 0.005
    fq_nsamples: int = 1024   # 1024 >> 128 for INT4 (more samples = more stable)
    fq_cali_bsz: int = 16
    fq_w_bits: int = 4
    fq_a_bits: int = 4

    # Learnable clipping (improves coverage of activation range)
    fq_lwc: bool = True   # learnable weight clipping
    fq_lac: bool = True   # learnable activation clipping

    # Diagonal per-channel scaling: bounded scaling d ∈ [1e-4, 10] per channel
    # Adds expressive power without changing the transform structure
    fq_add_diag: bool = True
    fq_diag_mlp: bool = True   # Stage A: train diagonal for MLP blocks only

    # Stage B: second pass with propagation and attention diagonal
    fq_stage_b_epochs: int = 10
    fq_stage_b_prop_alpha: float = 0.5   # x_cal = 0.5·x_adc + 0.5·x_fp
    fq_stage_b_diag_attn: bool = True    # add attention diagonal in stage B

    # Unipolar ADC: physical optical device has range [0, 2^ba − 1] only.
    # Mathematically equivalent to bipolar (shift-and-subtract), but models real hardware.
    unipolar_adc: bool = True

    # WandB project for result logging (empty string = no logging)
    wandb_project: str = "adc-optical-ptq"

    # Per-layer k: maps TiledLinearADC module path (or name substring) → k value.
    # Empty dict means use the global k for all layers.
    # Populated automatically by search_k_per_layer() in pipeline.py.
    k_per_layer: Dict[str, int] = field(default_factory=dict)
    # Candidate k values to sweep during the dead-rate search.
    k_search_candidates: Tuple[int, ...] = (4, 8, 16, 32, 64)
    # Accept the largest k whose dead_rate ≤ this threshold (5% → 95% of outputs useful).
    k_search_target_dead_rate: float = 0.05


@dataclass
class INT4NoADCConfig(_SharedFlatQuantConfig):
    """
    INT4 FlatQuant transforms evaluated WITHOUT ADC floor quantization (bypass mode).

    Same FlatQuant training as BestPTQConfig, but at eval time the ADC floor
    (floor(z_int / δ)) is disabled — only INT4 weight/activation quantization applies.

    This isolates the pure INT4 quantization error from the ADC hardware overhead,
    showing the PPL ceiling achievable with just INT4 quantization (~19 PPL).
    """
    name: str = "int4_no_adc"
    use_adc: bool = False   # disable ADC floor quantization at eval time


@dataclass
class BestPTQConfig(_SharedFlatQuantConfig):
    """
    INT4 FlatQuant + full ADC hardware model (TiledLinearADC with floor quantization).

    This is the best achievable result with pure PTQ (no learned correction).
    Shows the cost of the ADC hardware model on top of INT4 quantization (~27.6 PPL).
    """
    name: str = "best_ptq"
    use_adc: bool = True


@dataclass
class BestLoRAConfig(BestPTQConfig):
    """
    BestPTQConfig + post-ADC LoRA residual correction.

    After PTQ calibration, wraps each target layer with:
        y = TiledLinearADC(x) + (α/r) · B(A(x.float()))

    LoRA parameters are FP32, added AFTER the ADC floor quantization.
    Training loss: L = L_CE + λ·KL(student ‖ teacher_fp), λ=0.5, T=2.0

    Targets all 7 projections (MLP: down/up/gate; Attn: q/k/v/o).
    Achieves ADC PPL ~14.0 on WikiText2, ~23.3 on C4 — better than INT8 PTQ (14.4).
    """
    name: str = "best_lora"

    # LoRA architecture
    lora_rank: int = 4
    lora_alpha: float = 8.0     # effective scaling = alpha / rank = 2.0
    lora_target_modules: Tuple[str, ...] = (
        "down_proj", "up_proj", "gate_proj",   # MLP
        "q_proj", "k_proj", "v_proj", "o_proj" # Attention
    )
    lora_mode: str = "residual"  # post-ADC (not pre-ADC — pre-ADC diverges)

    # Training
    lora_loss: str = "ce_kl"          # CE + KL divergence from FP teacher
    lora_kl_weight: float = 0.5       # λ in L = CE + λ·KL
    lora_kl_temperature: float = 2.0  # temperature T for KL softmax
    lora_epochs: int = 5
    lora_lr: float = 1e-4


@dataclass
class BestPTQKConfig(BestPTQConfig):
    """
    BestPTQConfig + automatic per-layer k search (no FlatQuant recalibration).

    After FlatQuant calibration, runs one bypass forward pass to capture y_int
    statistics per layer, picks the largest k where dead_rate ≤ 5%, applies the
    found k values in-place to the TiledLinearADC layers, then evaluates.

    FlatQuant transforms were trained with global k=16 (slight delta mismatch),
    but transforms are already near-optimal — this is a fast, strong baseline.
    """
    name: str = "best_ptq_k"


@dataclass
class BestPTQKRecalConfig(BestPTQKConfig):
    """
    BestPTQKConfig + FlatQuant recalibrated with the found per-layer k.

    Pipeline:
      1. Train FlatQuant with global k (same as best_ptq) — discovers k_per_layer
      2. Reload FP model, re-run FlatQuant with per-layer k active in each
         FlatQuantLinear's ADC simulation → transforms optimised for actual δ
      3. Evaluate

    Quality ceiling: transforms are trained on the correct per-layer delta.
    Cost: ~2× total FlatQuant time vs best_ptq_k.
    """
    name: str = "best_ptq_k_recal"
    fq_recal_with_k: bool = True


# All configs in order: tells the story FP → INT4 → INT4+ADC → INT4+ADC+LoRA → per-layer k
ALL_CONFIGS = [
    FPConfig(),
    INT4NoADCConfig(),
    BestPTQConfig(),
    BestLoRAConfig(),
    BestPTQKConfig(),
    BestPTQKRecalConfig(),
]
