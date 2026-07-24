"""
Configuration presets for ADC-aware ViT quantization.

Adapted from the LLaMA ADC pipeline to timm ViT / ImageNet top-1, using the
ADC hardware model exactly as implemented in ADC/llama/core (bipolar signed
ADC — floor(y/δ).clamp(na, pa)·δ).  Four configurations:

  fp            → full precision timm model, no quantization
  int4_no_adc   → INT4 FlatQuant transforms, ADC floor bypassed
  ptq           → INT4 FlatQuant + ADC hardware model
  ptq_lora      → ptq + post-ADC residual LoRA (rank 4)

k (ADC parallelism) is exposed as a CLI flag; default k=4 per request.  The
thesis used a global k=16 (with 4 as the smallest per-layer-search candidate).
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class ViTBaseConfig:
    # Model / data
    model_name: str = "vit_tiny_patch16_224"
    seed: int = 42
    output_dir: str = "./outputs_vit"
    data_dir: str | None = None       # falls back to $IMAGENET_ROOT

    # ADC hardware constants (bipolar signed ADC, as in ADC/llama/core)
    #   δ = 2·tile_in·q_x·q_w / (2^ba·k)
    bx: int = 4           # activation bits
    bw: int = 4           # weight bits
    ba: int = 8           # ADC output bits
    k: int = 4            # ADC parallelism (CLI-overridable; thesis default = 16)
    mvm_limit: int = 256  # tile size (columns per MVM unit)

    # Evaluation
    val_batch_size: int = 128
    val_portion: float = 1.0   # <1.0 uses a strided val subset (fast checks)
    num_workers: int = 8

    use_adc: bool = True   # whether the ADC floor is active at eval time


@dataclass
class _SharedFlatQuantConfig(ViTBaseConfig):
    """FlatQuant + ADC training settings shared by all quantized ViT configs.

    Thesis-scale defaults; the pipeline's --smoke flag shrinks epochs/samples
    for a fast correctness run.
    """
    # FlatQuant calibration (thesis scale)
    fq_epochs: int = 30            # Stage A epochs
    fq_stage_b_epochs: int = 5    # Stage B epochs (lr × 0.1)
    fq_lr: float = 5e-3
    fq_nsamples: int = 1024
    fq_cali_bsz: int = 256
    fq_w_bits: int = 4
    fq_a_bits: int = 4

    # Learnable clipping
    fq_lwc: bool = True    # learnable weight clipping
    fq_lac: bool = True    # learnable activation clipping

    # Per-channel diagonal scaling
    fq_add_diag: bool = True
    fq_stage_b_prop_alpha: float = 0.5   # x_cal = 0.5·x_adc + 0.5·x_fp
    fq_stage_b_diag_attn: bool = True    # add attention diagonal in stage B

    # Calibration set for ADC percentile scales (drawn from train split).
    num_calibration_batches: int = 64
    calibration_method: str = "percentile"   # percentile (99.9th) >> absmax for INT4

    # Per-layer k search (unused by the four base configs; kept for parity).
    k_per_layer: Dict[str, int] = field(default_factory=dict)
    k_search_candidates: Tuple[int, ...] = (4, 8, 16, 32, 64)
    k_search_range_percentile: float = 0.999

    wandb_project: str = ""

    def apply_smoke(self) -> None:
        """Shrink calibration for a fast correctness run."""
        self.fq_epochs = 2
        self.fq_stage_b_epochs = 1
        self.fq_nsamples = 128
        self.fq_cali_bsz = 8
        self.num_calibration_batches = 8


@dataclass
class ViTFPConfig(ViTBaseConfig):
    """Full-precision timm baseline — no quantization."""
    name: str = "fp"


@dataclass
class ViTInt4NoADCConfig(_SharedFlatQuantConfig):
    """INT4 FlatQuant transforms, ADC floor DISABLED (bypass) at eval time.
    Isolates the pure INT4 quantization error from the ADC hardware overhead."""
    name: str = "int4_no_adc"
    use_adc: bool = False


@dataclass
class ViTPTQConfig(_SharedFlatQuantConfig):
    """INT4 FlatQuant + full ADC hardware model (no learned correction)."""
    name: str = "ptq"
    use_adc: bool = True


@dataclass
class ViTPTQLoRAConfig(ViTPTQConfig):
    """ptq + post-ADC residual LoRA correction.

        y = TiledLinearADC(x) + (α/r)·B(A(x.float()))

    LoRA is FP32, added after the ADC floor.  Trained with CE (labels) +
    λ·KL(student‖FP-teacher), matching the thesis LoRA recipe (r=4, α=8).
    """
    name: str = "ptq_lora"

    lora_rank: int = 4
    lora_alpha: float = 8.0     # effective scaling = α / r = 2.0
    lora_target_modules: Tuple[str, ...] = ("qkv", "proj", "fc1", "fc2")
    lora_mode: str = "residual"

    lora_loss: str = "ce_kl"          # CE + KL to FP teacher
    lora_kl_weight: float = 0.5       # λ
    lora_kl_temperature: float = 2.0  # T
    lora_epochs: int = 5
    lora_lr: float = 1e-4
    lora_nsamples: int = 1024
    lora_cali_bsz: int = 16

    def apply_smoke(self) -> None:
        super().apply_smoke()
        self.lora_epochs = 2
        self.lora_nsamples = 128
        self.lora_cali_bsz = 8


CONFIG_MAP = {
    "fp": ViTFPConfig,
    "int4_no_adc": ViTInt4NoADCConfig,
    "ptq": ViTPTQConfig,
    "ptq_lora": ViTPTQLoRAConfig,
}


def build_config(name: str, **overrides):
    """Instantiate a config by name, applying field overrides (model_name, k, …)."""
    if name not in CONFIG_MAP:
        raise ValueError(f"Unknown config '{name}'. Choose from {list(CONFIG_MAP)}.")
    cfg = CONFIG_MAP[name]()
    for key, val in overrides.items():
        if val is not None and hasattr(cfg, key):
            setattr(cfg, key, val)
    return cfg
