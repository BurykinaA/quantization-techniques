"""
Configuration presets for ADC-aware INT4 quantization.

Each config is one step in the story:
  fp           full precision, no quantization
  int4_no_adc  INT4 FlatQuant transforms, evaluated WITHOUT the ADC floor
  best_ptq     INT4 FlatQuant transforms + symmetric ADC hardware model
  best_lora    best_ptq + post-ADC LoRA correction

best_ptq and best_lora share identical FlatQuant training (staged + diagonal
+ propagation alpha=0.5); they differ only in whether LoRA correction is applied.
"""

from dataclasses import dataclass


@dataclass
class BaseConfig:
    # Model
    model_name: str = "meta-llama/Llama-3.2-1B"
    torch_dtype: str = "float16"
    seed: int = 42
    output_dir: str = "./outputs"

    # ADC hardware constants. Symmetric ADC step size:
    #   delta = 2 * tile_in * q_x * q_w / (2^ba * k)
    bx: int = 4           # activation bits
    bw: int = 4           # weight bits
    ba: int = 8           # ADC output bits
    k: int = 16           # ADC parallelism (columns per ADC converter)
    mvm_limit: int = 256  # tile size (columns per matrix-vector multiply unit)

    # Calibration
    calibration_dataset: str = "wikitext2"
    calibration_method: str = "percentile"  # 99.9th percentile clips outliers
    calibration_max_length: int = 512
    calibration_batch_size: int = 4
    num_calibration_batches: int = 100

    # Evaluation
    eval_datasets: tuple[str, ...] = ("wikitext2", "c4")
    eval_split: str = "test"
    max_length: int = 2048
    stride: int = 1024            # 50% overlap for sliding-window PPL
    max_eval_samples: int = 1000  # C4 only (WikiText2 uses full test set)


@dataclass
class FPConfig(BaseConfig):
    """Full precision baseline — model loaded as-is, no quantization."""
    name: str = "fp"


@dataclass
class _SharedFlatQuantConfig(BaseConfig):
    """
    FlatQuant training settings shared by int4_no_adc, best_ptq and best_lora.
    Do not instantiate directly.

      Stage A: 30 epochs, MLP diagonal scaling only, no propagation
      Stage B: 10 epochs, lr x 0.1, adds attention diagonal, propagation alpha=0.5
    """
    fq_epochs: int = 30
    fq_lr: float = 0.005
    fq_nsamples: int = 1024
    fq_cali_bsz: int = 16
    fq_w_bits: int = 4
    fq_a_bits: int = 4

    # Learnable clipping (improves coverage of the quantization range)
    fq_lwc: bool = True   # weight clipping
    fq_lac: bool = True   # activation clipping

    # Diagonal per-channel scaling d in [1e-4, 10]
    fq_add_diag: bool = True
    fq_diag_mlp: bool = True

    # Stage B
    fq_stage_b_epochs: int = 10
    fq_stage_b_prop_alpha: float = 0.5   # x_cal = 0.5*x_adc + 0.5*x_fp
    fq_stage_b_diag_attn: bool = True

    # WandB project for result logging (empty string = no logging)
    wandb_project: str = "adc-optical-ptq"


@dataclass
class INT4NoADCConfig(_SharedFlatQuantConfig):
    """
    INT4 FlatQuant transforms evaluated WITHOUT the ADC floor (bypass mode).

    Same training as best_ptq, but at eval time floor(z_int / delta) is disabled,
    isolating the pure INT4 quantization error from the ADC hardware overhead.
    """
    name: str = "int4_no_adc"
    use_adc: bool = False


@dataclass
class BestPTQConfig(_SharedFlatQuantConfig):
    """INT4 FlatQuant + full symmetric ADC hardware model (best pure PTQ result)."""
    name: str = "best_ptq"
    use_adc: bool = True


@dataclass
class BestLoRAConfig(BestPTQConfig):
    """
    best_ptq + post-ADC LoRA residual correction:
        y = TiledLinearADC(x) + (alpha / r) * B(A(x.float()))

    LoRA parameters are FP32 and added AFTER the ADC floor, so gradients never
    flow through the non-differentiable floor.
    Training loss: L = L_CE + lambda * KL(student || teacher_fp), lambda=0.5, T=2.0
    Targets all 7 projections (MLP: down/up/gate; Attn: q/k/v/o).
    """
    name: str = "best_lora"

    lora_rank: int = 4
    lora_alpha: float = 8.0     # effective scaling = alpha / rank = 2.0
    lora_target_modules: tuple[str, ...] = (
        "down_proj", "up_proj", "gate_proj",
        "q_proj", "k_proj", "v_proj", "o_proj",
    )

    lora_loss: str = "ce_kl"
    lora_kl_weight: float = 0.5
    lora_kl_temperature: float = 2.0
    lora_epochs: int = 5
    lora_lr: float = 1e-4


# Story order: FP -> INT4 -> INT4+ADC -> INT4+ADC+LoRA
ALL_CONFIGS = [
    FPConfig(),
    INT4NoADCConfig(),
    BestPTQConfig(),
    BestLoRAConfig(),
]
