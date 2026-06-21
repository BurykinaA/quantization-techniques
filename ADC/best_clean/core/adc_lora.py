"""
ADC-LoRA post-correction for FlatQuant PTQ models.

ResidualLoRATiledLinearADC (post-ADC, QLoRA-style):
    y = frozen_TiledLinearADC(x) + scaling * lora_B(lora_A(x.float()))

The correction is added AFTER the ADC floor, so gradients never touch the weight
quantizer or the ADC clamp. LoRA matrices are FP32 for stable training.
"""
import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

from .adc_layers import TiledLinearADC


class ResidualLoRATiledLinearADC(nn.Module):
    """
    Post-ADC residual LoRA wrapper.

        y = TiledLinearADC(x)  +  scaling * lora_B(lora_A(x.float()))

    lora_B initialized to zero → initial output identical to frozen model.
    """

    def __init__(self,
                 tiled_layer: TiledLinearADC,
                 r: int = 4,
                 alpha: float = 8.0,
                 dropout: float = 0.0):
        super().__init__()
        self.tiled_layer = tiled_layer
        self.scaling = alpha / r
        dev = tiled_layer.tiles[0].weight.device

        self.lora_A = nn.Linear(tiled_layer.in_features_total, r,
                                bias=False, dtype=torch.float32, device=dev)
        self.lora_B = nn.Linear(r, tiled_layer.out_features,
                                bias=False, dtype=torch.float32, device=dev)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        for p in self.tiled_layer.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_base = self.tiled_layer(x)
        y_lora = self.lora_B(self.lora_A(self.dropout(x.float()))) * self.scaling
        return y_base + y_lora.to(y_base.dtype)

    def set_quantizer_mode(self, mode): self.tiled_layer.set_quantizer_mode(mode)
    def set_adc_bits(self, ba):         self.tiled_layer.set_adc_bits(ba)
    def get_adc_bits(self):             return self.tiled_layer.get_adc_bits()

    def get_num_trainable_params(self) -> int:
        return self.lora_A.weight.numel() + self.lora_B.weight.numel()

    def get_compression_ratio(self) -> float:
        full = self.tiled_layer.in_features_total * self.tiled_layer.out_features
        lora = self.get_num_trainable_params()
        return full / lora if lora > 0 else float('inf')


def apply_adc_lora(
    model: nn.Module,
    target_modules: list,
    rank: int = 4,
    lora_alpha: float = 8.0,
    layer_indices: set | None = None,  # None = all layers
) -> nn.Module:
    """
    Wrap the .linear (TiledLinearADC) inside target FlatQuantLinear modules with
    a post-ADC residual LoRA adapter.

    target_modules: projection name suffixes, e.g. ["down_proj", "o_proj"]
    layer_indices:  which transformer layer indices to apply LoRA to (None = all)
    """
    from .flat_quant import FlatQuantLinear

    count = 0
    for name, module in model.named_modules():
        attr = name.rsplit(".", 1)[-1] if "." in name else name
        if attr not in target_modules:
            continue
        if not isinstance(module, FlatQuantLinear):
            continue
        if not isinstance(module.linear, TiledLinearADC):
            continue

        if layer_indices is not None:
            m = re.search(r'\.layers\.(\d+)\.', name)
            if m is None or int(m.group(1)) not in layer_indices:
                continue

        module.linear = ResidualLoRATiledLinearADC(
            module.linear, r=rank, alpha=lora_alpha)
        count += 1

    layer_info = (f"layers={sorted(layer_indices)[:3]}..."
                  if layer_indices is not None else "all layers")
    print(f"[ADC-LoRA] Applied residual LoRA (r={rank}, α={lora_alpha}) "
          f"to {count} modules in {layer_info}: {target_modules}")
    return model


# ============================================================
#  calibrate_adc_lora
# ============================================================

def calibrate_adc_lora(
    model: nn.Module,
    dataloader,
    device: torch.device,
    nsamples: int = 1024,
    cali_bsz: int = 4,
    epochs: int = 5,
    lora_lr: float = 1e-4,
    lora_loss: str = "ce",             # "ce" or "ce_kl"
    teacher_name_or_path: str | None = None,
    kl_weight: float = 0.5,
    kl_temperature: float = 2.0,
) -> nn.Module:
    """
    Fine-tune LoRA adapters.

    lora_loss="ce"     — standard LM cross-entropy
    lora_loss="ce_kl"  — CE + λ·KL(FP_teacher || student); requires teacher_name_or_path
    """
    # Freeze everything except LoRA adapters
    for n, p in model.named_parameters():
        is_lora = ("lora_A.weight" in n or "lora_B.weight" in n   # nn.Linear variant
                   or ("lora_A" in n and "weight" not in n)        # nn.Parameter variant
                   or ("lora_B" in n and "weight" not in n))
        p.requires_grad_(is_lora)

    lora_params = [p for p in model.parameters() if p.requires_grad]
    if not lora_params:
        print("[ADC-LoRA] Warning: no LoRA parameters found — skipping calibration")
        return model

    n_params = sum(p.numel() for p in lora_params)
    print(f"[ADC-LoRA] Trainable params: {n_params:,} across {len(lora_params)} tensors")

    optimizer = torch.optim.AdamW(lora_params, lr=lora_lr)

    # Load teacher for KL loss (on CPU to save GPU memory)
    teacher = None
    if lora_loss == "ce_kl":
        if teacher_name_or_path is None:
            print("[ADC-LoRA] Warning: ce_kl requested but teacher_name_or_path=None → "
                  "falling back to ce")
            lora_loss = "ce"
        else:
            from transformers import AutoModelForCausalLM
            print(f"[ADC-LoRA] Loading FP teacher on CPU: {teacher_name_or_path}")
            teacher = AutoModelForCausalLM.from_pretrained(
                teacher_name_or_path,
                torch_dtype=torch.float16,
                device_map="cpu",
            ).eval()
            for p in teacher.parameters():
                p.requires_grad_(False)

    # Collect calibration batches
    samples = []
    for batch in dataloader:
        if len(samples) * cali_bsz >= nsamples:
            break
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        samples.append((input_ids, attention_mask))

    print(f"[ADC-LoRA] Training {epochs} epochs on {len(samples)} batches  "
          f"lr={lora_lr}  loss={lora_loss}")

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for input_ids, attention_mask in samples:
            labels = input_ids.clone()
            if attention_mask is not None:
                labels[attention_mask == 0] = -100

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            ce_loss = outputs.loss

            if lora_loss == "ce_kl" and teacher is not None:
                with torch.no_grad():
                    t_out = teacher(
                        input_ids=input_ids.cpu(),
                        attention_mask=attention_mask.cpu() if attention_mask is not None else None,
                    )
                    teacher_logits = t_out.logits.to(device).float()
                student_logits = outputs.logits.float()
                kl = F.kl_div(
                    F.log_softmax(student_logits / kl_temperature, dim=-1),
                    F.softmax(teacher_logits  / kl_temperature, dim=-1),
                    reduction="batchmean",
                ) * (kl_temperature ** 2)
                loss = ce_loss + kl_weight * kl
            else:
                loss = ce_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[ADC-LoRA] Epoch {epoch + 1}/{epochs}  "
              f"avg loss={total_loss / len(samples):.4f}")

    if teacher is not None:
        del teacher
        torch.cuda.empty_cache()

    model.eval()
    return model
