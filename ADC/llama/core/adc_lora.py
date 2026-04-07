"""
ADC-LoRA post-correction for FlatQuant PTQ models.

Residual (post-ADC) approach — QLoRA-style:

    y = frozen_TiledLinearADC(x)              # exact quantized ADC path, frozen
    y += scaling * lora_B(lora_A(x.float()))  # FP32 residual, added AFTER ADC output

Gradients never touch round_ste / ADC clamp → stable fp32 training.
Base TiledLinearADC weights stay frozen in fp16.
"""
import math
import torch
import torch.nn as nn

from .adc_layers import TiledLinearADC


class ResidualLoRATiledLinearADC(nn.Module):
    """
    Residual LoRA wrapper for TiledLinearADC.

    The frozen TiledLinearADC produces the quantized ADC output; a low-rank
    FP32 linear path adds a residual correction:

        y = TiledLinearADC(x)  +  scaling * lora_B(lora_A(x.float()))

    lora_B is initialized to zero so the initial output is identical to the
    frozen quantized model.
    """

    def __init__(self,
                 tiled_layer: TiledLinearADC,
                 r: int = 4,
                 alpha: float = 8.0,
                 dropout: float = 0.0):
        super().__init__()

        self.tiled_layer = tiled_layer
        self.scaling = alpha / r

        in_dim  = tiled_layer.in_features_total
        out_dim = tiled_layer.out_features
        dev = tiled_layer.tiles[0].weight.device

        # FP32 adapters — always float32 regardless of base model dtype
        self.lora_A = nn.Linear(in_dim, r,   bias=False, dtype=torch.float32, device=dev)
        self.lora_B = nn.Linear(r,   out_dim, bias=False, dtype=torch.float32, device=dev)

        # Standard LoRA init: A ~ Kaiming, B = 0
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # Dropout applied to input (not to weight matrix)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Freeze all base layer parameters
        for p in self.tiled_layer.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_base = self.tiled_layer(x)                         # fp16, frozen ADC path
        x_fp   = self.dropout(x.float())                     # fp32, dropout on input
        y_lora = self.lora_B(self.lora_A(x_fp)) * self.scaling
        return y_base + y_lora.to(y_base.dtype)              # cast residual back

    # Delegate ADC control methods to the wrapped layer
    def set_quantizer_mode(self, mode: str):
        self.tiled_layer.set_quantizer_mode(mode)

    def set_adc_bits(self, ba: int):
        self.tiled_layer.set_adc_bits(ba)

    def get_adc_bits(self) -> int:
        return self.tiled_layer.get_adc_bits()

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
) -> nn.Module:
    """
    Replace .linear (TiledLinearADC) inside target FlatQuantLinear modules with
    ResidualLoRATiledLinearADC.

    target_modules: list of projection name suffixes, e.g. ["down_proj", "o_proj"].
    Must be called after reparameterize_model + LlamaADCConverter.
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
        module.linear = ResidualLoRATiledLinearADC(
            module.linear, r=rank, alpha=lora_alpha
        )
        count += 1

    print(f"[ADC-LoRA] Applied residual LoRA (r={rank}, α={lora_alpha}) "
          f"to {count} modules: {target_modules}")
    return model


def calibrate_adc_lora(
    model: nn.Module,
    dataloader,
    device: torch.device,
    nsamples: int = 1024,
    cali_bsz: int = 4,
    epochs: int = 5,
    lora_lr: float = 1e-4,
) -> nn.Module:
    """
    Fine-tune LoRA adapters using LM cross-entropy loss.
    All parameters except lora_A.weight / lora_B.weight are frozen.
    Passes attention_mask and masks padding in labels.
    """
    # Freeze everything except LoRA adapters
    for n, p in model.named_parameters():
        p.requires_grad_("lora_A.weight" in n or "lora_B.weight" in n)

    lora_params = [p for n, p in model.named_parameters() if p.requires_grad]
    if not lora_params:
        print("[ADC-LoRA] Warning: no LoRA parameters found — skipping calibration")
        return model

    n_params = sum(p.numel() for p in lora_params)
    print(f"[ADC-LoRA] Trainable params: {n_params:,} across {len(lora_params)} tensors")

    optimizer = torch.optim.AdamW(lora_params, lr=lora_lr)

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

    print(f"[ADC-LoRA] Training {epochs} epochs on {len(samples)} batches "
          f"(lr={lora_lr})")

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for input_ids, attention_mask in samples:
            # Mask padding tokens in labels
            labels = input_ids.clone()
            if attention_mask is not None:
                labels[attention_mask == 0] = -100

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 1 == 0:
            print(f"[ADC-LoRA] Epoch {epoch + 1}/{epochs}  "
                  f"avg loss={total_loss / len(samples):.4f}")

    model.eval()
    return model
