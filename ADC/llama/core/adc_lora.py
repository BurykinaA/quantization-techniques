"""
ADC-LoRA post-correction for FlatQuant PTQ models.

After reparameterize + ADC conversion, each FlatQuantLinear.linear is a TiledLinearADC.
This module replaces the TiledLinearADC with LoRATiledLinearADC, which computes:

    Y = QA(Qx(X) @ Qw(W + scaling * A_i @ B_i))  [per tile]

LoRA adapters are trained via LM cross-entropy with all other parameters frozen.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .grad_functions import round_ste, safe_divide, floor_ste
from .utils import compute_kurtosis_loss
from .adc_layers import TiledLinearADC


class LoRATiledLinearADC(nn.Module):
    """
    ADC-LoRA wrapper for TiledLinearADC.

    Applies per-tile LoRA adapters: each tile gets its own A_i / B_i matrices.
    The effective weight per tile is W_i + scaling * A_i @ B_i, which is then
    quantized through the full ADC pipeline (activation → weight → integer MM → ADC clamp).

    Initialized so that A_i @ B_i = 0 (B_i = zeros), so initial output equals
    the frozen quantized model.
    """

    def __init__(self,
                 tiled_layer: TiledLinearADC,
                 r: int = 8,
                 alpha: float = 16.0,
                 dropout: float = 0.0):
        super().__init__()

        self.tiled_layer = tiled_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.n_tiles = tiled_layer.n_tiles
        self.in_features_total = tiled_layer.in_features_total
        self.out_features = tiled_layer.out_features
        self.in_features_tile = tiled_layer.in_features_tile

        # Freeze all base tile weights
        for tile in self.tiled_layer.tiles:
            tile.weight.requires_grad = False
            if tile.bias is not None:
                tile.bias.requires_grad = False

        # Per-tile LoRA: A (out_features, r), B (r, in_features_tile)
        self.lora_A = nn.ParameterList([
            nn.Parameter(torch.zeros(self.out_features, r))
            for _ in range(self.n_tiles)
        ])
        self.lora_B = nn.ParameterList([
            nn.Parameter(torch.zeros(r, self.in_features_tile))
            for _ in range(self.n_tiles)
        ])

        # Standard LoRA init: A ~ Kaiming, B = 0
        for lora_a in self.lora_A:
            nn.init.kaiming_uniform_(lora_a, a=math.sqrt(5))

        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def set_quantizer_mode(self, mode: str):
        self.tiled_layer.set_quantizer_mode(mode)

    def set_adc_bits(self, ba: int):
        self.tiled_layer.set_adc_bits(ba)

    def get_adc_bits(self) -> int:
        return self.tiled_layer.get_adc_bits()

    def get_auxiliary_losses(self) -> dict:
        total = torch.tensor(0.0)
        for i, tile in enumerate(self.tiled_layer.tiles):
            lora_weight = self.scaling * (self.lora_A[i] @ self.lora_B[i])
            effective_weight = tile.weight + lora_weight
            if tile.use_kurtosis_loss:
                total = total + tile.kurtosis_weight * compute_kurtosis_loss(
                    effective_weight, tile.target_kurtosis
                )
        return {'total': total}

    def train(self, mode: bool = True):
        super().train(mode)
        self.tiled_layer.train(mode)
        return self

    def eval(self):
        super().eval()
        self.tiled_layer.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features_total:
            raise ValueError(
                f"Expected last dim={self.in_features_total}, got {x.shape[-1]}"
            )

        orig_shape = x.shape
        x2d = x.reshape(-1, self.in_features_total)
        batch_size_2d = x2d.shape[0]

        y2d = torch.zeros(
            batch_size_2d, self.out_features, dtype=x2d.dtype, device=x2d.device
        )

        tile_in = self.in_features_tile
        for i, tile in enumerate(self.tiled_layer.tiles):
            xi = (x2d.narrow(1, i * tile_in, tile_in)
                  if x2d.is_contiguous()
                  else x2d[:, i * tile_in:(i + 1) * tile_in])

            # Effective weight = W_i + scaling * A_i @ B_i
            lora_weight = self.scaling * (self.lora_A[i] @ self.lora_B[i])
            if self.training:
                lora_weight = self.lora_dropout(lora_weight)
            effective_weight = tile.weight + lora_weight

            # Activation quantization
            act_q = tile.activation_quantizer
            s_x = act_q.scale

            if act_q.symmetric:
                code_x = round_ste(safe_divide(xi, s_x))
                code_x = torch.clamp(code_x, act_q.qmin, act_q.qmax)
            else:
                zp_x = act_q.zero_point
                code_x_temp = round_ste(safe_divide(xi, s_x) + zp_x)
                code_x_temp = torch.clamp(code_x_temp, 0, act_q.qmax)
                code_x = (code_x_temp - tile.C
                          if tile.ashift else code_x_temp - zp_x)

            # Weight quantization on effective weight
            w_q = tile.weight_quantizer
            s_w_vec = w_q.scale
            s_w_b = s_w_vec.view(-1, 1)
            code_w = round_ste(safe_divide(effective_weight, s_w_b))
            code_w = torch.clamp(code_w, w_q.qmin, w_q.qmax)

            # Integer MVM
            y_int = F.linear(code_x, code_w, bias=None)

            # ADC quantization (clamp to ADC range)
            y_adc_codes = floor_ste(y_int / tile.delta)
            y_adc_codes = torch.clamp(y_adc_codes, tile.na, tile.pa)
            adc_output = y_adc_codes * tile.delta

            # Dequantize
            y_real = adc_output * s_x * s_w_vec

            # Asymmetric correction
            if not act_q.symmetric:
                wq_sum = code_w.sum(dim=1)
                if tile.ashift:
                    zp_x = act_q.zero_point
                    y_real = y_real - (zp_x * s_x) * (s_w_vec * wq_sum)
                    y_real = y_real + (tile.C * s_x) * (s_w_vec * wq_sum)

            if tile.bias is not None:
                y_real = y_real + tile.bias

            y2d.add_(y_real)

        return y2d.reshape(*orig_shape[:-1], self.out_features)

    def merge_lora_weights(self):
        """Merge LoRA into base tile weights (makes LoRA permanent)."""
        with torch.no_grad():
            for i, tile in enumerate(self.tiled_layer.tiles):
                lora_weight = self.scaling * (self.lora_A[i] @ self.lora_B[i])
                tile.weight.add_(lora_weight)
                self.lora_A[i].zero_()
                self.lora_B[i].zero_()

    def get_num_trainable_params(self) -> int:
        return sum(
            self.lora_A[i].numel() + self.lora_B[i].numel()
            for i in range(self.n_tiles)
        )

    def get_compression_ratio(self) -> float:
        full = self.in_features_total * self.out_features
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
    LoRATiledLinearADC.  target_modules is a list of projection name suffixes,
    e.g. ["down_proj", "o_proj"].

    Must be called after reparameterize_model + LlamaADCConverter (so .linear
    is already a TiledLinearADC).
    """
    # Inline import to avoid circular dependency at module load time
    from .flat_quant import FlatQuantLinear

    count = 0
    for name, module in model.named_modules():
        attr = name.rsplit(".", 1)[-1] if "." in name else name
        if attr not in target_modules:
            continue
        if not isinstance(module, FlatQuantLinear):
            continue
        if not isinstance(module.linear, TiledLinearADC):
            continue  # not yet converted — skip
        module.linear = LoRATiledLinearADC(
            module.linear, r=rank, alpha=lora_alpha
        )
        count += 1

    print(f"[ADC-LoRA] Applied LoRA (r={rank}, α={lora_alpha}) to {count} modules: "
          f"{target_modules}")
    return model


def calibrate_adc_lora(
    model: nn.Module,
    dataloader,
    device: torch.device,
    nsamples: int = 1024,
    cali_bsz: int = 4,
    epochs: int = 30,
    lora_lr: float = 1e-3,
) -> nn.Module:
    """
    Fine-tune LoRA adapters using LM cross-entropy loss.
    All parameters except lora_A / lora_B are frozen.
    """
    # Freeze everything except LoRA adapters
    for n, p in model.named_parameters():
        p.requires_grad_("lora_A" in n or "lora_B" in n)

    lora_params = [p for n, p in model.named_parameters() if p.requires_grad]
    if not lora_params:
        print("[ADC-LoRA] Warning: no LoRA parameters found — skipping calibration")
        return model

    n_params = sum(p.numel() for p in lora_params)
    print(f"[ADC-LoRA] Trainable params: {n_params:,} across {len(lora_params)} tensors")

    optimizer = torch.optim.AdamW(lora_params, lr=lora_lr)

    # Collect up to nsamples tokens worth of batches
    samples = []
    for batch in dataloader:
        if len(samples) * cali_bsz >= nsamples:
            break
        if isinstance(batch, dict):
            input_ids = batch["input_ids"].to(device)
        else:
            input_ids = batch.to(device)
        samples.append(input_ids)

    print(f"[ADC-LoRA] Training {epochs} epochs on {len(samples)} batches "
          f"(lr={lora_lr})")

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for input_ids in samples:
            # HF CausalLM shifts labels internally: pass same tensor for both
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"[ADC-LoRA] Epoch {epoch + 1}/{epochs}  "
                  f"avg loss={total_loss / len(samples):.4f}")

    model.eval()
    return model
