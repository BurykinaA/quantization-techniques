"""
ADC-LoRA post-correction for FlatQuant PTQ models.

Two variants:

  ResidualLoRATiledLinearADC  (mode="residual", default)
    Post-ADC correction — QLoRA-style:
        y = frozen_TiledLinearADC(x) + scaling * lora_B(lora_A(x.float()))
    Stable: gradients never touch Qw / ADC clamp. LoRA in fp32.

  PreADCLoRATiledLinearADC  (mode="pre_adc")
    Inside-quantization correction — RAOQ-style:
        y = QA(Qx(X) @ Qw(W + scaling * B_i @ A_i))   [per tile]
    LoRA matrices in fp32, cast to model dtype before adding to tile weights.
    B initialized to zero so initial output = frozen model.
"""
import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

from .grad_functions import round_ste, safe_divide, floor_ste
from .adc_layers import TiledLinearADC


# ============================================================
#  Variant 1: Residual (post-ADC) LoRA
# ============================================================

class ResidualLoRATiledLinearADC(nn.Module):
    """
    Post-ADC residual LoRA wrapper.

        y = TiledLinearADC(x)  +  scaling * lora_B(lora_A(x.float()))

    lora_B initialized to zero → initial output identical to frozen model.
    dtype sets the adapter weights and arithmetic (fp32 by default); the
    initialization is drawn in fp32 and then cast, so it does not depend on it.
    """

    def __init__(self,
                 tiled_layer: TiledLinearADC,
                 r: int = 4,
                 alpha: float = 8.0,
                 dropout: float = 0.0,
                 dtype: torch.dtype = torch.float32):
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
        self.lora_A.to(dtype)
        self.lora_B.to(dtype)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        for p in self.tiled_layer.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_base = self.tiled_layer(x)
        x_lora = self.dropout(x.to(self.lora_A.weight.dtype))
        y_lora = self.lora_B(self.lora_A(x_lora)) * self.scaling
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


# ============================================================
#  Variant 2: Pre-ADC (inside-quantization) LoRA
# ============================================================

class PreADCLoRATiledLinearADC(nn.Module):
    """
    Pre-ADC LoRA: correction inside quantization path.

    Per tile i:
        effective_W_i = W_i + (scaling * B_i @ A_i).to(W_i.dtype)
        y_i = QA(Qx(x_i) @ Qw(effective_W_i))

    A_i, B_i in fp32 — avoids fp16 Adam divergence.
    B_i initialized to zero → initial output = frozen model.
    """

    def __init__(self,
                 tiled_layer: TiledLinearADC,
                 r: int = 4,
                 alpha: float = 8.0):
        super().__init__()
        self.tiled_layer = tiled_layer
        self.scaling = alpha / r
        self.n_tiles = tiled_layer.n_tiles
        self.in_features_total = tiled_layer.in_features_total
        self.out_features = tiled_layer.out_features
        self.in_features_tile = tiled_layer.in_features_tile
        dev = tiled_layer.tiles[0].weight.device

        # Per-tile LoRA in fp32: A (r × tile_in), B (out × r)
        self.lora_A = nn.ParameterList([
            nn.Parameter(torch.zeros(r, self.in_features_tile,
                                     dtype=torch.float32, device=dev))
            for _ in range(self.n_tiles)
        ])
        self.lora_B = nn.ParameterList([
            nn.Parameter(torch.zeros(self.out_features, r,
                                     dtype=torch.float32, device=dev))
            for _ in range(self.n_tiles)
        ])
        for a in self.lora_A:
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))

        for p in self.tiled_layer.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x2d = x.reshape(-1, self.in_features_total)
        y2d = torch.zeros(x2d.shape[0], self.out_features,
                          dtype=x2d.dtype, device=x2d.device)

        tile_in = self.in_features_tile
        for i, tile in enumerate(self.tiled_layer.tiles):
            xi = (x2d.narrow(1, i * tile_in, tile_in)
                  if x2d.is_contiguous()
                  else x2d[:, i * tile_in:(i + 1) * tile_in])

            # Effective weight: W_i + fp32 correction cast to tile dtype
            delta = (self.scaling * (self.lora_B[i] @ self.lora_A[i])).to(tile.weight.dtype)
            effective_w = tile.weight + delta

            # Activation quantization
            act_q = tile.activation_quantizer
            s_x = act_q.scale
            if act_q.symmetric:
                code_x = round_ste(safe_divide(xi, s_x))
                code_x = torch.clamp(code_x, act_q.qmin, act_q.qmax)
            else:
                zp_x = act_q.zero_point
                tmp = round_ste(safe_divide(xi, s_x) + zp_x)
                tmp = torch.clamp(tmp, 0, act_q.qmax)
                code_x = tmp - tile.C if tile.ashift else tmp - zp_x

            # Weight quantization on effective weight
            w_q = tile.weight_quantizer
            s_w = w_q.scale
            s_w_b = s_w.view(-1, 1)
            code_w = round_ste(safe_divide(effective_w, s_w_b))
            code_w = torch.clamp(code_w, w_q.qmin, w_q.qmax)

            # Integer MVM + ADC clamp
            y_int = F.linear(code_x, code_w, bias=None)
            y_adc = floor_ste(y_int / tile.delta)
            y_adc = torch.clamp(y_adc, tile.na, tile.pa) * tile.delta
            y_real = y_adc * s_x * s_w

            if not act_q.symmetric:
                wq_sum = code_w.sum(dim=1)
                if tile.ashift:
                    zp_x = act_q.zero_point
                    y_real = y_real - (zp_x * s_x) * (s_w * wq_sum)
                    y_real = y_real + (tile.C * s_x) * (s_w * wq_sum)

            if tile.bias is not None:
                y_real = y_real + tile.bias
            y2d.add_(y_real)

        return y2d.reshape(*orig_shape[:-1], self.out_features)

    def set_quantizer_mode(self, mode): self.tiled_layer.set_quantizer_mode(mode)
    def set_adc_bits(self, ba):         self.tiled_layer.set_adc_bits(ba)
    def get_adc_bits(self):             return self.tiled_layer.get_adc_bits()

    def get_num_trainable_params(self) -> int:
        return sum(a.numel() + b.numel()
                   for a, b in zip(self.lora_A, self.lora_B))


# ============================================================
#  apply_adc_lora
# ============================================================

def apply_adc_lora(
    model: nn.Module,
    target_modules: list,
    rank: int = 4,
    lora_alpha: float = 8.0,
    mode: str = "residual",        # "residual" or "pre_adc"
    layer_indices: set | None = None,  # None = all layers
    lora_dtype: torch.dtype = torch.float32,
) -> nn.Module:
    """
    Wrap the TiledLinearADC of every target projection: the .linear of a
    FlatQuantLinear, or the projection itself when the model has no FlatQuant
    wrappers (--preprocess_method none / smooth_quant).

    target_modules: projection name suffixes, e.g. ["down_proj", "o_proj"]
    layer_indices:  which transformer layer indices to apply LoRA to (None = all)
    mode:           "residual" (post-ADC, default) or "pre_adc" (inside quantization)
    lora_dtype:     adapter weights and arithmetic; residual mode only
    """
    from .flat_quant import FlatQuantLinear

    if lora_dtype != torch.float32 and mode != "residual":
        raise ValueError(f"lora_dtype={lora_dtype} requires mode='residual'")

    count = 0
    for name, module in list(model.named_modules()):
        parent_name, _, attr = name.rpartition(".")
        if attr not in target_modules:
            continue
        if isinstance(module, FlatQuantLinear):
            parent, child = module, "linear"
        elif isinstance(module, TiledLinearADC):
            parent, child = model.get_submodule(parent_name), attr
        else:
            continue
        tiled_layer = getattr(parent, child)
        if not isinstance(tiled_layer, TiledLinearADC):
            continue

        # Layer index filtering
        if layer_indices is not None:
            m = re.search(r'\.layers\.(\d+)\.', name)
            if m is None or int(m.group(1)) not in layer_indices:
                continue

        if mode == "residual":
            wrapper = ResidualLoRATiledLinearADC(
                tiled_layer, r=rank, alpha=lora_alpha, dtype=lora_dtype)
        elif mode == "pre_adc":
            wrapper = PreADCLoRATiledLinearADC(tiled_layer, r=rank, alpha=lora_alpha)
        else:
            raise ValueError(f"Unknown lora mode: {mode!r}. Choose 'residual' or 'pre_adc'.")
        setattr(parent, child, wrapper)
        count += 1

    layer_info = (f"layers={sorted(layer_indices)[:3]}..."
                  if layer_indices is not None else "all layers")
    if count == 0:
        raise ValueError(
            f"No TiledLinearADC found for LoRA targets {target_modules} in {layer_info}"
        )
    print(f"[ADC-LoRA] Applied {mode} LoRA (r={rank}, α={lora_alpha}, "
          f"dtype={str(lora_dtype).removeprefix('torch.')}) "
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
    microbatch_size: int | None = None,
    gradient_accumulation_steps: int = 1,
    teacher_device: "str | torch.device" = "cpu",
    teacher_model: nn.Module | None = None,
) -> nn.Module:
    """
    Fine-tune LoRA adapters.

    lora_loss="ce"     — standard LM cross-entropy
    lora_loss="ce_kl"  — CE + λ·KL(FP_teacher || student); requires teacher_name_or_path
                         or a preloaded teacher_model
    teacher_device     — where the teacher runs. On the CPU a 1B teacher costs tens of
                         seconds per microbatch; on the GPU it needs its fp16 weights
                         plus the logits of one microbatch.
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

    # Teacher for the KL loss
    teacher = None
    teacher_device = torch.device(teacher_device)
    if lora_loss == "ce_kl":
        if teacher_model is not None:
            teacher = teacher_model
        elif teacher_name_or_path is None:
            print("[ADC-LoRA] Warning: ce_kl requested but teacher_name_or_path=None → "
                  "falling back to ce")
            lora_loss = "ce"
        else:
            from transformers import AutoModelForCausalLM
            print(f"[ADC-LoRA] Loading FP teacher on {teacher_device}: {teacher_name_or_path}")
            teacher = AutoModelForCausalLM.from_pretrained(
                teacher_name_or_path,
                torch_dtype=torch.float16,
            )
        if teacher is not None:
            teacher = teacher.to(teacher_device).eval()
            for p in teacher.parameters():
                p.requires_grad_(False)

    if microbatch_size is not None and microbatch_size < 1:
        raise ValueError("microbatch_size must be at least 1")
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")

    # Collect calibration batches on CPU. Moving one microbatch at a time to
    # CUDA avoids retaining the entire LoRA calibration set on the GPU.
    samples = []
    for batch in dataloader:
        if len(samples) * cali_bsz >= nsamples:
            break
        input_ids = batch["input_ids"].cpu()
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.cpu()
        samples.append((input_ids, attention_mask))

    if not samples:
        raise ValueError("LoRA calibration dataloader produced no samples")

    physical_batch_size = samples[0][0].shape[0]
    if microbatch_size is not None and microbatch_size > physical_batch_size:
        raise ValueError(
            f"microbatch_size={microbatch_size} exceeds calibration loader batch "
            f"size {physical_batch_size}"
        )
    effective_microbatch_size = microbatch_size or physical_batch_size
    effective_batch_size = effective_microbatch_size * gradient_accumulation_steps
    print(f"[ADC-LoRA] Training {epochs} epochs on {len(samples)} batches  "
          f"lr={lora_lr}  loss={lora_loss}  "
          f"microbatch={effective_microbatch_size}  "
          f"grad_accum={gradient_accumulation_steps}  "
          f"effective_batch={effective_batch_size}")

    use_cache = getattr(model.config, "use_cache", None)
    if use_cache is not None:
        model.config.use_cache = False
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        microbatch_count = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_input_ids, batch_attention_mask in samples:
            for start in range(0, batch_input_ids.shape[0], effective_microbatch_size):
                end = start + effective_microbatch_size
                input_ids = batch_input_ids[start:end].to(device)
                attention_mask = (
                    batch_attention_mask[start:end].to(device)
                    if batch_attention_mask is not None
                    else None
                )
                labels = input_ids.clone()
                if attention_mask is not None:
                    labels[attention_mask == 0] = -100

                # The teacher runs first, so its activations are freed before
                # the student's graph is built.
                teacher_logits = None
                if lora_loss == "ce_kl" and teacher is not None:
                    with torch.no_grad():
                        t_out = teacher(
                            input_ids=batch_input_ids[start:end].to(teacher_device),
                            attention_mask=(
                                batch_attention_mask[start:end].to(teacher_device)
                                if batch_attention_mask is not None
                                else None
                            ),
                            use_cache=False,
                        )
                        teacher_logits = t_out.logits.to(device).float()
                        del t_out

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False,
                )
                ce_loss = outputs.loss

                if teacher_logits is not None:
                    student_logits = outputs.logits.float()
                    kl = F.kl_div(
                        F.log_softmax(student_logits / kl_temperature, dim=-1),
                        F.softmax(teacher_logits / kl_temperature, dim=-1),
                        reduction="batchmean",
                    ) * (kl_temperature ** 2)
                    loss = ce_loss + kl_weight * kl
                else:
                    loss = ce_loss

                total_loss += loss.detach().item()
                (loss / gradient_accumulation_steps).backward()
                microbatch_count += 1
                if microbatch_count % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                del input_ids, attention_mask, labels, outputs, ce_loss, loss
                if teacher_logits is not None:
                    del student_logits, kl
                del teacher_logits

        if microbatch_count % gradient_accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        print(f"[ADC-LoRA] Epoch {epoch + 1}/{epochs}  "
              f"avg loss={total_loss / max(microbatch_count, 1):.4f}")

    if teacher is not None:
        del teacher
        torch.cuda.empty_cache()

    if use_cache is not None:
        model.config.use_cache = use_cache
    model.eval()
    return model


# ============================================================
#  Saving and reloading adapters
# ============================================================

def adc_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """The lora_A / lora_B tensors of every adapter, keyed as in model.state_dict()."""
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if ".lora_A." in name or ".lora_B." in name
    }


def save_adc_lora(model: nn.Module, path: str, config: dict) -> int:
    """Save the adapters with the settings they were trained under.

    Returns the number of saved tensors. The frozen backbone is not saved;
    it is rebuilt by the same conversion and calibration on reload.
    """
    tensors = adc_lora_state_dict(model)
    if not tensors:
        raise ValueError("No ADC-LoRA adapters found in the model")
    torch.save({"config": dict(config), "state_dict": tensors}, path)
    return len(tensors)


def load_adc_lora(model: nn.Module, path: str) -> dict:
    """Load adapters saved by save_adc_lora into an identically wrapped model.

    The model must already carry adapters of the same targets, rank, mode and
    dtype; any difference raises. Returns the saved config.
    """
    payload = torch.load(path, map_location="cpu")
    saved = payload["state_dict"]
    expected = adc_lora_state_dict(model)
    missing = sorted(set(expected) - set(saved))
    unexpected = sorted(set(saved) - set(expected))
    if missing or unexpected:
        raise ValueError(
            f"Adapter keys in {path} do not match the model: {len(missing)} missing, "
            f"{len(unexpected)} unexpected, e.g. {(missing + unexpected)[:3]}"
        )
    for name, tensor in saved.items():
        if tensor.shape != expected[name].shape:
            raise ValueError(
                f"{name}: saved shape {tuple(tensor.shape)} != model shape "
                f"{tuple(expected[name].shape)}"
            )
        if tensor.dtype != expected[name].dtype:
            raise ValueError(
                f"{name}: saved dtype {tensor.dtype} != model dtype {expected[name].dtype}; "
                "use the same --lora_dtype as the saved run"
            )
    # strict=False only tolerates the absent backbone; every adapter key was
    # checked above.
    model.load_state_dict(saved, strict=False)
    return payload.get("config", {})
