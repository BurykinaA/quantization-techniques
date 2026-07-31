"""
Post-ADC LoRA calibration for quantized ViTs.

The LoRA architecture (ResidualLoRATiledLinearADC) and injection
(apply_adc_lora) are reused unchanged from ``core.adc_lora`` — they match
``.linear`` inside FlatQuantLinear by module-name suffix, so passing
target_modules=("qkv","proj","fc1","fc2") wires LoRA into every ViT projection.

Only the training loop is ViT-specific: the teacher is a timm FP ViT and the
loss is image cross-entropy on ImageNet labels plus a KL term against the FP
teacher's logits (matching the thesis LoRA recipe: λ=0.5, T=2.0).
"""

import logging

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def calibrate_adc_lora_vit(
    model, dataloader, device,
    nsamples=1024, epochs=5, lora_lr=1e-4,
    lora_loss="ce_kl", teacher_model_name=None,
    kl_weight=0.5, kl_temperature=2.0,
):
    """Fine-tune the LoRA adapters on calibration images.

    lora_loss="ce"     — cross-entropy on ImageNet labels only.
    lora_loss="ce_kl"  — CE + λ·KL(student ‖ FP-teacher); needs teacher_model_name.
    """
    # Freeze everything except LoRA adapters
    for n, p in model.named_parameters():
        is_lora = ("lora_A.weight" in n or "lora_B.weight" in n
                   or ("lora_A" in n and "weight" not in n)
                   or ("lora_B" in n and "weight" not in n))
        p.requires_grad_(is_lora)

    lora_params = [p for p in model.parameters() if p.requires_grad]
    if not lora_params:
        logger.warning("[ViT-LoRA] no LoRA parameters found — skipping calibration")
        return model
    n_params = sum(p.numel() for p in lora_params)
    logger.info(f"[ViT-LoRA] Trainable params: {n_params:,} across {len(lora_params)} tensors")

    optimizer = torch.optim.AdamW(lora_params, lr=lora_lr)

    teacher = None
    if lora_loss == "ce_kl":
        if teacher_model_name is None:
            logger.warning("[ViT-LoRA] ce_kl requested but no teacher — falling back to ce")
            lora_loss = "ce"
        else:
            logger.info(f"[ViT-LoRA] Loading FP teacher: {teacher_model_name}")
            teacher = timm.create_model(teacher_model_name, pretrained=True).to(device).eval()
            for p in teacher.parameters():
                p.requires_grad_(False)

    # Collect a fixed set of calibration batches on CPU (moved to the GPU one
    # batch at a time in the training loop, so the whole set is never resident
    # on the device).  The dataloader's batch size defines the LoRA batch size.
    samples = []
    collected = 0
    for imgs, labels in dataloader:
        if collected >= nsamples:
            break
        samples.append((imgs, labels))
        collected += imgs.shape[0]

    ce = nn.CrossEntropyLoss()
    logger.info(f"[ViT-LoRA] Training {epochs} epochs on {len(samples)} batches "
                f"({collected} images) lr={lora_lr} loss={lora_loss}")

    model.train()
    for epoch in range(epochs):
        total = 0.0
        for imgs, labels in samples:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            loss = ce(logits, labels)
            if teacher is not None:
                with torch.no_grad():
                    t_logits = teacher(imgs).float()
                s_logits = logits.float()
                kl = F.kl_div(
                    F.log_softmax(s_logits / kl_temperature, dim=-1),
                    F.softmax(t_logits / kl_temperature, dim=-1),
                    reduction="batchmean",
                ) * (kl_temperature ** 2)
                loss = loss + kl_weight * kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        logger.info(f"[ViT-LoRA] epoch {epoch + 1}/{epochs}  avg loss={total / len(samples):.4f}")

    if teacher is not None:
        del teacher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model.eval()
    return model
