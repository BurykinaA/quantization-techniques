"""
Evaluation utilities for ViT ImageNet quantization: top-1/top-5 validation
and forward-pass latency.

Ported from vit_adc/validate_utils.py (validate, accuracy).  Self-contained;
works with any model that maps a (B, 3, H, W) image batch to (B, num_classes)
logits — the FP timm model, the FlatQuant-wrapped model, and the ADC model all
satisfy this.
"""

import time
import logging

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class AverageMeter:
    """Tracks a running average."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def accuracy(output, target, topk=(1,)):
    """Compute top-k accuracy (percent) for the given ks."""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


@torch.no_grad()
def validate(val_loader, model, criterion=None, device="cuda", print_freq=20,
             desc="val"):
    """Run the model over ``val_loader`` and return (loss, top1, top5).

    All averages are image-weighted so a strided/partial loader is handled
    correctly.
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
    model.eval()

    n_batches = len(val_loader)
    for i, (data, target) in enumerate(tqdm(val_loader, desc=desc, total=n_batches)):
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = criterion(output, target)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), data.size(0))
        top1.update(prec1.item(), data.size(0))
        top5.update(prec5.item(), data.size(0))

    logger.info(f"  {desc}: top1={top1.avg:.2f}  top5={top5.avg:.2f}  "
                f"loss={losses.avg:.4f}  (n={int(top1.count)})")
    return losses.avg, top1.avg, top5.avg


@torch.no_grad()
def measure_latency(model, device, img_size=224, batch_size=1,
                    n_warmup=5, n_runs=20):
    """Forward-pass latency for a (batch_size, 3, img_size, img_size) input."""
    model.eval()
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    use_cuda = torch.device(device).type == "cuda"

    for _ in range(n_warmup):
        model(x)
    if use_cuda:
        torch.cuda.synchronize(device)

    latencies_ms = []
    for _ in range(n_runs):
        if use_cuda:
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
            model(x)
            t1.record()
            torch.cuda.synchronize(device)
            latencies_ms.append(t0.elapsed_time(t1))
        else:
            s = time.perf_counter()
            model(x)
            latencies_ms.append((time.perf_counter() - s) * 1000.0)

    arr = np.array(latencies_ms)
    return {
        "latency_mean_ms": float(arr.mean()),
        "latency_p95_ms": float(np.percentile(arr, 95)),
        "throughput_img_s": float(batch_size / (arr.mean() / 1000.0)),
        "batch_size": batch_size,
    }
