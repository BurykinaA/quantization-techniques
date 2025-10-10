import argparse
import os
import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from transformers import AutoConfig, AutoTokenizer, BertForQuestionAnswering


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_qa_model_robust(checkpoint_dir: str) -> BertForQuestionAnswering:
    """
    Load BertForQuestionAnswering from a checkpoint that may contain extra
    keys from QAT layers. Falls back to strict=False state_dict load.
    """
    try:
        return BertForQuestionAnswering.from_pretrained(checkpoint_dir)
    except Exception as e:
        logger.warning(f"Standard from_pretrained failed, retrying with strict=False. Error: {e}")
        state_path = os.path.join(checkpoint_dir, 'pytorch_model.bin')
        if not os.path.exists(state_path):
            raise FileNotFoundError(f"No model weights found in {checkpoint_dir}")
        state = torch.load(state_path, map_location='cpu')
        config = AutoConfig.from_pretrained(checkpoint_dir)
        model = BertForQuestionAnswering(config)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            logger.info(f"Ignored {len(unexpected)} unexpected keys (likely QAT quantizer params).")
        if missing:
            logger.info(f"Missing keys count: {len(missing)} (randomly initialized).")
        return model


def collect_linear_weights(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Return a mapping from module name to its weight tensor for all Linear-like layers."""
    weights: Dict[str, torch.Tensor] = {}

    # QATLinear may exist; import lazily to avoid hard dependency
    try:
        from qat_layers import QATLinear  # type: ignore
        qat_cls = QATLinear
    except Exception:
        qat_cls = tuple()  # type: ignore

    for name, module in model.named_modules():
        is_linear = isinstance(module, nn.Linear)
        is_qat_linear = isinstance(module, qat_cls) if qat_cls else False
        if is_linear or is_qat_linear:
            w = getattr(module, 'weight', None)
            if isinstance(w, torch.Tensor):
                weights[name] = w.detach().cpu().view(-1)

    return weights


def plot_histograms(weights: Dict[str, torch.Tensor], out_dir: str, bins: int = 100) -> None:
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, 'summary.txt')
    with open(summary_path, 'w') as fsum:
        for name, w_flat in weights.items():
            if w_flat.numel() == 0:
                continue
            plt.figure(figsize=(6, 4))
            plt.hist(w_flat.numpy(), bins=bins, density=False, color='#1f77b4', edgecolor='black', linewidth=0.2)
            plt.title(name)
            plt.xlabel('Weight value')
            plt.ylabel('Count')
            safe_name = name.replace('.', '_')
            out_path = os.path.join(out_dir, f"{safe_name}.png")
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()

            mean = float(w_flat.mean())
            std = float(w_flat.std())
            min_v = float(w_flat.min())
            max_v = float(w_flat.max())
            fsum.write(f"{name}: numel={w_flat.numel()} mean={mean:.6f} std={std:.6f} min={min_v:.6f} max={max_v:.6f}\n")
            logger.info(f"Saved histogram for {name} -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot per-layer weight histograms from a QAT/fine-tuned checkpoint")
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Path to model output dir or specific checkpoint-* dir')
    parser.add_argument('--out_dir', type=str, default='./qat_weight_hists', help='Directory to save histograms')
    parser.add_argument('--bins', type=int, default=100, help='Number of bins in histograms')
    args = parser.parse_args()

    logger.info(f"Loading model from {args.checkpoint_dir}")
    model = load_qa_model_robust(args.checkpoint_dir)

    logger.info("Collecting weights from linear/QATLinear layers")
    weights = collect_linear_weights(model)
    logger.info(f"Found {len(weights)} layers with weights")

    logger.info(f"Saving histograms to {args.out_dir}")
    plot_histograms(weights, args.out_dir, bins=args.bins)
    logger.info("Done.")


if __name__ == '__main__':
    main()


