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


def load_state_dict_maybe_sharded(checkpoint_dir: str) -> Dict[str, torch.Tensor]:
    """Load state dict from pytorch_model.bin or model.safetensors (single-file)."""
    bin_path = os.path.join(checkpoint_dir, 'pytorch_model.bin')
    safe_path = os.path.join(checkpoint_dir, 'model.safetensors')
    if os.path.exists(bin_path):
        return torch.load(bin_path, map_location='cpu')
    if os.path.exists(safe_path):
        try:
            from safetensors.torch import load_file as safe_load_file  # type: ignore
        except Exception as e:
            raise RuntimeError(f"safetensors present but not loadable: {e}")
        return safe_load_file(safe_path)
    raise FileNotFoundError(f"No model weights found in {checkpoint_dir}")


def build_qat_model_from_checkpoint(checkpoint_dir: str, weight_bits: int, activation_bits: int,
                                    exclude_patterns=None) -> BertForQuestionAnswering:
    """Rebuild a model with QATLinear wrappers and load state dict with strict=False."""
    if exclude_patterns is None:
        exclude_patterns = ["embeddings", "pooler", "qa_outputs"]

    # Import converter locally to avoid circulars if script is moved
    from bert_qat_integration import BertQATConverter  # type: ignore

    config = AutoConfig.from_pretrained(checkpoint_dir)
    base = BertForQuestionAnswering(config)
    model_qat = BertQATConverter.replace_linear_with_qat(
        base,
        weight_bits=weight_bits,
        activation_bits=activation_bits,
        exclude_patterns=exclude_patterns,
    )

    state = load_state_dict_maybe_sharded(checkpoint_dir)
    missing, unexpected = model_qat.load_state_dict(state, strict=False)
    if unexpected:
        logger.info(f"Ignored {len(unexpected)} unexpected keys when loading QAT model (ok)")
    if missing:
        logger.info(f"Missing {len(missing)} keys when loading QAT model (random init for those)")
    return model_qat


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


def collect_qat_linear_modules(model: nn.Module) -> Dict[str, nn.Module]:
    """Return mapping from module name to QATLinear module (if available)."""
    modules: Dict[str, nn.Module] = {}
    try:
        from qat_layers import QATLinear  # type: ignore
    except Exception:
        return modules

    for name, module in model.named_modules():
        if isinstance(module, QATLinear):
            modules[name] = module
    return modules


def plot_qat_int_code_histograms(qat_modules: Dict[str, nn.Module], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, 'summary_int_codes.txt')
    with open(summary_path, 'w') as fsum:
        for name, m in qat_modules.items():
            # Expect per-channel symmetric weight quantizer by default
            w: torch.Tensor = m.weight.detach().cpu()  # [out_features, in_features]
            q = m.weight_quantizer
            qmin, qmax = int(q.qmin), int(q.qmax)
            scale: torch.Tensor = q.scale.detach().cpu().view(-1, 1)  # [out_features, 1]
            if getattr(q, 'symmetric', True):
                q_codes = torch.round(w / scale)
            else:
                zero_point: torch.Tensor = q.zero_point.detach().cpu().view(-1, 1)
                q_codes = torch.round(w / scale + zero_point)
            q_codes = torch.clamp(q_codes, qmin, qmax).to(torch.int32)

            # Histogram over all integer levels present
            codes_flat = q_codes.view(-1).to(torch.int32)
            # Use one bin per integer level
            bins = int(qmax - qmin + 1)
            plt.figure(figsize=(6, 4))
            plt.hist(codes_flat.numpy(), bins=bins, range=(qmin - 0.5, qmax + 0.5), color='#ff7f0e', edgecolor='black', linewidth=0.2)
            plt.title(f"{name} (int codes)")
            plt.xlabel('Integer code')
            plt.ylabel('Count')
            safe_name = name.replace('.', '_')
            out_path = os.path.join(out_dir, f"{safe_name}_intcodes.png")
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()

            unique_codes = torch.unique(codes_flat, sorted=True)
            fsum.write(f"{name}: numel={codes_flat.numel()} unique_levels={len(unique_codes)} range=[{int(unique_codes.min())},{int(unique_codes.max())}]\n")
            logger.info(f"Saved integer-code histogram for {name} -> {out_path}")


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
    parser.add_argument('--bins', type=int, default=100, help='Number of bins in histograms (float/dequant modes)')
    parser.add_argument('--plot_int_codes', action='store_true', help='Plot integer code histograms for QATLinear weights')
    parser.add_argument('--weight_bits', type=int, default=8, help='Weight bits for QAT reconstruction (if needed)')
    parser.add_argument('--activation_bits', type=int, default=8, help='Activation bits for QAT reconstruction (if needed)')
    args = parser.parse_args()

    logger.info(f"Loading model from {args.checkpoint_dir}")
    model = load_qa_model_robust(args.checkpoint_dir)

    if args.plot_int_codes:
        logger.info("Collecting QATLinear modules for integer-code histograms")
        qat_modules = collect_qat_linear_modules(model)
        if not qat_modules:
            # Auto-detect: if checkpoint contains quantizer keys, rebuild QAT model and reload
            try:
                state = load_state_dict_maybe_sharded(args.checkpoint_dir)
                has_qat_keys = any('quantizer' in k for k in state.keys())
            except Exception:
                has_qat_keys = False

            if has_qat_keys:
                logger.info("No QATLinear modules in loaded model; reconstructing QAT architecture and reloading weights...")
                try:
                    model = build_qat_model_from_checkpoint(
                        args.checkpoint_dir,
                        weight_bits=args.weight_bits,
                        activation_bits=args.activation_bits,
                    )
                    qat_modules = collect_qat_linear_modules(model)
                except Exception as e:
                    logger.warning(f"Failed to reconstruct QAT model: {e}")

        if not qat_modules:
            logger.warning("No QATLinear modules found; cannot plot integer codes. Did you run a QAT model?")
        else:
            logger.info(f"Found {len(qat_modules)} QATLinear modules")
            logger.info(f"Saving integer-code histograms to {args.out_dir}")
            plot_qat_int_code_histograms(qat_modules, args.out_dir)
    else:
        logger.info("Collecting weights from linear/QATLinear layers (float weights)")
        weights = collect_linear_weights(model)
        logger.info(f"Found {len(weights)} layers with weights")
        logger.info(f"Saving float histograms to {args.out_dir}")
        plot_histograms(weights, args.out_dir, bins=args.bins)
    logger.info("Done.")


if __name__ == '__main__':
    main()


