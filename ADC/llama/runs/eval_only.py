#!/usr/bin/env python
"""
eval_only.py — score a previously calibrated FlatQuant + ADC checkpoint on
additional lm-evaluation-harness tasks WITHOUT re-running calibration.

Use case
--------
The main `llama_smooth_quant_adc_ptq.py` saves a HuggingFace state_dict at
the end of every run (via `model.save_pretrained(args.output_dir)`).  That
state_dict contains:

    * Reparameterized (post-fold) linear weights inside TiledLinearADC tiles
    * LWC clip factors (`clip_factor_w_max/min`)  in each FlatQuantLinear
    * PACT raw_alpha_adc                          in each FlatQuantLinear
    * LoRA `lora_A` / `lora_B`                    if --lora_rank > 0

The KroneckerTransform parameters (`u_left`, `v_left`, `diag_*`) are GONE
by the time of save — they were collapsed into the linear weights during
`reparameterize_model()` and the SVD parameters are deleted in
`KroneckerTransform.to_eval_mode()`.

So to *load* the checkpoint and re-score it we just have to recreate the
SAME wrapper stack with random-init transforms, immediately call
`reparameterize_model()` (which clears the transforms exactly as at save
time), apply the same ADC conversion (and LoRA if applicable), and then
`load_state_dict(strict=False)`.  Random transforms are overwritten by
the loaded weights; what gets preserved through `reparameterize_model()`
matches the saved keys.

Pipeline
--------
1. Load FP base model from --model_name
2. apply_flatquant_to_model(...)                      with same fq config
3. reparameterize_model(model)                        identical to save-time
4. LlamaADCConverter.replace_linear_with_adc(...)     with same ADC config
5. propagate_alpha_adc_to_tiled(model)                if --pact_inference
6. apply_adc_lora(...)                                if --lora_rank > 0
7. model.load_state_dict(saved_state, strict=False)   log missing/unexpected
8. _run_lm_eval (ADC)
9. (optional) flip bypass_adc and _run_lm_eval again
10. Merge new keys into results.json under matching --run_name

If a smoke test of step 7 (--dry_run_state_dict_check) shows >10 missing
or unexpected keys other than rotary inv_freq buffers, fall back to a
full recalibration via `run_paper_comparison.sh`.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, LlamaForCausalLM

# Make project root importable so we can re-use the existing helpers.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ADC.llama.core.adc_layers import TiledLinearADC
from ADC.llama.core.adc_lora import apply_adc_lora
from ADC.llama.core.flat_quant import (
    apply_flatquant_to_model,
    propagate_alpha_adc_to_tiled,
    reparameterize_model as fq_reparameterize_model,
)
# LlamaADCConverter is defined in the main calibration script, not in core.
# Top-level of llama_smooth_quant_adc_ptq.py is import-safe (no side effects;
# main() is guarded by `if __name__ == "__main__"`).
from ADC.llama.runs.llama_smooth_quant_adc_ptq import LlamaADCConverter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("eval_only")


# -----------------------------------------------------------------------------
# lm-eval helper (copied verbatim from llama_smooth_quant_adc_ptq.py:328-377
# to avoid importing the entire 3000-line training script and its argparse).
# -----------------------------------------------------------------------------

def _run_lm_eval(model, tokenizer, args) -> dict:
    """Run lm-evaluation-harness tasks; return dict of {lm_<task>: acc%}.  Never raises."""
    results: dict[str, float] = {}
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        logger.warning("[lm-eval] lm_eval not installed. Run: pip install lm-eval>=0.4.0")
        return results

    def _get_acc(r: dict):
        for key in ("acc_norm,none", "acc_norm", "acc,none", "acc"):
            if key in r and r[key] is not None:
                return float(r[key])
        return None

    try:
        lm_wrapper = HFLM(
            pretrained=model, tokenizer=tokenizer,
            batch_size=getattr(args, "lm_eval_batch_size", 4),
        )
        tasks = list(args.lm_eval_tasks)
        zero_shot_tasks = [t for t in tasks if t in ("hellaswag", "piqa", "arc_easy", "arc_challenge",
                                                      "openbookqa", "boolq")]
        five_shot_tasks = [t for t in tasks if t not in zero_shot_tasks]

        task_groups = []
        if zero_shot_tasks:
            task_groups.append((zero_shot_tasks, 0))
        if five_shot_tasks:
            task_groups.append((five_shot_tasks, 5))

        for task_list, nfewshot in task_groups:
            logger.info(f"[lm-eval] Running {task_list} ({nfewshot}-shot)")
            eval_out = lm_eval.simple_evaluate(
                model=lm_wrapper, tasks=task_list, num_fewshot=nfewshot,
            )
            for task, r in eval_out["results"].items():
                acc = _get_acc(r)
                if acc is not None:
                    results[f"lm_{task}"] = round(acc * 100, 2)
                    logger.info(f"[lm-eval] {task}: {acc:.4f}  ({acc * 100:.2f}%)")
    except Exception as e:
        logger.warning(f"[lm-eval] FAILED: {e}")
    return results


# -----------------------------------------------------------------------------
# Wrapper-stack reconstruction
# -----------------------------------------------------------------------------

def build_wrapped_model(args, device: torch.device) -> tuple[torch.nn.Module, Any]:
    """Reproduce the wrapping pipeline used by llama_smooth_quant_adc_ptq.py.

    Returns the wrapped model (on `device`) plus the tokenizer (loaded from
    --checkpoint_dir so we get the exact tokenizer files saved at run time).
    """
    logger.info(f"Loading base model: {args.model_name}")
    model = LlamaForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)

    # 1. FlatQuant wrappers (random-init transforms; will be cleared by reparam)
    fq_signed = not args.ashift
    fq_adc_config = {
        "bx": args.bx, "bw": args.bw, "ba": args.ba, "k": args.k,
        "mvm_limit": args.mvm_limit, "signed_activations": fq_signed,
    }
    logger.info(
        f"[build] FlatQuant wrappers  w{args.fq_w_bits}a{args.fq_a_bits}  "
        f"lwc={args.fq_lwc} lac={args.fq_lac} add_diag={args.fq_add_diag}"
    )
    model = apply_flatquant_to_model(
        model,
        w_bits=args.fq_w_bits, a_bits=args.fq_a_bits,
        add_diag=args.fq_add_diag, lwc=args.fq_lwc, lac=args.fq_lac,
        adc_config=fq_adc_config,
    )

    # 2. Reparameterize — clears KroneckerTransform.u/v/diag_* exactly as at save time
    logger.info("[build] Reparameterizing (folds transforms into weights)")
    model = fq_reparameterize_model(model)

    # 3. ADC conversion — replaces inner nn.Linear with TiledLinearADC
    logger.info(
        f"[build] ADC conversion  bx={args.bx} bw={args.bw} ba={args.ba} "
        f"k={args.k} mvm_limit={args.mvm_limit}"
    )
    model = LlamaADCConverter.replace_linear_with_adc(
        model,
        bx=args.bx, bw=args.bw, ba=args.ba, k=args.k,
        ashift=args.ashift,
        signed_activations=fq_signed,
        exclude_patterns=["embed_tokens", "lm_head", "_orig_attn"],
        mvm_limit=args.mvm_limit,
        use_kurtosis_loss=False,
        kurtosis_weight=0.0,
        target_kurtosis=1.8,
    )

    model = model.to(device)

    # 4. PACT alpha propagation (only if it was used at calibration time).
    if args.pact_inference:
        logger.info("[build] Propagating PACT alpha → TiledLinearADC tiles")
        propagate_alpha_adc_to_tiled(model)

    # 5. LoRA wrap (only if rank > 0)
    if args.lora_rank > 0:
        logger.info(
            f"[build] Applying ADC-LoRA  rank={args.lora_rank} "
            f"α={args.lora_alpha}  targets={args.lora_target_modules}"
        )
        model = apply_adc_lora(
            model,
            target_modules=args.lora_target_modules,
            rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            mode=args.lora_mode,
            layer_indices=None,
        )

    return model, tokenizer


def load_checkpoint(model: torch.nn.Module, checkpoint_dir: str) -> tuple[list[str], list[str]]:
    """Load saved HF state_dict into the wrapped model.

    Uses strict=False (rotary inv_freq buffers ok) and assign=True so that
    parameters with shape mismatch (e.g. LearnableQuantizer.scale: [1]
    uninitialized vs [out_features] calibrated) are replaced with the
    checkpoint tensor instead of failing.  Returns (missing, unexpected)."""
    safetensors_path = os.path.join(checkpoint_dir, "model.safetensors")
    bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")

    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        logger.info(f"[load] Loading state_dict from {safetensors_path}")
        state_dict = load_file(safetensors_path)
    elif os.path.exists(bin_path):
        logger.info(f"[load] Loading state_dict from {bin_path}")
        state_dict = torch.load(bin_path, map_location="cpu")
    else:
        # Sharded checkpoint?
        index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            from safetensors.torch import load_file
            with open(index_path) as f:
                idx = json.load(f)
            shards = sorted(set(idx["weight_map"].values()))
            logger.info(f"[load] Loading {len(shards)} safetensors shards")
            state_dict = {}
            for shard in shards:
                state_dict.update(load_file(os.path.join(checkpoint_dir, shard)))
        else:
            raise FileNotFoundError(
                f"No model.safetensors / pytorch_model.bin / index found in {checkpoint_dir}"
            )

    # assign=True (PyTorch ≥ 2.1) bypasses in-place copy and assigns the
    # checkpoint tensor directly to each parameter. This lets us load tensors
    # whose shapes don't match the freshly-constructed module (typical for
    # LearnableQuantizer.scale which starts as [1] and is resized to
    # [out_features] only on calibration).
    try:
        missing, unexpected = model.load_state_dict(
            state_dict, strict=False, assign=True
        )
    except TypeError:
        # Older PyTorch without `assign` kwarg — fall back to manual shape resize.
        logger.warning("[load] torch.load_state_dict has no `assign` kwarg; "
                       "falling back to manual shape-tolerant load.")
        missing, unexpected = _manual_shape_tolerant_load(model, state_dict)
    return list(missing), list(unexpected)


def _manual_shape_tolerant_load(model: torch.nn.Module, state_dict: dict) -> tuple[list[str], list[str]]:
    """Fallback for PyTorch < 2.1 that doesn't support assign=True."""
    own = dict(model.state_dict())
    missing: list[str] = []
    unexpected: list[str] = []
    with torch.no_grad():
        for name, src in state_dict.items():
            if name not in own:
                unexpected.append(name)
                continue
            dst = own[name]
            if dst.shape != src.shape:
                # Replace parameter data with the source tensor (shape change).
                # Find the parent module and reassign the parameter/buffer.
                *parents, attr = name.split(".")
                parent_mod = model
                for p in parents:
                    parent_mod = getattr(parent_mod, p)
                target = getattr(parent_mod, attr)
                if isinstance(target, torch.nn.Parameter):
                    setattr(parent_mod, attr,
                            torch.nn.Parameter(src.clone().to(target.device),
                                                requires_grad=target.requires_grad))
                else:
                    setattr(parent_mod, attr, src.clone().to(target.device))
            else:
                dst.copy_(src)
        for name in own:
            if name not in state_dict:
                missing.append(name)
    return missing, unexpected


# -----------------------------------------------------------------------------
# Bypass-mode toggling
# -----------------------------------------------------------------------------

def _set_bypass_adc(model: torch.nn.Module, bypass: bool) -> None:
    for _, m in model.named_modules():
        if isinstance(m, TiledLinearADC):
            m.set_bypass_adc(bypass)


# -----------------------------------------------------------------------------
# Results merge
# -----------------------------------------------------------------------------

def merge_into_results_json(results_json_path: str, run_name: str,
                            new_metrics: dict, checkpoint_dir: str | None = None,
                            mode: str = "checkpoint") -> None:
    """Update the entry matching run_name in-place; if not found, append.

    Records `checkpoint_dir` used to produce `new_metrics` under
    `eval_only_history` so multiple eval_only passes can be traced.
    """
    existing: list[dict] = []
    if os.path.exists(results_json_path):
        try:
            with open(results_json_path) as f:
                existing = json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"[merge] {results_json_path} is corrupt, starting fresh list")
            existing = []

    history_entry = {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,                                          # "checkpoint" / "fp_only"
        "checkpoint_dir": (os.path.abspath(checkpoint_dir)
                           if checkpoint_dir else None),
        "added_keys": sorted(new_metrics.keys()),
    }

    found = False
    for entry in existing:
        if entry.get("run_name") == run_name:
            entry.setdefault("results", {}).update(new_metrics)
            entry["timestamp_eval_only"] = history_entry["timestamp"]
            entry.setdefault("eval_only_history", []).append(history_entry)
            found = True
            break
    if not found:
        logger.warning(f"[merge] run_name {run_name!r} not in JSON — appending new entry")
        existing.append({
            "run_name": run_name,
            "timestamp": history_entry["timestamp"],
            "status": "eval_only",
            "output_dir": history_entry["checkpoint_dir"],
            "results": new_metrics,
            "eval_only_history": [history_entry],
        })

    os.makedirs(os.path.dirname(os.path.abspath(results_json_path)), exist_ok=True)
    with open(results_json_path, "w") as f:
        json.dump(existing, f, indent=2)
    logger.info(
        f"[merge] Wrote {len(new_metrics)} new keys to {results_json_path}"
        f" (checkpoint_dir={checkpoint_dir or '<fp_only>'})"
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Identity / paths
    p.add_argument("--checkpoint_dir", default=None,
                   help="Directory with model.safetensors / pytorch_model.bin saved by "
                        "save_pretrained(). Required UNLESS --fp_only is set.")
    p.add_argument("--model_name", required=True,
                   help="HuggingFace name of the BASE FP model (e.g. meta-llama/Llama-3.2-1B)")
    p.add_argument("--run_name", required=True,
                   help="Run identifier — must match the run_name in --results_json_path")
    p.add_argument("--results_json_path", required=True,
                   help="Path to results.json — new lm-eval keys are merged into the matching entry")
    p.add_argument("--fp_only", action="store_true",
                   help="Skip all FlatQuant/ADC/LoRA wrapping and load the plain FP16 model "
                        "directly from --model_name. Used to add new lm-eval tasks to the FP16 "
                        "entry. --run_lm_eval_bypass is ignored in this mode (no ADC to bypass).")

    # FlatQuant wrapper config (must match what was used at calibration time;
    # only required for quantized eval-only — ignored when --fp_only).
    p.add_argument("--fq_w_bits", type=int, default=8)
    p.add_argument("--fq_a_bits", type=int, default=8)
    p.add_argument("--fq_add_diag", action="store_true")
    p.add_argument("--fq_lwc",      action="store_true")
    p.add_argument("--fq_lac",      action="store_true")

    # ADC hardware config (ignored when --fp_only)
    p.add_argument("--bx", type=int, default=8)
    p.add_argument("--bw", type=int, default=8)
    p.add_argument("--ba", type=int, default=8)
    p.add_argument("--k",  type=int, default=4)
    p.add_argument("--mvm_limit", type=int, default=256)
    p.add_argument("--ashift", action="store_true")
    p.add_argument("--pact_inference", action="store_true",
                   help="Match the original run's --pact_inference flag (rare, default off)")

    # LoRA — must match calibration if it was used
    p.add_argument("--lora_rank", type=int, default=0)
    p.add_argument("--lora_alpha", type=float, default=8.0)
    p.add_argument("--lora_mode", choices=["residual", "pre_adc"], default="residual")
    p.add_argument("--lora_target_modules", nargs="+",
                   default=["down_proj", "up_proj", "gate_proj",
                            "q_proj", "k_proj", "v_proj", "o_proj"])

    # lm-eval
    p.add_argument("--run_lm_eval", action="store_true", default=True,
                   help="(always true here — kept for arg-shape compatibility with _run_lm_eval)")
    p.add_argument("--run_lm_eval_bypass", action="store_true",
                   help="Also score every task with ADC disabled and save under bypass_lm_<task>")
    p.add_argument("--lm_eval_tasks", nargs="+",
                   default=["arc_easy", "arc_challenge", "piqa", "openbookqa", "boolq"])
    p.add_argument("--lm_eval_batch_size", type=int, default=4)

    # Debug
    p.add_argument("--dry_run_state_dict_check", action="store_true",
                   help="Build wrappers, attempt load_state_dict(strict=False), log missing/unexpected, and exit")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return p


def _load_fp_only(args, device: torch.device) -> tuple[torch.nn.Module, Any]:
    """Load plain FP16 base model — no FlatQuant/ADC/LoRA wrapping."""
    logger.info(f"[fp_only] Loading base FP model: {args.model_name}")
    model = LlamaForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    return model, tokenizer


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)

    logger.info(f"[eval_only] run_name={args.run_name}  device={device}")

    if args.fp_only:
        # FP-only branch: just load the base model and run lm-eval — no
        # wrappers, no state_dict load, no bypass mode.
        logger.info("[eval_only] --fp_only set: skipping all FlatQuant/ADC/LoRA wrapping")
        model, tokenizer = _load_fp_only(args, device)
    else:
        if not args.checkpoint_dir:
            raise SystemExit("--checkpoint_dir is required unless --fp_only is set")
        logger.info(f"[eval_only] checkpoint_dir={args.checkpoint_dir}")

        # 1-6: rebuild the wrapped model
        model, tokenizer = build_wrapped_model(args, device)

        # 7: load_state_dict
        missing, unexpected = load_checkpoint(model, args.checkpoint_dir)
        logger.info(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
        for k in missing[:10]:
            logger.warning(f"   missing: {k}")
        if len(missing) > 10:
            logger.warning(f"   ... and {len(missing) - 10} more missing")
        for k in unexpected[:10]:
            logger.warning(f"   unexpected: {k}")
        if len(unexpected) > 10:
            logger.warning(f"   ... and {len(unexpected) - 10} more unexpected")

        # Heuristic: rotary `inv_freq` buffers and similar should be the only
        # tolerable misses.  Anything large is a problem.
        intolerable_missing = [k for k in missing if "inv_freq" not in k]
        if intolerable_missing and not args.dry_run_state_dict_check:
            logger.warning(
                f"[load] WARNING: {len(intolerable_missing)} missing keys are not "
                f"rotary inv_freq buffers. Results may be wrong. First 5: "
                f"{intolerable_missing[:5]}"
            )

        if args.dry_run_state_dict_check:
            logger.info("[eval_only] --dry_run_state_dict_check set, exiting before lm-eval")
            return

    # 8: lm-eval (ADC mode)
    model.eval()
    logger.info("[eval_only] Running lm-eval (ADC active)")
    adc_results = _run_lm_eval(model, tokenizer, args)
    logger.info(f"[eval_only] ADC lm-eval done, got {len(adc_results)} keys")

    # 9: lm-eval (bypass mode), if requested. Skipped for --fp_only (no ADC).
    if args.run_lm_eval_bypass and not args.fp_only:
        logger.info("[eval_only] Running lm-eval (bypass — ADC disabled)")
        _set_bypass_adc(model, True)
        try:
            bypass_results = _run_lm_eval(model, tokenizer, args)
        finally:
            _set_bypass_adc(model, False)
        for k, v in bypass_results.items():
            adc_results[f"bypass_{k}"] = v
        logger.info(f"[eval_only] Bypass lm-eval done, got {len(bypass_results)} keys")
    elif args.run_lm_eval_bypass and args.fp_only:
        logger.info("[eval_only] --run_lm_eval_bypass ignored under --fp_only (no ADC)")

    if not adc_results:
        logger.warning("[eval_only] No lm-eval results produced — nothing to merge")
        return

    # 10: merge into results.json — record which checkpoint produced these keys
    merge_into_results_json(
        args.results_json_path, args.run_name, adc_results,
        checkpoint_dir=(args.checkpoint_dir if not args.fp_only else None),
        mode=("fp_only" if args.fp_only else "checkpoint"),
    )


if __name__ == "__main__":
    main()
