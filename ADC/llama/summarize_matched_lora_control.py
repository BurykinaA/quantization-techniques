#!/usr/bin/env python3
"""Perplexity before and after the no-ADC LoRA residual on its backbone.

Reads the per-run JSON files written by run_matched_lora_control.sh, from one
or more result directories (e.g. the bfloat16 and float16 runs). Models whose
run has not finished yet are shown as "—".
"""

import argparse
import json
from pathlib import Path


MODELS = [
    ("llama32_1b", "Llama-3.2-1B"),
    ("smollm2_17b", "SmolLM2-1.7B"),
    ("tinyllama_11b", "TinyLlama-1.1B"),
]
DATASETS = ("wikitext2", "c4")
BACKBONES = ("flatquant", "adcquant")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, nargs="+", required=True,
                        help="Directories with the per-run JSON files")
    parser.add_argument("--run-suffix", default="",
                        help="Run-name suffix, e.g. _smoke for smoke runs")
    parser.add_argument("--markdown-output", type=Path, default=None)
    return parser.parse_args()


def control_runs(results_dirs: list[Path], suffix: str) -> list[tuple[str, str, str, dict]]:
    """(model key, backbone, dtype, results) for every finished control run."""
    run_names = {
        f"{key}_{backbone}_noadc_lora{suffix}": (key, backbone)
        for key, _ in MODELS
        for backbone in BACKBONES
    }
    runs = []
    for results_dir in results_dirs:
        for path in sorted(results_dir.glob("*.json")):
            with path.open(encoding="utf-8") as handle:
                for record in json.load(handle):
                    match = run_names.get(record.get("run_name"))
                    if match is None or record.get("status") != "success":
                        continue
                    dtype = record.get("config", {}).get("torch_dtype", "?")
                    runs.append((*match, dtype, record.get("results", {})))
    return runs


def fmt_change(before: float | None, after: float | None) -> str:
    if before is None or after is None:
        return "—"
    change = 100.0 * (after - before) / before
    return f"{before:.2f} → {after:.2f} ({change:+.1f}%)"


def build_report(runs: list[tuple[str, str, str, dict]]) -> str:
    lines = [
        "| Model | Backbone | dtype | Wiki: no ADC → + LoRA | C4: no ADC → + LoRA |",
        "| :-- | :-- | :-- | --: | --: |",
    ]
    for key, label in MODELS:
        model_runs = sorted(
            (
                (backbone, dtype, results)
                for run_key, backbone, dtype, results in runs
                if run_key == key
            ),
            key=lambda item: (item[0], item[1]),
        )
        if not model_runs:
            lines.append(f"| {label} | — | — | — | — |")
        for backbone, dtype, results in model_runs:
            cells = [
                fmt_change(
                    results.get(f"ppl_noadc_prelora_{dataset}"),
                    results.get(f"ppl_noadc_lora_{dataset}"),
                )
                for dataset in DATASETS
            ]
            lines.append(f"| {label} | {backbone} | {dtype} | {cells[0]} | {cells[1]} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    report = build_report(control_runs(args.results_dir, args.run_suffix))
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
