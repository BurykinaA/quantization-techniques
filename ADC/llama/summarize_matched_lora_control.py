#!/usr/bin/env python3
"""Perplexity before and after the no-ADC LoRA residual on FlatQuant.

Reads the per-run JSON files written by run_matched_lora_control.sh. Models
whose run has not finished yet are shown as "—".
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Directory with the per-run JSON files")
    parser.add_argument("--run-suffix", default="",
                        help="Run-name suffix, e.g. _smoke for smoke runs")
    parser.add_argument("--markdown-output", type=Path, default=None)
    return parser.parse_args()


def successful_results(results_dir: Path) -> dict[str, dict]:
    results = {}
    for path in sorted(results_dir.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            for record in json.load(handle):
                if record.get("status") == "success":
                    results[record["run_name"]] = record.get("results", {})
    return results


def fmt_change(before: float | None, after: float | None) -> str:
    if before is None or after is None:
        return "—"
    change = 100.0 * (after - before) / before
    return f"{before:.2f} → {after:.2f} ({change:+.1f}%)"


def build_report(results: dict[str, dict], suffix: str) -> str:
    lines = [
        "| Model | Wiki: FlatQuant → + LoRA | C4: FlatQuant → + LoRA |",
        "| :-- | --: | --: |",
    ]
    for key, label in MODELS:
        run = results.get(f"{key}_flatquant_noadc_lora{suffix}", {})
        cells = [
            fmt_change(run.get(f"ppl_noadc_prelora_{dataset}"), run.get(f"ppl_noadc_lora_{dataset}"))
            for dataset in DATASETS
        ]
        lines.append(f"| {label} | {cells[0]} | {cells[1]} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    report = build_report(successful_results(args.results_dir), args.run_suffix)
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
