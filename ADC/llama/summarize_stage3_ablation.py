#!/usr/bin/env python3
"""Stage 3 on backbones with less pre-ADC calibration.

Reads the per-run JSON files written by run_stage3_ablation.sh and reports,
per corpus and model, converter-on perplexity before and after the same
Stage 3 residual: without calibration, next to the model's Table 2 Stages
1-3 row. For Llama it adds Stage 1 only and Stages 1-2 rerun with the
current code when those runs exist. Runs made with LM_EVAL=1 also get the
eight-task downstream table next to the Table 2 rows. Unfinished runs are
shown as "—".
"""

import argparse
import json
from pathlib import Path


MODELS = [
    ("llama32_1b", "Llama-3.2-1B"),
    ("smollm2_17b", "SmolLM2-1.7B"),
    ("tinyllama_11b", "TinyLlama-1.1B"),
]
DATASETS = (("wikitext2", "WikiText-2"), ("c4", "C4"))
# Table 2 Stages 1-2 -> Stages 1-3 rows, (before, after) Stage 3. Llama:
# run_overnight_int4_v8.sh r4_all_ce_kl; SmolLM2 and TinyLlama:
# <key>_adc_transfer.
TABLE2_STAGES12 = {
    "llama32_1b": {"wikitext2": (26.66, 14.03), "c4": (45.62, 23.20)},
    "smollm2_17b": {"wikitext2": (14.70, 10.73), "c4": (23.37, 17.64)},
    "tinyllama_11b": {"wikitext2": (11.62, 10.90), "c4": (20.23, 18.69)},
}
TASKS = [
    ("hellaswag", "HSwag"),
    ("mmlu", "MMLU"),
    ("winogrande", "WinGr"),
    ("arc_easy", "ARC-E"),
    ("arc_challenge", "ARC-C"),
    ("piqa", "PIQA"),
    ("openbookqa", "OBQA"),
    ("boolq", "BoolQ"),
]
# Table 2 downstream accuracy (%) in TASKS order, from
# adc_aware_llm_aaai2027.tex.
TABLE2_DOWNSTREAM = {
    "llama32_1b": {
        "BF16 reference": (64.19, 32.15, 63.06, 61.83, 36.77, 74.70, 37.40, 63.91),
        "Stages 1–2, Table 2": (40.03, 24.41, 53.43, 42.63, 26.02, 61.70, 27.00, 60.67),
        "Stages 1–3, Table 2": (49.96, 25.72, 53.67, 54.63, 29.27, 68.06, 34.60, 60.67),
    },
    "smollm2_17b": {
        "BF16 reference": (71.44, 49.99, 65.75, 73.36, 47.27, 77.42, 43.80, 72.39),
        "Stages 1–2, Table 2": (53.24, 26.86, 54.70, 58.75, 32.25, 68.23, 36.20, 55.23),
        "Stages 1–3, Table 2": (60.12, 31.23, 60.06, 64.31, 38.57, 71.27, 38.60, 66.09),
    },
    "tinyllama_11b": {
        "BF16 reference": (55.57, 26.16, 49.17, 42.59, 30.12, 70.46, 23.80, 54.77),
        "Stages 1–2, Table 2": (46.74, 25.48, 52.25, 39.56, 26.45, 62.89, 25.60, 55.54),
        "Stages 1–3, Table 2": (46.68, 25.42, 51.07, 41.50, 25.43, 63.55, 27.80, 56.15),
    },
}
# Relative difference above which two evaluations of the same backbone are
# reported as inconsistent.
CONSISTENCY_TOLERANCE = 0.005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, nargs="+", required=True,
                        help="Directories with the per-run JSON files")
    parser.add_argument("--run-suffix", default="",
                        help="Run-name suffix, e.g. _smoke for smoke runs")
    parser.add_argument("--markdown-output", type=Path, default=None)
    return parser.parse_args()


def load_results(directories: list[Path]) -> dict[str, dict]:
    """run_name -> results of its successful record."""
    runs = {}
    for directory in directories:
        for path in sorted(directory.glob("*.json")):
            with path.open(encoding="utf-8") as handle:
                for record in json.load(handle):
                    if record.get("status") == "success":
                        runs[record.get("run_name")] = record.get("results", {})
    return runs


def fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def fmt_change(before: float | None, after: float | None) -> str:
    if before is None or after is None:
        return "—"
    return f"{100.0 * (after - before) / before:+.1f}%"


def downstream_values(results: dict | None, key: str) -> list[float | None] | None:
    """Per-task accuracy (%) of results[key] in TASKS order, None when absent."""
    downstream = (results or {}).get(key) or {}
    values = [(downstream.get(task) or {}).get("accuracy_pct") for task, _ in TASKS]
    return None if all(value is None for value in values) else values


def fmt_mean(values: list[float | None]) -> str:
    if any(value is None for value in values):
        return "—"
    return f"{sum(values) / len(values):.2f}"


def build_downstream(runs: dict[str, dict], suffix: str) -> list[str]:
    stage1 = runs.get(f"llama32_1b_stage1{suffix}")
    stage1_stage3 = runs.get(f"llama32_1b_stage1_stage3{suffix}")
    stage12_stage3 = runs.get(f"llama32_1b_stage12_stage3{suffix}")

    rows = []
    for key, label in MODELS:
        # A run without LoRA stores its final downstream under downstream_adc_ptq.
        measured = [
            ("a) No calibration + Stage 3",
             downstream_values(runs.get(f"{key}_no_calib_stage3{suffix}"), "downstream_adc_lora")),
        ]
        if key == "llama32_1b":
            measured += [
                ("b) Stage 1 only", downstream_values(stage1, "downstream_adc_ptq")),
                ("b) Stage 1 + Stage 3", downstream_values(stage1_stage3, "downstream_adc_lora")),
                ("c) Stages 1–2 + Stage 3, current code",
                 downstream_values(stage12_stage3, "downstream_adc_lora")),
            ]
        measured = [(backbone, values) for backbone, values in measured if values is not None]
        if not measured:
            continue
        rows += [(label, backbone, values) for backbone, values in measured]
        rows += [(label, backbone, list(values)) for backbone, values in TABLE2_DOWNSTREAM[key].items()]

    if not rows:
        return ["## Downstream", "", "- no downstream results yet (run with LM_EVAL=1)", ""]
    return [
        "## Downstream",
        "",
        "Accuracy (%), lm-evaluation-harness, MMLU 5-shot and the rest 0-shot, "
        "measured after Stage 3 only.",
        "",
        "| Model | Backbone | " + " | ".join(short for _, short in TASKS) + " | Avg |",
        "| :-- | :-- | " + " | ".join("--:" for _ in TASKS) + " | --: |",
        *(
            f"| {label} | {backbone} | " + " | ".join(fmt(value) for value in values)
            + f" | {fmt_mean(values)} |"
            for label, backbone, values in rows
        ),
        "",
    ]


def build_report(runs: dict[str, dict], suffix: str) -> str:
    stage1 = runs.get(f"llama32_1b_stage1{suffix}")
    stage1_stage3 = runs.get(f"llama32_1b_stage1_stage3{suffix}")
    stage12_stage3 = runs.get(f"llama32_1b_stage12_stage3{suffix}")

    lines = [
        "# Stage 3 without Stages 1–2",
        "",
        "Perplexity with the converter on, lower is better. Every row trains the same "
        "Stage 3 residual (rank 4, all seven projections, CE + 0.5 KL, 5 epochs).",
        "",
    ]
    checks = []

    def check(label: str, first: float | None, second: float | None) -> None:
        if first is None or second is None:
            return
        relative = abs(first - second) / max(abs(first), 1e-12)
        status = "OK" if relative <= CONSISTENCY_TOLERANCE else "MISMATCH"
        checks.append(f"- {status}: {label}: {first:.4f} vs {second:.4f}")

    for ds, title in DATASETS:
        rows = []
        for key, label in MODELS:
            no_calib = runs.get(f"{key}_no_calib_stage3{suffix}", {})
            rows.append((label, "a) No calibration",
                         no_calib.get(f"ppl_adc_prelora_{ds}"), no_calib.get(f"ppl_adc_{ds}")))
            if key == "llama32_1b" and (stage1 is not None or stage1_stage3 is not None):
                rows.append((label, "b) Stage 1 only",
                             (stage1 or {}).get(f"ppl_adc_{ds}"),
                             (stage1_stage3 or {}).get(f"ppl_adc_{ds}")))
            if key == "llama32_1b" and stage12_stage3 is not None:
                rows.append((label, "c) Stages 1–2, current code",
                             stage12_stage3.get(f"ppl_adc_prelora_{ds}"),
                             stage12_stage3.get(f"ppl_adc_{ds}")))
            rows.append((label, "c) Stages 1–2, Table 2", *TABLE2_STAGES12[key][ds]))

        lines += [
            f"## {title}",
            "",
            "| Model | Backbone | Before Stage 3 | After Stage 3 | Change |",
            "| :-- | :-- | --: | --: | --: |",
            *(
                f"| {label} | {backbone} | {fmt(before)} | {fmt(after)} | "
                f"{fmt_change(before, after)} |"
                for label, backbone, before, after in rows
            ),
            "",
        ]
        check(
            f"Llama-3.2-1B {title}: Stage 1 eval vs Stage 1 reloaded before Stage 3",
            (stage1 or {}).get(f"ppl_adc_{ds}"),
            (stage1_stage3 or {}).get(f"ppl_adc_prelora_{ds}"),
        )

    lines += build_downstream(runs, suffix)
    lines += [
        "## Checks",
        "",
        *(checks or ["- no overlapping measurements yet"]),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    report = build_report(load_results(args.results_dir), args.run_suffix)
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
