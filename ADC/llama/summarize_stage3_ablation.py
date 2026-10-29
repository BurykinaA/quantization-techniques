#!/usr/bin/env python3
"""Stage 3 on Llama-3.2-1B backbones with less pre-ADC calibration.

Reads the per-run JSON files written by run_stage3_ablation.sh and reports,
per corpus, converter-on perplexity before and after the same Stage 3
residual for: no calibration, Stage 1 only, Stages 1-2 rerun with the same
code, and the Table 2 Stages 1-3 row. Unfinished runs are shown as "—".
"""

import argparse
import json
from pathlib import Path


DATASETS = (("wikitext2", "WikiText-2"), ("c4", "C4"))
# Table 2, Llama-3.2-1B: run_overnight_int4_v8.sh r4_all_ce_kl (LoRA microbatch
# 4 x 1) and the "ADC, no calib." row.
TABLE2_STAGES12 = {"wikitext2": (26.66, 14.03), "c4": (45.62, 23.20)}
TABLE2_NO_CALIB = {"wikitext2": 2402.19, "c4": 2614.93}
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


def build_report(runs: dict[str, dict], suffix: str) -> str:
    key = "llama32_1b"
    no_calib = runs.get(f"{key}_no_calib_stage3{suffix}", {})
    stage1 = runs.get(f"{key}_stage1{suffix}", {})
    stage1_stage3 = runs.get(f"{key}_stage1_stage3{suffix}", {})
    stage12_stage3 = runs.get(f"{key}_stage12_stage3{suffix}", {})

    lines = [
        "# Stage 3 without Stages 1–2, Llama-3.2-1B",
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

    def compare(label: str, measured: float | None, reference: float) -> None:
        if measured is None:
            return
        checks.append(
            f"- reference: {label}: {measured:.2f} vs {reference:.2f} "
            f"({fmt_change(reference, measured)})"
        )

    for ds, title in DATASETS:
        rows = [
            ("a) No calibration", no_calib.get(f"ppl_adc_prelora_{ds}"),
             no_calib.get(f"ppl_adc_{ds}")),
            ("b) Stage 1 only", stage1.get(f"ppl_adc_{ds}"),
             stage1_stage3.get(f"ppl_adc_{ds}")),
            ("c) Stages 1–2, same code", stage12_stage3.get(f"ppl_adc_prelora_{ds}"),
             stage12_stage3.get(f"ppl_adc_{ds}")),
            ("c) Stages 1–2, Table 2", *TABLE2_STAGES12[ds]),
        ]
        lines += [
            f"## {title}",
            "",
            "| Backbone | Before Stage 3 | After Stage 3 | Change |",
            "| :-- | --: | --: | --: |",
            *(
                f"| {label} | {fmt(before)} | {fmt(after)} | {fmt_change(before, after)} |"
                for label, before, after in rows
            ),
            "",
        ]
        check(
            f"{title}: Stage 1 eval vs Stage 1 reloaded before Stage 3",
            stage1.get(f"ppl_adc_{ds}"),
            stage1_stage3.get(f"ppl_adc_prelora_{ds}"),
        )
        compare(
            f"{title}: no calibration before Stage 3 vs Table 2 'ADC, no calib.'",
            no_calib.get(f"ppl_adc_prelora_{ds}"),
            TABLE2_NO_CALIB[ds],
        )
        compare(
            f"{title}: Stages 1–2 same code vs Table 2, before Stage 3",
            stage12_stage3.get(f"ppl_adc_prelora_{ds}"),
            TABLE2_STAGES12[ds][0],
        )
        compare(
            f"{title}: Stages 1–2 same code vs Table 2, after Stage 3",
            stage12_stage3.get(f"ppl_adc_{ds}"),
            TABLE2_STAGES12[ds][1],
        )

    lines += [
        "## Checks",
        "",
        *(checks or ["- no measurements yet"]),
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
