#!/usr/bin/env python3
"""Floor versus midpoint ADC reconstruction on the Stage 1-2 transforms.

Reads the per-run JSON files written by run_adc_midpoint.sh. With
--transfer-json it also reports the share of the ADC arm's Stage 3 gain
(<key>_adc_transfer, Stages 1-2 -> 1-3 with floor) that midpoint
reconstruction alone recovers. Unfinished runs are shown as "—".
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
# Relative difference above which two evaluations of the same model are
# reported as inconsistent.
CONSISTENCY_TOLERANCE = 0.005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, nargs="+", required=True,
                        help="Directories with the per-run JSON files")
    parser.add_argument("--transfer-json", type=Path, default=None,
                        help="multi_arch_transfer.json with the <key>_adc_transfer records")
    parser.add_argument("--run-suffix", default="",
                        help="Run-name suffix, e.g. _smoke for smoke runs")
    parser.add_argument("--markdown-output", type=Path, default=None)
    return parser.parse_args()


def load_records(paths: list[Path]) -> list[dict]:
    records = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            records.extend(
                record for record in json.load(handle) if record.get("status") == "success"
            )
    return records


def runs_by_name(records: list[dict]) -> dict[str, list[tuple[str, dict]]]:
    """run_name -> [(dtype, results)], one entry per result directory."""
    runs: dict[str, list[tuple[str, dict]]] = {}
    for record in records:
        dtype = record.get("config", {}).get("torch_dtype", "?")
        runs.setdefault(record.get("run_name"), []).append((dtype, record.get("results", {})))
    return runs


def fmt_change(before: float | None, after: float | None) -> str:
    if before is None or after is None:
        return "—"
    change = 100.0 * (after - before) / before
    return f"{before:.2f} → {after:.2f} ({change:+.1f}%)"


def fmt_share(floor: float | None, midpoint: float | None, stage3: float | None) -> str:
    if floor is None or midpoint is None or stage3 is None or floor == stage3:
        return "—"
    return f"{100.0 * (floor - midpoint) / (floor - stage3):.0f}%"


def build_report(runs: dict, transfer: dict, suffix: str) -> str:
    eval_rows, stage3_rows, checks = [], [], []

    def check(label: str, first: float | None, second: float | None) -> None:
        if first is None or second is None:
            return
        relative = abs(first - second) / max(abs(first), 1e-12)
        status = "OK" if relative <= CONSISTENCY_TOLERANCE else "MISMATCH"
        checks.append(f"- {status}: {label}: {first:.4f} vs {second:.4f}")

    for key, label in MODELS:
        adc_arm = transfer.get(f"{key}_adc_transfer", {})
        floor_runs = dict(runs.get(f"{key}_adc_floor_eval{suffix}", []))
        midpoint_runs = dict(runs.get(f"{key}_adc_midpoint_eval{suffix}", []))
        stage3_runs = dict(runs.get(f"{key}_adc_midpoint_stage3{suffix}", []))

        for dtype in sorted(set(floor_runs) | set(midpoint_runs)):
            floor = floor_runs.get(dtype, {})
            midpoint = midpoint_runs.get(dtype, {})
            # The transfer runs are bfloat16, so the share is only defined there.
            stage13 = adc_arm if dtype == "bfloat16" else {}
            cells = [
                fmt_change(floor.get(f"ppl_adc_{ds}"), midpoint.get(f"ppl_adc_{ds}"))
                for ds in DATASETS
            ]
            shares = [
                fmt_share(
                    floor.get(f"ppl_adc_{ds}"),
                    midpoint.get(f"ppl_adc_{ds}"),
                    stage13.get(f"ppl_adc_{ds}"),
                )
                for ds in DATASETS
            ]
            eval_rows.append(f"| {label} | {dtype} | {cells[0]} | {cells[1]} | {shares[0]} / {shares[1]} |")
            for ds in DATASETS:
                check(
                    f"{label} {dtype} {ds}: floor eval vs transfer Stage 1-2",
                    floor.get(f"ppl_adc_{ds}"),
                    stage13.get(f"ppl_adc_prelora_{ds}"),
                )

        for dtype, results in sorted(stage3_runs.items()):
            cells = [
                fmt_change(results.get(f"ppl_adc_prelora_{ds}"), results.get(f"ppl_adc_{ds}"))
                for ds in DATASETS
            ]
            stage13 = adc_arm if dtype == "bfloat16" else {}
            reference = " / ".join(
                f"{stage13[f'ppl_adc_{ds}']:.2f}" if f"ppl_adc_{ds}" in stage13 else "—"
                for ds in DATASETS
            )
            stage3_rows.append(f"| {label} | {dtype} | {cells[0]} | {cells[1]} | {reference} |")
            for ds in DATASETS:
                check(
                    f"{label} {dtype} {ds}: stage3 pre-LoRA vs midpoint eval",
                    results.get(f"ppl_adc_prelora_{ds}"),
                    midpoint_runs.get(dtype, {}).get(f"ppl_adc_{ds}"),
                )

    lines = [
        "## Stage 1–2 transforms, converter on",
        "",
        "Share = (floor − midpoint) / (floor − Stage 1–3 with floor): the part of the "
        "ADC arm's Stage 3 gain that midpoint reconstruction alone recovers.",
        "",
        "| Model | dtype | Wiki: floor → midpoint | C4: floor → midpoint | Share of Stage 3 gain (Wiki / C4) |",
        "| :-- | :-- | --: | --: | --: |",
        *(eval_rows or ["| — | — | — | — | — |"]),
        "",
        "## Stage 3 retrained with midpoint",
        "",
        "| Model | dtype | Wiki: midpoint → + LoRA | C4: midpoint → + LoRA | Stage 1–3 with floor (Wiki / C4) |",
        "| :-- | :-- | --: | --: | --: |",
        *(stage3_rows or ["| — | — | — | — | — |"]),
        "",
        "## Consistency checks",
        "",
        *(checks or ["- no overlapping measurements yet"]),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    runs = runs_by_name(
        load_records([path for directory in args.results_dir for path in sorted(directory.glob("*.json"))])
    )
    transfer = {}
    if args.transfer_json is not None:
        transfer = {
            record.get("run_name"): record.get("results", {})
            for record in load_records([args.transfer_json])
        }
    report = build_report(runs, transfer, args.run_suffix)
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
