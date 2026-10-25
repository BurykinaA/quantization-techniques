#!/usr/bin/env python3
"""Compare the post-ADC LoRA gain with and without the ADC (matched control).

The ADC arm comes from the multi-architecture transfer JSON; the no-ADC arms
come from the per-run JSON files written by run_matched_lora_control.sh.
Missing runs are shown as "—" so the table can be read while runs are still
in progress.
"""

import argparse
from pathlib import Path

from summarize_multi_arch_transfer import TASKS, accuracy_pct, load_records


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
    parser.add_argument("--transfer-json", type=Path, default=None,
                        help="multi_arch_transfer.json with the ADC-arm records")
    parser.add_argument("--control-dir", type=Path, required=True,
                        help="Directory with the per-run control JSON files")
    parser.add_argument("--markdown-output", type=Path, default=None)
    parser.add_argument("--run-suffix", default="",
                        help="Run-name suffix, e.g. _smoke for smoke runs")
    return parser.parse_args()


def successful_records(paths: list[Path]) -> dict[str, dict]:
    records = {}
    for path in paths:
        for record in load_records(path):
            if record.get("status") == "success":
                records[record.get("run_name")] = record
    return records


def mean_accuracy(downstream: dict | None) -> float | None:
    values = [accuracy_pct((downstream or {}).get(task, {})) for task, _ in TASKS]
    if any(value is None for value in values):
        return None
    return sum(values) / len(values)


def results_of(records: dict[str, dict], run_name: str) -> dict:
    return records.get(run_name, {}).get("results", {})


def ppl_pair(results: dict, prefix: str) -> tuple:
    return tuple(results.get(f"{prefix}{dataset}") for dataset in DATASETS)


def fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def fmt_change(before: float | None, after: float | None) -> str:
    if before is None or after is None:
        return f"{fmt(before)} → {fmt(after)}"
    change = 100.0 * (after - before) / before
    return f"{before:.2f} → {after:.2f} ({change:+.1f}%)"


def fmt_accuracy(before: float | None, after: float | None) -> str:
    return f"{fmt(before)} → {fmt(after)}"


def task_row(
    model_label: str,
    row_label: str,
    ppl_results: dict,
    ppl_prefix: str,
    downstream: dict | None,
) -> tuple:
    tasks = [accuracy_pct((downstream or {}).get(task, {})) for task, _ in TASKS]
    return (
        model_label,
        row_label,
        *(fmt(value) for value in ppl_pair(ppl_results, ppl_prefix)),
        *(fmt(value) for value in tasks),
        fmt(mean_accuracy(downstream)),
    )


def build_report(transfer: dict[str, dict], control: dict[str, dict], suffix: str) -> str:
    gain_rows = []
    cross_rows = []
    task_rows = []
    checks = []

    def check(label: str, first: float | None, second: float | None) -> None:
        if first is None or second is None:
            return
        relative = abs(first - second) / max(abs(first), 1e-12)
        status = "OK" if relative <= CONSISTENCY_TOLERANCE else "MISMATCH"
        checks.append(f"- {status}: {label}: {first:.4f} vs {second:.4f}")

    for key, label in MODELS:
        adc = results_of(transfer, f"{key}_adc_transfer")
        adc_off_ptq = results_of(transfer, f"{key}_int4_ptq_adc_off")
        same = results_of(control, f"{key}_noadc_lora_same_backbone{suffix}")
        same_downstream = results_of(
            control, f"{key}_noadc_lora_same_backbone{suffix}_downstream"
        )
        flat = results_of(control, f"{key}_noadc_lora_flatquant{suffix}")
        flat_downstream = results_of(
            control, f"{key}_noadc_lora_flatquant{suffix}_downstream"
        )

        if adc:
            before, after = ppl_pair(adc, "ppl_adc_prelora_"), ppl_pair(adc, "ppl_adc_")
            gain_rows.append((
                label, "ADC-Quant Stages 1–2", "ADC on",
                *(fmt_change(b, a) for b, a in zip(before, after)),
                fmt_accuracy(
                    mean_accuracy(adc.get("downstream_adc_ptq")),
                    mean_accuracy(adc.get("downstream_adc_lora")),
                ),
            ))
            cross_rows.append((
                label, "ADC-Quant Stages 1–2", "ADC on", "ADC off",
                *(fmt(v) for v in ppl_pair(adc, "ppl_bypass_")),
            ))
        if same:
            before = ppl_pair(same, "ppl_noadc_prelora_")
            after = ppl_pair(same, "ppl_noadc_lora_")
            gain_rows.append((
                label, "ADC-Quant Stages 1–2", "ADC off",
                *(fmt_change(b, a) for b, a in zip(before, after)),
                fmt_accuracy(
                    mean_accuracy(adc_off_ptq.get("downstream_int4_ptq")),
                    mean_accuracy(same_downstream.get("downstream_noadc_lora")),
                ),
            ))
            cross_rows.append((
                label, "ADC-Quant Stages 1–2", "ADC off", "ADC on",
                *(fmt(v) for v in ppl_pair(same, "ppl_adc_on_lora_")),
            ))
            for dataset in DATASETS:
                check(
                    f"{label} {dataset}: control pre-LoRA ADC off vs transfer int4_ptq_adc_off",
                    same.get(f"ppl_noadc_prelora_{dataset}"),
                    adc_off_ptq.get(f"ppl_int4_ptq_{dataset}"),
                )
                check(
                    f"{label} {dataset}: control pre-LoRA ADC on vs transfer pre-LoRA ADC",
                    same.get(f"ppl_adc_on_prelora_{dataset}"),
                    adc.get(f"ppl_adc_prelora_{dataset}"),
                )
                check(
                    f"{label} {dataset}: phase-2 reload vs phase-1 adapter (ADC off)",
                    same_downstream.get(f"ppl_noadc_lora_{dataset}"),
                    same.get(f"ppl_noadc_lora_{dataset}"),
                )
        if flat:
            before = ppl_pair(flat, "ppl_noadc_prelora_")
            after = ppl_pair(flat, "ppl_noadc_lora_")
            gain_rows.append((
                label, "FlatQuant, no ADC floor in calibration", "ADC off",
                *(fmt_change(b, a) for b, a in zip(before, after)),
                fmt_accuracy(
                    mean_accuracy(flat_downstream.get("downstream_noadc_prelora")),
                    mean_accuracy(flat_downstream.get("downstream_noadc_lora")),
                ),
            ))
            cross_rows.append((
                label, "FlatQuant, no ADC floor in calibration", "none (pre-LoRA)", "ADC on",
                *(fmt(v) for v in ppl_pair(flat, "ppl_adc_on_prelora_")),
            ))
            cross_rows.append((
                label, "FlatQuant, no ADC floor in calibration", "ADC off", "ADC on",
                *(fmt(v) for v in ppl_pair(flat, "ppl_adc_on_lora_")),
            ))
            for dataset in DATASETS:
                check(
                    f"{label} {dataset}: FlatQuant phase-2 vs phase-1 before LoRA (ADC off)",
                    flat_downstream.get(f"ppl_noadc_prelora_{dataset}"),
                    flat.get(f"ppl_noadc_prelora_{dataset}"),
                )
                check(
                    f"{label} {dataset}: FlatQuant phase-2 reload vs phase-1 adapter (ADC off)",
                    flat_downstream.get(f"ppl_noadc_lora_{dataset}"),
                    flat.get(f"ppl_noadc_lora_{dataset}"),
                )

        # Table 2 layout: perplexities from the training run, accuracies from
        # the downstream run of the same adapter.
        for row_label, ppl_results, ppl_prefix, downstream in (
            ("ADC-Quant, Stages 1–2", adc, "ppl_adc_prelora_", adc.get("downstream_adc_ptq")),
            ("ADC-Quant, Stages 1–3", adc, "ppl_adc_", adc.get("downstream_adc_lora")),
            ("Stages 1–2, ADC off", adc_off_ptq, "ppl_int4_ptq_",
             adc_off_ptq.get("downstream_int4_ptq")),
            ("Stages 1–2, ADC off + residual", same, "ppl_noadc_lora_",
             same_downstream.get("downstream_noadc_lora")),
            ("FlatQuant, no ADC", flat, "ppl_noadc_prelora_",
             flat_downstream.get("downstream_noadc_prelora")),
            ("FlatQuant, no ADC + residual", flat, "ppl_noadc_lora_",
             flat_downstream.get("downstream_noadc_lora")),
        ):
            if ppl_results:
                task_rows.append(task_row(label, row_label, ppl_results, ppl_prefix, downstream))

    task_headers = [task_label for _, task_label in TASKS]
    lines = [
        "## Adapter gain with and without the ADC",
        "",
        "Same rank-4 residual, data, CE+KL objective, and schedule in every row; "
        "only the converter during adapter training and evaluation differs.",
        "",
        "| Model | Backbone | Converter | Wiki before → after | C4 before → after | Avg acc before → after |",
        "| :-- | :-- | :-- | --: | --: | --: |",
        *("| " + " | ".join(row) + " |" for row in gain_rows),
        "",
        "## Cross-deployment",
        "",
        "| Model | Backbone | Adapter trained with | Evaluated with | Wiki | C4 |",
        "| :-- | :-- | :-- | :-- | --: | --: |",
        *("| " + " | ".join(row) + " |" for row in cross_rows),
        "",
        "## Per-task results",
        "",
        "| Model | Row | Wiki | C4 | " + " | ".join(task_headers) + " | Avg |",
        "| :-- | :-- | --: | --: | " + " | ".join(["--:"] * len(task_headers)) + " | --: |",
        *("| " + " | ".join(row) + " |" for row in task_rows),
        "",
        "## Consistency checks",
        "",
        *(checks or ["- no overlapping measurements yet"]),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    transfer = {}
    if args.transfer_json is not None and args.transfer_json.is_file():
        transfer = successful_records([args.transfer_json])
    else:
        print(f"Transfer JSON not found ({args.transfer_json}); ADC-arm rows omitted.\n")
    control = successful_records(sorted(args.control_dir.glob("*.json")))

    report = build_report(transfer, control, args.run_suffix)
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
