#!/usr/bin/env python3
"""Validate and summarize multi-architecture ADC transfer results."""

import argparse
import json
from pathlib import Path


MODELS = [
    ("meta-llama/Llama-3.2-1B", "Llama-3.2-1B", "llama32_1b"),
    ("Qwen/Qwen2.5-1.5B", "Qwen2.5-1.5B", "qwen25_15b"),
    ("allenai/OLMo-1B-hf", "OLMo-1B", "olmo_1b"),
    ("TinyLlama/TinyLlama_v1.1", "TinyLlama-1.1B", "tinyllama_11b"),
]

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

ADC_TARGETS = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_json", type=Path)
    parser.add_argument("--markdown-output", type=Path, default=None)
    parser.add_argument("--latex-output", type=Path, default=None)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Generate available rows and report missing fields without failing",
    )
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return records


def accuracy_pct(task_result: dict) -> float | None:
    if not isinstance(task_result, dict):
        return None
    value = task_result.get("accuracy_pct")
    if isinstance(value, (int, float)):
        return float(value)
    value = task_result.get("accuracy")
    if isinstance(value, (int, float)):
        return float(value) * 100.0
    return None


def build_row(
    model_label: str,
    method: str,
    results: dict,
    wiki_key: str,
    c4_key: str,
    downstream_key: str,
) -> tuple[dict | None, list[str]]:
    missing = []
    wiki = results.get(wiki_key)
    c4 = results.get(c4_key)
    downstream = results.get(downstream_key, {})
    if not isinstance(wiki, (int, float)):
        missing.append(wiki_key)
    if not isinstance(c4, (int, float)):
        missing.append(c4_key)

    task_values = {}
    for task, _ in TASKS:
        value = accuracy_pct(downstream.get(task, {}))
        if value is None:
            missing.append(f"{downstream_key}.{task}")
        else:
            task_values[task] = value

    if missing:
        return None, missing

    mean_accuracy = sum(task_values.values()) / len(task_values)
    return {
        "model": model_label,
        "method": method,
        "wiki": float(wiki),
        "c4": float(c4),
        "tasks": task_values,
        "mean": mean_accuracy,
    }, []


def validate_adc_config(record: dict) -> list[str]:
    config = record.get("config", {})
    expected = {
        "torch_dtype": "bfloat16",
        "preprocess_method": "flat_quant",
        "bx": 4,
        "bw": 4,
        "ba": 8,
        "k": 16,
        "mvm_limit": 256,
        "activation_quant": "symmetric",
        "ashift": False,
        "lora_rank": 4,
        "lora_epochs": 5,
        "lora_loss": "ce_kl",
        "fq_epochs": 30,
        "fq_nsamples": 1024,
        "fq_cali_bsz": 16,
        "fq_diag_attn": False,
        "fq_diag_mlp": True,
        "fq_stage_b_epochs": 10,
        "fq_stage_b_prop_alpha": 0.5,
        "fq_stage_b_diag_attn": True,
        "fq_stage_b_diag_mlp": False,
        "calibration_batch_size": 4,
        "max_length": 2048,
        "stride": 1024,
        "max_eval_samples": 1000,
    }
    mismatches = []
    for name, value in expected.items():
        if config.get(name) != value:
            mismatches.append(
                f"config.{name}: expected {value!r}, got {config.get(name)!r}"
            )
    if set(config.get("lora_targets", [])) != ADC_TARGETS:
        mismatches.append(
            "config.lora_targets: expected all seven q/k/v/o/gate/up/down projections"
        )
    return mismatches


def collect_rows(records: list[dict]) -> tuple[list[dict], list[str]]:
    by_run_name = {
        record.get("run_name"): record
        for record in records
        if record.get("status") == "success"
    }
    rows = []
    problems = []

    for model_id, model_label, key in MODELS:
        bf16_record = by_run_name.get(f"{key}_bf16")
        adc_record = by_run_name.get(f"{key}_adc_transfer")

        if bf16_record is None:
            problems.append(f"{model_id}: missing successful {key}_bf16 record")
        elif bf16_record.get("model_name") != model_id:
            problems.append(
                f"{key}_bf16: expected model_name={model_id!r}, "
                f"got {bf16_record.get('model_name')!r}"
            )
        else:
            row, missing = build_row(
                model_label,
                "BF16",
                bf16_record.get("results", {}),
                "ppl_bf16_wikitext2",
                "ppl_bf16_c4",
                "downstream_bf16",
            )
            if row is not None:
                rows.append(row)
            else:
                problems.append(f"{key}_bf16: missing {', '.join(missing)}")

        if adc_record is None:
            problems.append(f"{model_id}: missing successful {key}_adc_transfer record")
            continue
        if adc_record.get("model_name") != model_id:
            problems.append(
                f"{key}_adc_transfer: expected model_name={model_id!r}, "
                f"got {adc_record.get('model_name')!r}"
            )
            continue

        for mismatch in validate_adc_config(adc_record):
            problems.append(f"{key}_adc_transfer: {mismatch}")

        adc_results = adc_record.get("results", {})
        for method, wiki_key, c4_key, downstream_key in (
            (
                "W4A4 + ADC PTQ",
                "ppl_adc_prelora_wikitext2",
                "ppl_adc_prelora_c4",
                "downstream_adc_ptq",
            ),
            (
                "W4A4 + ADC + LoRA",
                "ppl_adc_wikitext2",
                "ppl_adc_c4",
                "downstream_adc_lora",
            ),
        ):
            row, missing = build_row(
                model_label,
                method,
                adc_results,
                wiki_key,
                c4_key,
                downstream_key,
            )
            if row is not None:
                rows.append(row)
            else:
                problems.append(
                    f"{key}_adc_transfer/{method}: missing {', '.join(missing)}"
                )

    return rows, problems


def markdown_table(rows: list[dict]) -> str:
    task_headers = [label for _, label in TASKS]
    headers = ["Model", "Method", "Wiki", "C4", *task_headers, "Avg"]
    alignments = [":--", ":--", "--:", "--:", *(["--:"] * (len(TASKS) + 1))]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(alignments) + " |",
    ]
    for row in rows:
        values = [
            row["model"],
            row["method"],
            f"{row['wiki']:.2f}",
            f"{row['c4']:.2f}",
            *(f"{row['tasks'][task]:.2f}" for task, _ in TASKS),
            f"{row['mean']:.2f}",
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def latex_rows(rows: list[dict]) -> str:
    lines = []
    previous_model = None
    for row in rows:
        if previous_model is not None and row["model"] != previous_model:
            lines.append(r"\midrule")
        values = [
            row["model"].replace("_", r"\_"),
            row["method"].replace("_", r"\_"),
            f"{row['wiki']:.2f}",
            f"{row['c4']:.2f}",
            *(f"{row['tasks'][task]:.2f}" for task, _ in TASKS),
            f"{row['mean']:.2f}",
        ]
        lines.append(" & ".join(values) + r" \\")
        previous_model = row["model"]
    return "\n".join(lines) + "\n"


def write_output(path: Path | None, content: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    records = load_records(args.results_json)
    rows, problems = collect_rows(records)

    if problems:
        print("Result completeness/configuration problems:")
        for problem in problems:
            print(f"- {problem}")
        if not args.allow_incomplete:
            raise SystemExit(2)

    markdown = markdown_table(rows)
    latex = latex_rows(rows)
    write_output(args.markdown_output, markdown)
    write_output(args.latex_output, latex)

    print(markdown, end="")
    print("\nLaTeX rows:")
    print(latex, end="")


if __name__ == "__main__":
    main()
