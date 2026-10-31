#!/usr/bin/env python3
"""Seed variance of Stage 3 on the Table 2 Stages 1-2 backbone.

Reads the per-run JSON files written by run_stage3_seeds.sh and reports, per
model, converter-on perplexity and the eight downstream accuracies after
Stage 3 for every seed, their mean and sample standard deviation, and the
downstream gain over the Stages 1-2 backbone with a 95% confidence interval,
next to the Table 2 row (one seed). Unfinished seeds are shown as "—".
"""

import argparse
import math
import re
from pathlib import Path

try:
    from ADC.llama.summarize_stage3_ablation import (
        DATASETS,
        MODELS,
        TABLE2_DOWNSTREAM,
        TABLE2_STAGES12,
        TASKS,
        downstream_values,
        load_results,
    )
except ImportError:  # run as a script from ADC/llama
    from summarize_stage3_ablation import (
        DATASETS,
        MODELS,
        TABLE2_DOWNSTREAM,
        TABLE2_STAGES12,
        TASKS,
        downstream_values,
        load_results,
    )

# Two-sided 95% Student t quantiles by sample size (n - 1 degrees of freedom).
T_CRITICAL_95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
                 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}
# Relative spread of the pre-Stage 3 perplexity across seeds above which the
# backbone is reported as not reproducible.
PRELORA_TOLERANCE = 0.005
RUN_NAME = re.compile(r"^(?P<key>.+)_stage3_seed(?P<seed>\d+)(?P<suffix>.*)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, nargs="+", required=True,
                        help="Directories with the per-run JSON files")
    parser.add_argument("--run-suffix", default="",
                        help="Run-name suffix, e.g. _smoke for smoke runs")
    parser.add_argument("--markdown-output", type=Path, default=None)
    return parser.parse_args()


def mean_std(values: list[float]) -> tuple[float, float | None]:
    """Mean and sample standard deviation (None for a single value)."""
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, None
    return mean, math.sqrt(sum((value - mean) ** 2 for value in values) / (n - 1))


def confidence_interval(values: list[float]) -> tuple[float, float | None]:
    """Mean and the half-width of its 95% t confidence interval."""
    mean, std = mean_std(values)
    if std is None:
        return mean, None
    n = len(values)
    t_value = T_CRITICAL_95.get(n, 2.0)
    return mean, t_value * std / math.sqrt(n)


def seed_runs(runs: dict[str, dict], key: str, suffix: str) -> list[tuple[int, dict]]:
    """(seed, results) of the model's finished seed runs, sorted by seed."""
    found = []
    for run_name, results in runs.items():
        match = RUN_NAME.match(run_name)
        if match and match["key"] == key and match["suffix"] == suffix:
            found.append((int(match["seed"]), results))
    return sorted(found, key=lambda item: item[0])


def fmt(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def fmt_signed(value: float | None) -> str:
    return "—" if value is None else f"{value:+.2f}"


def fmt_pm(values: list[float], signed: bool = False) -> str:
    if not values:
        return "—"
    mean, std = mean_std(values)
    mean_text = f"{mean:+.2f}" if signed else f"{mean:.2f}"
    return mean_text if std is None else f"{mean_text} ± {std:.2f}"


def avg(values: list[float | None] | None) -> float | None:
    if not values or any(value is None for value in values):
        return None
    return sum(values) / len(values)


def present(values: list[float | None]) -> list[float]:
    return [value for value in values if value is not None]


def build_model_section(key: str, label: str, seeds: list[tuple[int, dict]]) -> list[str]:
    lines = [f"## {label}", ""]
    if not seeds:
        lines += ["- no finished seeds yet", ""]
        return lines

    table2_pre = TABLE2_STAGES12[key]
    table2_down = TABLE2_DOWNSTREAM[key]
    table2_pre_avg = avg(list(table2_down["Stages 1–2, Table 2"]))
    table2_post = list(table2_down["Stages 1–3, Table 2"])

    # The backbone before Stage 3 is the same for every seed: perplexity from
    # every run, downstream from the one run that measured it.
    prelora_ppl = {ds: present([results.get(f"ppl_adc_prelora_{ds}") for _, results in seeds])
                   for ds, _ in DATASETS}
    pre_down = next((values for values in
                     (downstream_values(results, "downstream_adc_ptq") for _, results in seeds)
                     if values is not None), None)
    pre_down_avg = avg(pre_down)
    baseline_avg = pre_down_avg if pre_down_avg is not None else table2_pre_avg
    backbone = []
    for ds, ds_label in DATASETS:
        values = prelora_ppl[ds]
        backbone.append(f"{ds_label} {fmt(values[0]) if values else '—'} (Table 2 {table2_pre[ds][0]:.2f})")
    if pre_down_avg is not None:
        backbone.append(f"downstream mean {pre_down_avg:.2f}% (Table 2 {table2_pre_avg:.2f}%)")
        baseline_note = "measured here"
    else:
        backbone.append(f"downstream mean {table2_pre_avg:.2f}% from Table 2 (not re-measured yet)")
        baseline_note = "Table 2 row"
    lines += [f"Stages 1–2 backbone before Stage 3: {', '.join(backbone)}. "
              f"Δ Avg is the downstream mean minus this backbone's ({baseline_note}).", ""]

    header = ["Seed", "Wiki", "C4", *[short for _, short in TASKS], "Avg", "Δ Avg"]
    lines += ["| " + " | ".join(header) + " |",
              "| :-- | " + " | ".join("--:" for _ in header[1:]) + " |"]

    columns: dict[str, list[float]] = {name: [] for name in header[1:]}
    for seed, results in seeds:
        wiki = results.get("ppl_adc_wikitext2")
        c4 = results.get("ppl_adc_c4")
        down = downstream_values(results, "downstream_adc_lora") or [None] * len(TASKS)
        down_avg = avg(down)
        delta = None if down_avg is None else down_avg - baseline_avg
        cells = [wiki, c4, *down, down_avg, delta]
        for name, value in zip(header[1:], cells):
            if value is not None:
                columns[name].append(value)
        lines.append("| " + " | ".join([str(seed), fmt(wiki), fmt(c4),
                                          *[fmt(value) for value in down],
                                          fmt(down_avg), fmt_signed(delta)]) + " |")

    lines.append("| " + " | ".join(["mean ± std",
                                      *[fmt_pm(columns[name], signed=(name == "Δ Avg"))
                                        for name in header[1:]]]) + " |")
    lines.append("| " + " | ".join([
        "Table 2 (seed 42)",
        f"{table2_pre['wikitext2'][1]:.2f}",
        f"{table2_pre['c4'][1]:.2f}",
        *[f"{value:.2f}" for value in table2_post],
        f"{avg(table2_post):.2f}",
        fmt_signed(avg(table2_post) - table2_pre_avg),
    ]) + " |")
    lines.append("")

    gains = columns["Δ Avg"]
    if gains:
        mean, half = confidence_interval(gains)
        interval = "" if half is None else f", 95% CI [{mean - half:+.2f}, {mean + half:+.2f}]"
        lines.append(f"- Downstream gain of Stage 3 over Stages 1–2: {fmt_pm(gains, signed=True)}"
                     f"{interval} (n = {len(gains)}; Table 2: {avg(table2_post) - table2_pre_avg:+.2f})")
    else:
        lines.append("- no downstream results yet")
    for ds, ds_label in DATASETS:
        values = columns["Wiki" if ds == "wikitext2" else "C4"]
        if values:
            lines.append(f"- {ds_label} after Stage 3: {fmt_pm(values)} (n = {len(values)}; "
                         f"Table 2: {table2_pre[ds][1]:.2f})")
    lines.append("")
    return lines


def build_checks(runs_by_model: dict[str, list[tuple[int, dict]]]) -> list[str]:
    checks = []
    for key, label in MODELS:
        seeds = runs_by_model.get(key, [])
        if not seeds:
            continue
        for ds, ds_label in DATASETS:
            values = present([results.get(f"ppl_adc_prelora_{ds}") for _, results in seeds])
            if len(values) > 1 and (max(values) - min(values)) / min(values) > PRELORA_TOLERANCE:
                checks.append(f"- {label}: pre-Stage 3 {ds_label} perplexity differs between seeds "
                              f"({min(values):.2f}–{max(values):.2f}); the backbone should not depend on the seed")
        missing = [str(seed) for seed, results in seeds
                   if downstream_values(results, "downstream_adc_lora") is None]
        if missing:
            checks.append(f"- {label}: seeds {', '.join(missing)} have no downstream results")
        if all(downstream_values(results, "downstream_adc_ptq") is None for _, results in seeds):
            checks.append(f"- {label}: the Stages 1–2 backbone's downstream was not measured; "
                          f"Δ Avg uses the Table 2 row")
    return checks or ["- none"]


def build_report(runs: dict[str, dict], suffix: str) -> str:
    runs_by_model = {key: seed_runs(runs, key, suffix) for key, _ in MODELS}
    lines = [
        "# Stage 3 seed variance",
        "",
        "Converter on. Every seed retrains the Table 2 Stage 3 residual (rank 4, all seven "
        "projections, CE + 0.5 KL, 5 epochs) on the Table 2 Stages 1–2 backbone; the seed sets "
        "the adapter initialization and, with DATA_SEED=1, the 256 calibration sequences and "
        "their order. Perplexity on WikiText-2 and C4, accuracy (%) from lm-evaluation-harness "
        "(MMLU 5-shot, the rest 0-shot). std is the sample standard deviation.",
        "",
    ]
    for key, label in MODELS:
        lines += build_model_section(key, label, runs_by_model[key])
    lines += ["## Checks", "", *build_checks(runs_by_model), ""]
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
