#!/usr/bin/env python3
"""Table 2 retrained from scratch with several seeds.

Reads the per-run JSON files written by run_table2_seeds.sh and reports, per
ADC reading convention (midpoint, floor) and model, the two ADC-Quant rows
of Table 2 for every seed: Stages 1-2 (converter-on perplexity on WikiText-2
and C4 and the eight downstream accuracies of the calibrated backbone) and
Stages 1-3 (the same after the residual), their mean and sample standard
deviation, the Stage 3 gain with a 95% t confidence interval, and the
paper's Table 2 rows (floor reading, one seed) next to them. Unfinished
seeds are shown as "—".

--latex-output writes the rows of Table 2 with mean ± std cells written as
\\pmcell{mean}{std}; define the macro in the paper, for example
\\newcommand{\\pmcell}[2]{#1\\,{\\scriptsize$\\pm$#2}}. The digital reference
rows (BF16, FlatQuant without ADC) are copied from Table 2; the "ADC, no
calib." row is not part of this run.
"""

import argparse
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
    from ADC.llama.summarize_stage3_seeds import (
        avg,
        confidence_interval,
        fmt,
        fmt_pm,
        mean_std,
        present,
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
    from summarize_stage3_seeds import (
        avg,
        confidence_interval,
        fmt,
        fmt_pm,
        mean_std,
        present,
    )

CONVENTIONS = (
    ("midpoint", "Midpoint reading, (q + 0.5)·Δ"),
    ("floor", "Floor reading, q·Δ"),
)
RUN_NAME = re.compile(
    r"^(?P<key>.+)_(?P<convention>midpoint|floor)_seed(?P<seed>\d+)"
    r"(?P<noadc>_noadc)?(?P<suffix>.*)$"
)
# Table 2 digital reference rows (Wiki, C4, accuracies in TASKS order, Avg),
# from adc_aware_llm_aaai2027.tex; copied into the LaTeX output unchanged.
# Avg is the paper's value (the mean of the unrounded accuracies).
TABLE2_REFERENCE = {
    "llama32_1b": {
        "BF16 reference": (8.68, 13.13, (64.19, 32.15, 63.06, 61.83, 36.77, 74.70, 37.40, 63.91), 54.25),
        "FlatQuant, no ADC": (14.13, 22.83, (51.98, 24.86, 55.25, 60.44, 33.02, 69.48, 32.00, 62.11), 48.64),
    },
    "smollm2_17b": {
        "BF16 reference": (7.23, 11.82, (71.44, 49.99, 65.75, 73.36, 47.27, 77.42, 43.80, 72.39), 62.68),
        "FlatQuant, no ADC": (15.94, 24.52, (56.90, 28.95, 56.43, 56.94, 34.64, 69.75, 35.00, 61.74), 50.05),
    },
    "tinyllama_11b": {
        "BF16 reference": (7.01, 10.72, (55.57, 26.16, 49.17, 42.59, 30.12, 70.46, 23.80, 54.77), 44.08),
        "FlatQuant, no ADC": (10.11, 16.27, (48.97, 25.47, 52.01, 40.45, 25.43, 66.65, 26.20, 61.99), 43.40),
    },
}
# Table 2 "ADC, no calib." perplexities (floor reading), for the LaTeX comment.
TABLE2_NO_CALIB = {
    "llama32_1b": (2402.19, 2614.93),
    "smollm2_17b": (15073.82, 11190.46),
    "tinyllama_11b": (113.20, 336.00),
}
# (id, label, LaTeX label, perplexity key pattern, downstream key,
#  Table 2 downstream label, index of the (before, after) Stage 3 pair)
STAGES = (
    ("12", "ADC-Quant, Stages 1–2", "ADC-Quant, Stages 1--2",
     "ppl_adc_prelora_{ds}", "downstream_adc_ptq", "Stages 1–2, Table 2", 0),
    ("13", "ADC-Quant, Stages 1–3", "ADC-Quant, Stages 1--3",
     "ppl_adc_{ds}", "downstream_adc_lora", "Stages 1–3, Table 2", 1),
)
COLUMNS = ["Wiki", "C4", *[short for _, short in TASKS], "Avg"]
LATEX_LABEL_WIDTH = 26


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results-dir", type=Path, nargs="+", required=True,
                        help="Directories with the per-run JSON files")
    parser.add_argument("--run-suffix", default="",
                        help="Run-name suffix, e.g. _smoke for smoke runs")
    parser.add_argument("--expected-seeds", type=int, default=3,
                        help="Seeds per model the checks expect (default 3)")
    parser.add_argument("--latex-convention", choices=[name for name, _ in CONVENTIONS],
                        default="midpoint", help="Convention written to --latex-output")
    parser.add_argument("--markdown-output", type=Path, default=None)
    parser.add_argument("--latex-output", type=Path, default=None)
    return parser.parse_args()


def _matches(run_name: str, key: str, convention: str, suffix: str):
    match = RUN_NAME.match(run_name)
    if match and match["key"] == key and match["convention"] == convention \
            and match["suffix"] == suffix:
        return match
    return None


def seed_runs(runs: dict[str, dict], key: str, convention: str,
              suffix: str) -> list[tuple[int, dict]]:
    """(seed, results) of the model's finished full runs, sorted by seed."""
    found = []
    for run_name, results in runs.items():
        match = _matches(run_name, key, convention, suffix)
        if match and not match["noadc"]:
            found.append((int(match["seed"]), results))
    return sorted(found, key=lambda item: item[0])


def noadc_runs(runs: dict[str, dict], key: str, convention: str,
               suffix: str) -> dict[int, dict]:
    """seed -> results of the converter-bypassed evaluations (NO_ADC_EVAL=1)."""
    found = {}
    for run_name, results in runs.items():
        match = _matches(run_name, key, convention, suffix)
        if match and match["noadc"]:
            found[int(match["seed"])] = results
    return dict(sorted(found.items()))


def conventions_present(runs: dict[str, dict], suffix: str) -> list[str]:
    """Conventions with at least one finished run, in CONVENTIONS order."""
    return [
        convention for convention, _ in CONVENTIONS
        if any(seed_runs(runs, key, convention, suffix) or noadc_runs(runs, key, convention, suffix)
               for key, _ in MODELS)
    ]


def stage_row(results: dict, stage: tuple) -> dict:
    """Wiki, C4, the eight accuracies and their mean of one Table 2 row."""
    _, _, _, ppl_key, down_key, _, _ = stage
    tasks = downstream_values(results, down_key) or [None] * len(TASKS)
    return {
        "wiki": results.get(ppl_key.format(ds="wikitext2")),
        "c4": results.get(ppl_key.format(ds="c4")),
        "tasks": tasks,
        "avg": avg(tasks),
    }


def row_cells(row: dict) -> list[float | None]:
    return [row["wiki"], row["c4"], *row["tasks"], row["avg"]]


def stage_columns(rows: list[dict]) -> dict[str, list[float]]:
    """Present values per column over the seeds."""
    columns: dict[str, list[float]] = {name: [] for name in COLUMNS}
    for row in rows:
        for name, value in zip(COLUMNS, row_cells(row)):
            if value is not None:
                columns[name].append(value)
    return columns


def table2_row(key: str, stage: tuple) -> list[float]:
    _, _, _, _, _, table2_label, pair_index = stage
    ppl = TABLE2_STAGES12[key]
    down = list(TABLE2_DOWNSTREAM[key][table2_label])
    return [ppl["wikitext2"][pair_index], ppl["c4"][pair_index], *down, avg(down)]


def build_stage_table(key: str, stage: tuple, seeds: list[tuple[int, dict]]) -> list[str]:
    label = stage[1]
    lines = [
        f"#### {label}",
        "",
        "| Seed | " + " | ".join(COLUMNS) + " |",
        "| :-- | " + " | ".join("--:" for _ in COLUMNS) + " |",
    ]
    rows = [stage_row(results, stage) for _, results in seeds]
    for (seed, _), row in zip(seeds, rows):
        lines.append("| " + " | ".join([str(seed), *[fmt(value) for value in row_cells(row)]]) + " |")
    columns = stage_columns(rows)
    lines.append("| mean ± std | " + " | ".join(fmt_pm(columns[name]) for name in COLUMNS) + " |")
    lines.append("| Table 2, floor (seed 42) | "
                 + " | ".join(f"{value:.2f}" for value in table2_row(key, stage)) + " |")
    lines.append("")
    return lines


def build_gains(seeds: list[tuple[int, dict]]) -> list[str]:
    """Stage 3 minus Stages 1-2, paired per seed."""
    pairs = [(stage_row(results, STAGES[0]), stage_row(results, STAGES[1]))
             for _, results in seeds]
    lines = []
    for name, field in (("Wiki", "wiki"), ("C4", "c4")):
        deltas = [after[field] - before[field] for before, after in pairs
                  if before[field] is not None and after[field] is not None]
        relative = [100.0 * (after[field] - before[field]) / before[field]
                    for before, after in pairs
                    if before[field] is not None and after[field] is not None]
        if deltas:
            lines.append(f"- Stage 3 on {name}: {fmt_pm(deltas, signed=True)} "
                         f"({fmt_pm(relative, signed=True)}%; n = {len(deltas)})")
    gains = [after["avg"] - before["avg"] for before, after in pairs
             if before["avg"] is not None and after["avg"] is not None]
    if gains:
        mean, half = confidence_interval(gains)
        interval = "" if half is None else f", 95% CI [{mean - half:+.2f}, {mean + half:+.2f}]"
        lines.append(f"- Stage 3 on the downstream mean: {fmt_pm(gains, signed=True)} points"
                     f"{interval} (n = {len(gains)})")
    return lines


def build_noadc(noadc: dict[int, dict]) -> list[str]:
    if not noadc:
        return []
    per_seed = [
        f"seed {seed}: {fmt(results.get('ppl_int4_ptq_wikitext2'))} / "
        f"{fmt(results.get('ppl_int4_ptq_c4'))}"
        for seed, results in noadc.items()
    ]
    wiki = present([results.get("ppl_int4_ptq_wikitext2") for results in noadc.values()])
    c4 = present([results.get("ppl_int4_ptq_c4") for results in noadc.values()])
    return [f"- FlatQuant without ADC on this seed's transforms (Wiki / C4): "
            f"{'; '.join(per_seed)}; mean ± std {fmt_pm(wiki)} / {fmt_pm(c4)}"]


def build_model_section(key: str, label: str, seeds: list[tuple[int, dict]],
                        noadc: dict[int, dict]) -> list[str]:
    lines = [f"### {label}", ""]
    if not seeds:
        return lines + ["- no finished seeds yet", ""]
    for stage in STAGES:
        lines += build_stage_table(key, stage, seeds)
    extra = build_gains(seeds) + build_noadc(noadc)
    if extra:
        lines += extra + [""]
    return lines


def build_checks(runs: dict[str, dict], suffix: str, conventions: list[str],
                 expected_seeds: int) -> list[str]:
    checks = []
    for convention in conventions:
        for key, label in MODELS:
            seeds = seed_runs(runs, key, convention, suffix)
            if len(seeds) < expected_seeds:
                done = ", ".join(str(seed) for seed, _ in seeds) or "none"
                checks.append(f"- {label} ({convention}): {len(seeds)} of {expected_seeds} "
                              f"seeds finished ({done})")
            for seed, results in seeds:
                for stage in STAGES:
                    row = stage_row(results, stage)
                    if row["wiki"] is None or row["c4"] is None:
                        checks.append(f"- {label} ({convention}), seed {seed}: "
                                      f"no perplexity for {stage[1]}")
                    if any(value is None for value in row["tasks"]):
                        checks.append(f"- {label} ({convention}), seed {seed}: "
                                      f"no downstream results for {stage[1]}")
                before = stage_row(results, STAGES[0])
                after = stage_row(results, STAGES[1])
                worse = [
                    name for name, field in (("Wiki", "wiki"), ("C4", "c4"))
                    if before[field] is not None and after[field] is not None
                    and after[field] > before[field]
                ]
                if worse:
                    checks.append(f"- {label} ({convention}), seed {seed}: Stage 3 raised the "
                                  f"{' and '.join(worse)} perplexity")
    return checks or ["- none"]


def build_report(runs: dict[str, dict], suffix: str, expected_seeds: int = 3) -> str:
    conventions = conventions_present(runs, suffix) or ["midpoint"]
    lines = [
        "# Table 2 with seeds",
        "",
        "Every seed retrains ADC-Quant from scratch: FlatQuant Stage A, the converter-aware "
        "Stage B, ADC scale calibration, then the Stage 3 residual (rank 4, all seven "
        "projections, CE + 0.5 KL, 5 epochs). The seed sets the initialization of the "
        "transforms and adapters and, with DATA_SEED=1, the residual's 256 calibration "
        "sequences. Perplexity on WikiText-2 and C4 with the converter on, accuracy (%) from "
        "lm-evaluation-harness (MMLU 5-shot, the rest 0-shot). std is the sample standard "
        "deviation; the Table 2 rows (floor reading, one seed) are shown for comparison.",
        "",
    ]
    if suffix == "_smoke":
        lines += ["**Smoke run:** truncated corpora, one epoch on two sequences and four "
                  "examples per task. The numbers only show that the pipeline runs; they are "
                  "not comparable with Table 2.", ""]
    titles = dict(CONVENTIONS)
    for convention in conventions:
        lines += [f"## {titles[convention]}", ""]
        for key, label in MODELS:
            lines += build_model_section(
                key, label,
                seed_runs(runs, key, convention, suffix),
                noadc_runs(runs, key, convention, suffix),
            )
    lines += ["## Checks", "", *build_checks(runs, suffix, conventions, expected_seeds), ""]
    return "\n".join(lines)


def latex_number(values: list[float], bold: bool = False) -> str:
    if not values:
        return "---"
    mean, std = mean_std(values)
    cell = f"{mean:.2f}" if std is None else f"\\pmcell{{{mean:.2f}}}{{{std:.2f}}}"
    return f"\\textbf{{{cell}}}" if bold else cell


def _latex_row(label: str, cells: list[str], comment: str = "") -> str:
    row = " & ".join([f"{label:<{LATEX_LABEL_WIDTH}}", *cells]) + r" \\"
    return f"{row}  % {comment}" if comment else row


def build_latex(runs: dict[str, dict], suffix: str, convention: str) -> str:
    """Rows of Table 2 for one convention, mean ± std over the finished seeds."""
    lines = [
        f"% Table 2 rows, {convention} reading, mean ± std over seeds; "
        "digital reference rows copied from Table 2.",
        r"% \newcommand{\pmcell}[2]{#1\,{\scriptsize$\pm$#2}}",
    ]
    for index, (key, label) in enumerate(MODELS):
        if index:
            lines.append(r"\midrule")
        lines.append(f"\\multicolumn{{12}}{{@{{}}l}}{{\\emph{{{label}}}}}\\\\")
        for reference_label, (wiki, c4, accuracies, mean) in TABLE2_REFERENCE[key].items():
            cells = [f"{wiki:.2f}", f"{c4:.2f}", *[f"{value:.2f}" for value in accuracies],
                     f"{mean:.2f}"]
            lines.append(_latex_row(reference_label, cells))
        lines.append(r"\addlinespace[3pt]")
        no_calib = TABLE2_NO_CALIB[key]
        lines.append(_latex_row(
            "ADC, no calib.\\", ["---"] * len(COLUMNS),
            f"not measured with {convention} reading; Table 2 (floor): "
            f"{no_calib[0]:.2f} / {no_calib[1]:.2f}",
        ))
        seeds = seed_runs(runs, key, convention, suffix)
        if not seeds:
            lines.append(f"% {label}: no finished seeds yet")
            continue
        columns = {
            stage[0]: stage_columns([stage_row(results, stage) for _, results in seeds])
            for stage in STAGES
        }

        def best_stage(name: str, lower_is_better: bool) -> str | None:
            first, second = columns["12"][name], columns["13"][name]
            if not first or not second:
                return None
            mean_first, mean_second = mean_std(first)[0], mean_std(second)[0]
            if mean_first == mean_second:
                return None
            return "13" if (mean_second < mean_first) == lower_is_better else "12"

        best = {"Wiki": best_stage("Wiki", True), "C4": best_stage("C4", True),
                "Avg": best_stage("Avg", False)}
        for stage in STAGES:
            stage_id, _, latex_label = stage[0], stage[1], stage[2]
            cells = [latex_number(columns[stage_id][name], bold=(best.get(name) == stage_id))
                     for name in COLUMNS]
            lines.append(_latex_row(latex_label, cells, f"n = {len(seeds)}"))
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    runs = load_results(args.results_dir)
    report = build_report(runs, args.run_suffix, args.expected_seeds)
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(report, encoding="utf-8")
    if args.latex_output is not None:
        args.latex_output.parent.mkdir(parents=True, exist_ok=True)
        args.latex_output.write_text(
            build_latex(runs, args.run_suffix, args.latex_convention), encoding="utf-8",
        )
    print(report, end="")


if __name__ == "__main__":
    main()
