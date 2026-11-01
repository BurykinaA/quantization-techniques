"""Tests for the Table 2 seed runs: midpoint reading in the FlatQuant
calibration loss and the summary of run_table2_seeds.sh."""

import pytest
import torch
import torch.nn as nn

from ADC.llama import summarize_table2_seeds as table2
from ADC.llama.core.flat_quant import FlatQuantLinear


def _flat_linear(linear: nn.Linear, k: int, **extra) -> FlatQuantLinear:
    config = {"bx": 4, "bw": 4, "ba": 8, "k": k, "mvm_limit": 4,
              "signed_activations": True, **extra}
    return FlatQuantLinear(linear, w_bits=4, a_bits=4, lwc=False, lac=False, adc_config=config)


# k=1 leaves most codes inside the ADC range, k=16 saturates most of them.
@pytest.mark.parametrize("k", [1, 16])
def test_midpoint_in_the_calibration_loss_shifts_only_unsaturated_codes(k: int) -> None:
    torch.manual_seed(5)
    linear = nn.Linear(8, 6, bias=False)
    layer = _flat_linear(linear, k)
    inputs = torch.randn(5, 8)

    with torch.no_grad():
        floor_output = layer.train_forward(inputs)
        layer._adc_config["midpoint"] = True
        midpoint_output = layer.train_forward(inputs)

    tile_in, na, pa = 4, -128, 127
    delta = 2.0 * tile_in * 7 * 7 / (256 * k)
    weight = linear.weight.detach().float()
    expected = torch.zeros_like(floor_output)
    for i in range(8 // tile_in):
        xi = inputs[:, i * tile_in:(i + 1) * tile_in]
        wi = weight[:, i * tile_in:(i + 1) * tile_in]
        s_w = wi.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 7.0
        s_x = xi.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 7.0
        code_w = torch.round(wi / s_w).clamp(-8, 7)
        code_x = torch.round(xi / s_x).clamp(-8, 7)
        codes = torch.floor((code_x @ code_w.T) / delta).clamp(na, pa)
        inside = ((codes > na) & (codes < pa)).float()
        expected += 0.5 * delta * inside * s_x * s_w.T

    torch.testing.assert_close(midpoint_output - floor_output, expected, rtol=1e-4, atol=1e-5)


def test_midpoint_does_not_change_the_converter_free_calibration_loss() -> None:
    torch.manual_seed(5)
    layer = _flat_linear(nn.Linear(8, 6, bias=False), k=16, bypass_floor_ste=True)
    inputs = torch.randn(5, 8)

    with torch.no_grad():
        expected = layer.train_forward(inputs)
        layer._adc_config["midpoint"] = True
        actual = layer.train_forward(inputs)

    torch.testing.assert_close(actual, expected)


def _results(pre_wiki: float, pre_c4: float, pre_acc: float,
             wiki: float, c4: float, acc: float) -> dict:
    return {
        "ppl_adc_prelora_wikitext2": pre_wiki,
        "ppl_adc_prelora_c4": pre_c4,
        "downstream_adc_ptq": {task: {"accuracy_pct": pre_acc} for task, _ in table2.TASKS},
        "ppl_adc_wikitext2": wiki,
        "ppl_adc_c4": c4,
        "downstream_adc_lora": {task: {"accuracy_pct": acc} for task, _ in table2.TASKS},
    }


RUNS = {
    "tinyllama_11b_midpoint_seed42": _results(10.2, 18.0, 41.0, 10.0, 17.5, 43.0),
    "tinyllama_11b_midpoint_seed43": _results(10.0, 17.8, 42.0, 9.8, 17.3, 44.0),
    "tinyllama_11b_midpoint_seed44": _results(10.4, 18.2, 43.0, 10.2, 17.7, 45.0),
    "tinyllama_11b_midpoint_seed42_noadc": {
        "ppl_int4_ptq_wikitext2": 10.1, "ppl_int4_ptq_c4": 16.3,
    },
    "tinyllama_11b_floor_seed42": _results(11.6, 20.2, 41.8, 10.9, 18.7, 42.2),
    "tinyllama_11b_midpoint_seed42_smoke": _results(1.0, 2.0, 50.0, 1.0, 2.0, 50.0),
    # Another experiment in the same directory is ignored.
    "tinyllama_11b_stage3_seed42": _results(500.0, 700.0, 30.0, 400.0, 600.0, 31.0),
}


def test_seed_runs_select_one_convention_and_suffix() -> None:
    assert [s for s, _ in table2.seed_runs(RUNS, "tinyllama_11b", "midpoint", "")] == [42, 43, 44]
    assert [s for s, _ in table2.seed_runs(RUNS, "tinyllama_11b", "floor", "")] == [42]
    assert [s for s, _ in table2.seed_runs(RUNS, "tinyllama_11b", "midpoint", "_smoke")] == [42]
    assert list(table2.noadc_runs(RUNS, "tinyllama_11b", "midpoint", "")) == [42]
    assert table2.seed_runs(RUNS, "smollm2_17b", "midpoint", "") == []
    assert table2.conventions_present(RUNS, "") == ["midpoint", "floor"]
    assert table2.conventions_present(RUNS, "_smoke") == ["midpoint"]


def test_report_lists_both_rows_per_seed_their_mean_and_the_table2_rows() -> None:
    report = table2.build_report(RUNS, suffix="")

    assert "## Midpoint reading" in report and "## Floor reading" in report
    assert "#### ADC-Quant, Stages 1–2" in report and "#### ADC-Quant, Stages 1–3" in report
    assert "| 42 | 10.20 | 18.00 | 41.00 |" in report
    assert "| 42 | 10.00 | 17.50 | 43.00 |" in report
    assert "| mean ± std | 10.20 ± 0.20 | 18.00 ± 0.20 | 42.00 ± 1.00 |" in report
    assert "| mean ± std | 10.00 ± 0.20 | 17.50 ± 0.20 | 44.00 ± 1.00 |" in report
    assert "| Table 2, floor (seed 42) | 11.62 | 20.23 | 46.74 |" in report
    assert "| Table 2, floor (seed 42) | 10.90 | 18.69 | 46.68 |" in report
    assert "- Stage 3 on Wiki: -0.20 ± 0.00 (" in report and "n = 3)" in report
    assert ("- Stage 3 on the downstream mean: +2.00 ± 0.00 points, "
            "95% CI [+2.00, +2.00] (n = 3)") in report
    assert ("- FlatQuant without ADC on this seed's transforms (Wiki / C4): "
            "seed 42: 10.10 / 16.30; mean ± std 10.10 / 16.30") in report
    # The floor run is its own section with a single seed.
    assert "| 42 | 11.60 | 20.20 | 41.80 |" in report
    assert "### SmolLM2-1.7B\n\n- no finished seeds yet" in report
    assert "- SmolLM2-1.7B (midpoint): 0 of 3 seeds finished (none)" in report
    assert "- TinyLlama-1.1B (floor): 1 of 3 seeds finished (42)" in report
    assert "500.00" not in report and "| 42 | 1.00 | 2.00 |" not in report


def test_smoke_suffix_selects_only_smoke_runs() -> None:
    report = table2.build_report(RUNS, suffix="_smoke")

    assert "**Smoke run:**" in report
    assert "| 42 | 1.00 | 2.00 | 50.00 |" in report
    assert "| 42 | 10.20 | 18.00 |" not in report


def test_checks_flag_missing_downstream_and_a_stage3_that_hurts() -> None:
    results = _results(10.2, 18.0, 41.0, 10.5, 17.5, 43.0)
    del results["downstream_adc_lora"]

    report = table2.build_report({"llama32_1b_midpoint_seed42": results}, suffix="",
                                 expected_seeds=1)

    assert ("- Llama-3.2-1B (midpoint), seed 42: no downstream results for "
            "ADC-Quant, Stages 1–3") in report
    assert "- Llama-3.2-1B (midpoint), seed 42: Stage 3 raised the Wiki perplexity" in report
    assert "- Llama-3.2-1B (midpoint): 0 of 1" not in report
    assert "- SmolLM2-1.7B (midpoint): 0 of 1 seeds finished (none)" in report


def test_latex_rows_use_pmcell_and_bold_the_better_converter_row() -> None:
    latex = table2.build_latex(RUNS, suffix="", convention="midpoint")

    assert r"% \newcommand{\pmcell}[2]{#1\,{\scriptsize$\pm$#2}}" in latex
    assert r"\multicolumn{12}{@{}l}{\emph{TinyLlama-1.1B}}\\" in latex
    assert f"{'BF16 reference':<26} & 7.01 & 10.72 & 55.57 & 26.16" in latex
    assert f"{'FlatQuant, no ADC':<26} & 10.11 & 16.27 & 48.97" in latex
    assert "Table 2 (floor): 113.20 / 336.00" in latex
    assert (f"{'ADC-Quant, Stages 1--2':<26} & " + r"\pmcell{10.20}{0.20} & \pmcell{18.00}{0.20} "
            r"& \pmcell{42.00}{1.00}") in latex
    assert (f"{'ADC-Quant, Stages 1--3':<26} & " + r"\textbf{\pmcell{10.00}{0.20}} "
            r"& \textbf{\pmcell{17.50}{0.20}} & \pmcell{44.00}{1.00}") in latex
    assert r"& \textbf{\pmcell{44.00}{1.00}} \\  % n = 3" in latex
    assert "% Llama-3.2-1B: no finished seeds yet" in latex
    assert "% SmolLM2-1.7B: no finished seeds yet" in latex
    assert latex.count(r"\midrule") == 2

    floor = table2.build_latex(RUNS, suffix="", convention="floor")
    assert (f"{'ADC-Quant, Stages 1--3':<26} & " + r"\textbf{10.90} & \textbf{18.70} & 42.20") in floor
    assert r"& \textbf{42.20} \\  % n = 1" in floor
