"""Tests for the Stage 3 seed-variance tooling."""

import math

import pytest
import torch

from ADC.llama import summarize_stage3_seeds as seeds
from ADC.llama.runs import llama_smooth_quant_adc_ptq as runner


def _pool(size: int = 20) -> list[dict]:
    return [{"input_ids": [i] * 4, "attention_mask": [1] * 4} for i in range(size)]


def _collate(features: list[dict]) -> dict:
    return {
        "input_ids": torch.tensor([f["input_ids"] for f in features]),
        "attention_mask": torch.tensor([f["attention_mask"] for f in features]),
    }


def _order(loader) -> list[int]:
    return [int(index) for batch in loader for index in batch["input_ids"][:, 0]]


def test_lora_dataloader_without_a_seed_keeps_the_corpus_order() -> None:
    loader = runner.build_lora_dataloader(_pool(), 4, _collate, None)

    assert _order(loader) == list(range(20))
    assert next(iter(loader))["input_ids"].shape == (4, 4)


def test_lora_dataloader_seed_permutes_the_pool_once() -> None:
    loader = runner.build_lora_dataloader(_pool(), 4, _collate, 7)
    order = _order(loader)

    assert sorted(order) == list(range(20))
    assert order != list(range(20))
    # Every epoch sees the same sequences in the same order.
    assert _order(loader) == order
    assert _order(runner.build_lora_dataloader(_pool(), 4, _collate, 7)) == order
    assert _order(runner.build_lora_dataloader(_pool(), 4, _collate, 8)) != order


def test_lora_dataloader_seed_leaves_the_global_generator_alone() -> None:
    torch.manual_seed(0)
    expected = torch.rand(3)
    torch.manual_seed(0)
    runner.build_lora_dataloader(_pool(), 4, _collate, 7)

    torch.testing.assert_close(torch.rand(3), expected)


def test_mean_std_and_confidence_interval() -> None:
    assert seeds.mean_std([5.0]) == (5.0, None)
    mean, std = seeds.mean_std([1.0, 2.0, 3.0])
    assert mean == 2.0
    assert std == pytest.approx(1.0)

    center, half = seeds.confidence_interval([1.0, 2.0, 3.0])
    assert center == 2.0
    assert half == pytest.approx(4.303 / math.sqrt(3))
    assert seeds.confidence_interval([1.0]) == (1.0, None)


def _record(wiki: float, c4: float, accuracies: list[float],
            pre_accuracies: list[float] | None = None) -> dict:
    results = {
        "ppl_adc_prelora_wikitext2": 11.62,
        "ppl_adc_prelora_c4": 20.23,
        "ppl_adc_wikitext2": wiki,
        "ppl_adc_c4": c4,
        "downstream_adc_lora": {
            task: {"accuracy_pct": value} for (task, _), value in zip(seeds.TASKS, accuracies)
        },
    }
    if pre_accuracies is not None:
        results["downstream_adc_ptq"] = {
            task: {"accuracy_pct": value} for (task, _), value in zip(seeds.TASKS, pre_accuracies)
        }
    return results


def test_report_lists_every_seed_their_mean_and_the_table2_row() -> None:
    pre = [40.0] * 8
    runs = {
        "tinyllama_11b_stage3_seed42": _record(10.9, 18.7, [41.0] * 8, pre_accuracies=pre),
        "tinyllama_11b_stage3_seed43": _record(10.8, 18.6, [42.0] * 8),
        "tinyllama_11b_stage3_seed44": _record(11.0, 18.8, [43.0] * 8),
        # Another experiment in the same directory is ignored.
        "tinyllama_11b_no_calib_stage3": _record(500.0, 700.0, [30.0] * 8),
    }

    report = seeds.build_report(runs, suffix="")

    assert "## TinyLlama-1.1B" in report
    assert "| 42 | 10.90 | 18.70 |" in report
    assert "| 44 | 11.00 | 18.80 |" in report
    assert "| 43 |" in report and "500.00" not in report
    # Downstream gain over the measured backbone (40.0): 1, 2, 3 -> 2.00 ± 1.00.
    assert "Stage 3 over Stages 1–2: +2.00 ± 1.00, 95% CI [-0.48, +4.48] (n = 3" in report
    assert "| Table 2 (seed 42) | 10.90 | 18.69 |" in report
    assert "## SmolLM2-1.7B\n\n- no finished seeds yet" in report
    assert "## Checks\n\n- none" in report


def test_report_falls_back_to_the_table2_backbone_and_flags_it() -> None:
    runs = {"tinyllama_11b_stage3_seed42": _record(10.9, 18.7, [42.0] * 8)}

    report = seeds.build_report(runs, suffix="")

    table2_pre_avg = sum(seeds.TABLE2_DOWNSTREAM["tinyllama_11b"]["Stages 1–2, Table 2"]) / 8
    assert f"| 42 | 10.90 | 18.70 |" in report
    assert f"{42.0 - table2_pre_avg:+.2f}" in report
    assert "downstream was not measured; Δ Avg uses the Table 2 row" in report


def test_smoke_suffix_selects_only_smoke_runs() -> None:
    runs = {
        "tinyllama_11b_stage3_seed42_smoke": _record(1.0, 2.0, [50.0] * 8),
        "tinyllama_11b_stage3_seed42": _record(10.9, 18.7, [42.0] * 8),
    }

    assert [seed for seed, _ in seeds.seed_runs(runs, "tinyllama_11b", "_smoke")] == [42]
    assert "| 42 | 1.00 | 2.00 |" in seeds.build_report(runs, suffix="_smoke")
    assert "| 42 | 10.90 | 18.70 |" in seeds.build_report(runs, suffix="")
