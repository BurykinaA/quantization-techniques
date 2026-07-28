#!/usr/bin/env python3
"""Report FlatQuant diagonal ranges from transform checkpoints or run logs.

Two sources are supported:

``*.pt``
    Checkpoints written by ``save_flat_transforms``. These are saved before
    ``reparameterize_model`` runs, so they still contain the per-channel
    scaling diagonal (``diag_scale``) and both Kronecker factor diagonals
    (``diag_left`` / ``diag_right``) for every layer.

``*.log`` / ``*.txt``
    Calibration logs. ``calibrate_flat_quant`` prints one min/max/mean line per
    layer and diagonal. When a log covers several calibration passes over the
    same layer, the last line for that layer wins, which is the trained state.
"""

import argparse
import re
from pathlib import Path

import torch


# Clamp bounds enforced by _project_flatquant_parameters during calibration.
BOUNDS = {
    "diag_scale": (1e-4, 10.0),
    "diag_left": (0.1, 10.0),
    "diag_right": (0.1, 10.0),
}

# Checkpoint state-dict key and log parameter prefix for each transform group.
GROUPS = [
    ("attn_ln_trans", "self_attn.ln_trans", "attention q/k/v"),
    ("mlp_up_gate_trans", "mlp.up_gate_trans", "MLP gate/up"),
    ("mlp_down_trans", "mlp.down_trans", "MLP down"),
]

PARAMS = ["diag_scale", "diag_left", "diag_right"]

# Relative tolerance for calling a channel "sitting at the clamp bound".
BOUND_TOLERANCE = 1e-3

LOG_PATTERN = re.compile(
    r"layer\s+(\d+)\s+\[([\w.]+)\]\s+"
    r"min=([-\d.eE+]+)\s+max=([-\d.eE+]+)\s+mean=([-\d.eE+]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        type=Path,
        nargs="+",
        help="Checkpoints, logs, or directories scanned for both",
    )
    parser.add_argument(
        "--per-layer",
        action="store_true",
        help="Also print one row per layer and parameter group",
    )
    return parser.parse_args()


def resolve_inputs(paths: list[Path]) -> list[Path]:
    resolved: list[Path] = []
    for path in paths:
        if path.is_dir():
            for pattern in ("flat_quant_*.pt", "*.log", "*.txt"):
                resolved.extend(sorted(path.rglob(pattern)))
        else:
            resolved.append(path)
    return resolved


def records_from_checkpoint(path: Path) -> tuple[list[dict], dict]:
    """Read exact per-channel diagonals from a transform checkpoint."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected checkpoint payload in {path}")

    metadata = payload.get("_metadata")
    records: list[dict] = []

    for layer_index, layer_state in sorted(
        (key, value) for key, value in payload.items() if isinstance(key, int)
    ):
        if not isinstance(layer_state, dict):
            continue
        for state_key, _, label in GROUPS:
            group_state = layer_state.get(state_key)
            if not isinstance(group_state, dict):
                continue
            for param in PARAMS:
                tensor = group_state.get(param)
                if not isinstance(tensor, torch.Tensor):
                    continue
                values = tensor.detach().float().abs().flatten()
                lower, upper = BOUNDS[param]
                records.append({
                    "group": label,
                    "param": param,
                    "layer": layer_index,
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "mean": float(values.mean()),
                    "channels": values.numel(),
                    "at_lower": int((values <= lower * (1.0 + BOUND_TOLERANCE)).sum()),
                    "at_upper": int((values >= upper * (1.0 - BOUND_TOLERANCE)).sum()),
                })

    return records, metadata if isinstance(metadata, dict) else {}


def records_from_log(path: Path) -> tuple[list[dict], int]:
    """Read per-layer diagonal ranges printed during calibration."""
    label_by_prefix = {prefix: label for _, prefix, label in GROUPS}
    latest: dict[tuple[int, str], dict] = {}
    matches = 0

    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = LOG_PATTERN.search(line)
            if match is None:
                continue
            matches += 1
            layer_index = int(match.group(1))
            qualified_name = match.group(2)
            prefix, _, param = qualified_name.rpartition(".")
            if param not in BOUNDS:
                continue
            label = label_by_prefix.get(prefix)
            if label is None:
                continue
            # A later line for the same layer supersedes an earlier pass.
            latest[(layer_index, qualified_name)] = {
                "group": label,
                "param": param,
                "layer": layer_index,
                "min": float(match.group(3)),
                "max": float(match.group(4)),
                "mean": float(match.group(5)),
                "channels": None,
                "at_lower": None,
                "at_upper": None,
            }

    return list(latest.values()), matches


def aggregate(records: list[dict]) -> dict[tuple[str, str], dict]:
    summary: dict[tuple[str, str], dict] = {}
    for record in records:
        key = (record["group"], record["param"])
        entry = summary.setdefault(key, {
            "min": float("inf"),
            "max": float("-inf"),
            "mean_sum": 0.0,
            "layers": 0,
            "argmax_layer": None,
            "at_lower": 0,
            "at_upper": 0,
            "exact_counts": True,
            "per_layer": [],
        })
        if record["min"] < entry["min"]:
            entry["min"] = record["min"]
        if record["max"] > entry["max"]:
            entry["max"] = record["max"]
            entry["argmax_layer"] = record["layer"]
        entry["mean_sum"] += record["mean"]
        entry["layers"] += 1
        if record["at_lower"] is None:
            entry["exact_counts"] = False
        else:
            entry["at_lower"] += record["at_lower"]
            entry["at_upper"] += record["at_upper"]
        entry["per_layer"].append(record)

    for entry in summary.values():
        entry["per_layer"].sort(key=lambda record: record["layer"])
    return summary


def format_report(path: Path, records: list[dict], note: str, per_layer: bool) -> str:
    lines = [f"== {path}"]
    if note:
        lines.append(f"   {note}")

    summary = aggregate(records)
    if not summary:
        lines.append("   no diagonal parameters found")
        return "\n".join(lines)

    lines.append(
        f"   {'group':<18} {'param':<11} {'min':>10} {'max':>10} {'mean':>10}"
        f" {'@lo':>7} {'@hi':>7}  bound"
    )
    for _, _, label in GROUPS:
        for param in PARAMS:
            entry = summary.get((label, param))
            if entry is None:
                continue
            lower, upper = BOUNDS[param]
            at_lower = str(entry["at_lower"]) if entry["exact_counts"] else "-"
            at_upper = str(entry["at_upper"]) if entry["exact_counts"] else "-"
            lines.append(
                f"   {label:<18} {param:<11}"
                f" {entry['min']:>10.4f} {entry['max']:>10.4f}"
                f" {entry['mean_sum'] / entry['layers']:>10.4f}"
                f" {at_lower:>7} {at_upper:>7}"
                f"  [{lower:g}, {upper:g}]"
                f"  ({entry['layers']} layers, max in layer {entry['argmax_layer']})"
            )

    scale = [entry for (_, param), entry in summary.items() if param == "diag_scale"]
    kron = [entry for (_, param), entry in summary.items() if param != "diag_scale"]
    if scale:
        lines.append(
            f"   Gamma (diag_scale) overall: [{min(e['min'] for e in scale):.4f},"
            f" {max(e['max'] for e in scale):.4f}]"
        )
    if kron:
        lines.append(
            f"   Kronecker diagonals overall: [{min(e['min'] for e in kron):.4f},"
            f" {max(e['max'] for e in kron):.4f}]"
        )

    if per_layer:
        for _, _, label in GROUPS:
            for param in PARAMS:
                entry = summary.get((label, param))
                if entry is None:
                    continue
                lines.append(f"   -- {label} / {param}")
                for record in entry["per_layer"]:
                    lines.append(
                        f"      layer {record['layer']:>3}"
                        f"  min={record['min']:.4f}  max={record['max']:.4f}"
                    )

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    inputs = resolve_inputs(args.paths)
    if not inputs:
        raise SystemExit("No checkpoints or logs found")

    for path in inputs:
        if path.suffix == ".pt":
            records, metadata = records_from_checkpoint(path)
            note = ""
            if metadata:
                note = "metadata: " + ", ".join(
                    f"{key}={value}" for key, value in sorted(metadata.items())
                )
        else:
            records, matches = records_from_log(path)
            if not matches:
                continue
            note = f"parsed {matches} diagonal log lines (last pass per layer wins)"

        print(format_report(path, records, note, args.per_layer))
        print()


if __name__ == "__main__":
    main()
