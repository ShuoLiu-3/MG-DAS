from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


VARIANT_ORDER = (
    "Original",
    "Probe-ranked, Kmax=1",
    "Response-ranked, Kmax=1",
    "MG-DAS w/o certification",
    "Full MG-DAS, Kmax=3",
)


def aggregate_ablation(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    attributes = sorted({str(record["attribute"]) for record in records})
    if len(attributes) != 9:
        raise ValueError(f"core ablation requires nine attributes, found {len(attributes)}")
    baseline_rows = [record for record in records if record["variant"] == "Original"]
    if len(baseline_rows) != 9:
        raise ValueError(f"Original: expected nine attribute rows, found {len(baseline_rows)}")
    baseline_by_attribute = {row["attribute"]: row for row in baseline_rows}
    if set(baseline_by_attribute) != set(attributes):
        raise ValueError("Original must contain one row for every attribute")

    output = []
    for variant in VARIANT_ORDER:
        rows = [record for record in records if record["variant"] == variant]
        if len(rows) != 9:
            raise ValueError(f"{variant}: expected nine attribute rows, found {len(rows)}")
        rows.sort(key=lambda item: item["attribute"])
        if {row["attribute"] for row in rows} != set(attributes):
            raise ValueError(f"{variant}: duplicate or missing attribute rows")
        cardinalities = [int(row["selected_k"]) for row in rows]
        if variant == "Original" and any(cardinalities):
            raise ValueError("Original must have selected_k=0 for every attribute")
        if "Kmax=1" in variant and any(value not in {0, 1} for value in cardinalities):
            raise ValueError(f"{variant}: selected_k must be zero or one")
        if variant in {"MG-DAS w/o certification", "Full MG-DAS, Kmax=3"} and any(
            value < 0 or value > 3 for value in cardinalities
        ):
            raise ValueError(f"{variant}: selected_k must be in [0, 3]")
        reversals = 0
        if variant != "Original":
            for row in rows:
                baseline_score = float(baseline_by_attribute[row["attribute"]]["s_dis"])
                reversals += int(baseline_score * float(row["s_dis"]) < 0)
        output.append(
            {
                "Method": variant,
                "Mean K": float(np.mean(cardinalities)),
                "Mean |sDIS|": float(np.mean([abs(row["s_dis"]) for row in rows])),
                "Mean |sAMB|": float(np.mean([abs(row["s_amb"]) for row in rows])),
                "sDIS Rev.": "—" if variant == "Original" else f"{reversals}/9",
                "BBQ AccDIS": float(np.mean([row["acc_dis"] for row in rows])),
                "MMLU": float(np.mean([row["mmlu"] for row in rows])),
            }
        )
    return output


def write_ablation_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty ablation table")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
