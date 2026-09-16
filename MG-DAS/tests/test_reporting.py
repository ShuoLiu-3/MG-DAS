import pytest

from mgdas_v2.reporting import VARIANT_ORDER, aggregate_ablation


def test_ablation_aggregation_uses_attribute_macro_means() -> None:
    records = []
    for attribute_index in range(9):
        for variant in VARIANT_ORDER:
            selected_k = 0
            if "Kmax=1" in variant:
                selected_k = 1
            elif variant == "MG-DAS w/o certification":
                selected_k = 3
            elif variant == "Full MG-DAS, Kmax=3":
                selected_k = 3 if attribute_index < 2 else 2
            records.append(
                {
                    "attribute": f"attribute-{attribute_index}",
                    "variant": variant,
                    "selected_k": selected_k,
                    "s_dis": -1.0 if variant != "Original" and attribute_index == 0 else 2.0,
                    "s_amb": 1.0,
                    "acc_dis": 60.0 + attribute_index,
                    "mmlu": 47.0,
                }
            )
    rows = aggregate_ablation(records)
    full = next(row for row in rows if row["Method"] == "Full MG-DAS, Kmax=3")
    assert full["Mean K"] == pytest.approx(20 / 9)
    assert full["BBQ AccDIS"] == pytest.approx(64.0)
    assert full["sDIS Rev."] == "1/9"

