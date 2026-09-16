import pytest

from mgdas_v2.metrics import compute_bbq_metrics, count_s_dis_reversals
from mgdas_v2.types import BBQExample, BehaviorMetrics


def example(identifier: str, ambiguous: bool, stereotype: int, label: int) -> BBQExample:
    return BBQExample(
        example_id=identifier,
        group_id=identifier,
        attribute="Age",
        context="context",
        question="question",
        choices=("stereotype", "anti", "unknown"),
        label_index=label,
        stereotype_index=stereotype,
        anti_stereotype_index=1 - stereotype,
        unknown_index=2,
        is_ambiguous=ambiguous,
    )


def test_bbq_metrics_follow_paper_definitions() -> None:
    examples = [
        example("d1", False, 0, 0),
        example("d2", False, 0, 1),
        example("a1", True, 0, 2),
        example("a2", True, 0, 2),
    ]
    metrics = compute_bbq_metrics(examples, [0, 0, 2, 0])
    assert metrics.acc_dis == pytest.approx(50.0)
    assert metrics.s_dis == pytest.approx(100.0)
    assert metrics.acc_amb == pytest.approx(50.0)
    assert metrics.s_amb == pytest.approx(50.0)


def test_reversal_count_is_attribute_level() -> None:
    baseline = {
        "Age": BehaviorMetrics(4, 0, 60, 30),
        "Gender": BehaviorMetrics(-3, 0, 60, 30),
    }
    changed = {
        "Age": BehaviorMetrics(-1, 0, 60, 30),
        "Gender": BehaviorMetrics(-1, 0, 60, 30),
    }
    assert count_s_dis_reversals(baseline, changed) == 1

