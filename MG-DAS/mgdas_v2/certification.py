from __future__ import annotations

from collections.abc import Callable

from .search import normalized_gain
from .types import BehaviorMetrics, EvaluatedConfig, InterventionConfig


CertificationEvaluator = Callable[[InterventionConfig], BehaviorMetrics]


def certify_shortlist(
    shortlist: list[EvaluatedConfig],
    baseline: BehaviorMetrics,
    evaluator: CertificationEvaluator,
    task_tolerance_pp: float,
    mmlu_tolerance_pp: float,
    gain_tolerance: float,
    epsilon_score: float,
) -> tuple[EvaluatedConfig, list[EvaluatedConfig]]:
    if baseline.mmlu is None:
        raise ValueError("certification baseline must include MMLU")
    certified = []
    unique_configs = {item.config.identifier: item.config for item in shortlist}
    unique_configs[InterventionConfig.zero().identifier] = InterventionConfig.zero()
    for config in unique_configs.values():
        metrics = baseline if config.cardinality == 0 else evaluator(config)
        metrics.validate()
        if metrics.mmlu is None:
            raise ValueError("every certification result must include MMLU")
        reasons = []
        if abs(metrics.s_dis) > abs(baseline.s_dis) + 1e-12:
            reasons.append("bias_magnitude")
        if baseline.s_dis * metrics.s_dis < -1e-12:
            reasons.append("s_dis_reversal")
        if metrics.acc_dis - baseline.acc_dis < -task_tolerance_pp - 1e-12:
            reasons.append("bbq_accuracy")
        if metrics.mmlu - baseline.mmlu < -mmlu_tolerance_pp - 1e-12:
            reasons.append("mmlu")
        certified.append(
            EvaluatedConfig(
                config=config,
                metrics=metrics,
                gain=normalized_gain(baseline.s_dis, metrics.s_dis, epsilon_score),
                delta_acc_dis=metrics.acc_dis - baseline.acc_dis,
                direction_violation=max(0.0, -baseline.s_dis * metrics.s_dis),
                feasible=not reasons,
                rejection_reasons=tuple(reasons),
            )
        )
    feasible = [item for item in certified if item.feasible]
    if not feasible:
        raise RuntimeError("zero intervention must make the feasible set non-empty")
    best_gain = max(item.gain for item in feasible)
    near_optimal = [item for item in feasible if item.gain >= best_gain - gain_tolerance]
    selected = min(
        near_optimal,
        key=lambda item: (
            item.config.cardinality,
            -item.gain,
            -item.delta_acc_dis,
            item.config.identifier,
        ),
    )
    return selected, certified

