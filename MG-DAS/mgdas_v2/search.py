from __future__ import annotations

from collections.abc import Callable
from .types import BehaviorMetrics, EvaluatedConfig, InterventionAtom, InterventionConfig


EvaluationFunction = Callable[[InterventionConfig], BehaviorMetrics]


def normalized_gain(baseline_score: float, intervention_score: float, epsilon: float) -> float:
    return (abs(baseline_score) - abs(intervention_score)) / max(abs(baseline_score), epsilon)


def evaluate_search_config(
    config: InterventionConfig,
    baseline: BehaviorMetrics,
    evaluation_function: EvaluationFunction,
    epsilon_score: float,
) -> EvaluatedConfig:
    metrics = evaluation_function(config)
    metrics.validate()
    return EvaluatedConfig(
        config=config,
        metrics=metrics,
        gain=normalized_gain(baseline.s_dis, metrics.s_dis, epsilon_score),
        delta_acc_dis=metrics.acc_dis - baseline.acc_dis,
        direction_violation=max(0.0, -baseline.s_dis * metrics.s_dis),
    )


def _dominates(left: EvaluatedConfig, right: EvaluatedConfig, tolerance: float = 1e-12) -> bool:
    left_values = (left.gain, left.delta_acc_dis, -left.direction_violation)
    right_values = (right.gain, right.delta_acc_dis, -right.direction_violation)
    no_worse = all(lvalue >= rvalue - tolerance for lvalue, rvalue in zip(left_values, right_values))
    strictly_better = any(lvalue > rvalue + tolerance for lvalue, rvalue in zip(left_values, right_values))
    return no_worse and strictly_better


def pareto_select(candidates: list[EvaluatedConfig], beam_width: int) -> list[EvaluatedConfig]:
    frontier = [
        candidate
        for candidate in candidates
        if not any(_dominates(other, candidate) for other in candidates if other is not candidate)
    ]
    frontier.sort(
        key=lambda item: (
            -item.gain,
            -item.delta_acc_dis,
            item.direction_violation,
            item.config.cardinality,
            item.config.identifier,
        )
    )
    return frontier[:beam_width]


def _legal_support(atoms: tuple[InterventionAtom, ...]) -> bool:
    return len({atom.layer for atom in atoms}) == len(atoms)


def beam_search(
    atoms: list[InterventionAtom],
    baseline: BehaviorMetrics,
    evaluation_function: EvaluationFunction,
    max_units: int,
    beam_width: int,
    deployment_strengths: tuple[float, ...],
    weight_grids: dict[int, tuple[tuple[float, ...], ...]],
    epsilon_score: float,
) -> list[EvaluatedConfig]:
    if not atoms:
        return [
            EvaluatedConfig(
                config=InterventionConfig.zero(),
                metrics=baseline,
                gain=0.0,
                delta_acc_dis=0.0,
                direction_violation=0.0,
            )
        ]
    atom_lookup = {atom.atom_id: atom for atom in atoms}
    if len(atom_lookup) != len(atoms):
        raise ValueError("atom identifiers must be unique")
    cache: dict[str, EvaluatedConfig] = {}

    def evaluate(config: InterventionConfig) -> EvaluatedConfig:
        if config.identifier not in cache:
            cache[config.identifier] = evaluate_search_config(
                config, baseline, evaluation_function, epsilon_score
            )
        return cache[config.identifier]

    all_beams = []
    supports = [(atom,) for atom in sorted(atoms, key=lambda item: item.atom_id)]
    for cardinality in range(1, max_units + 1):
        candidates = []
        for support in supports:
            canonical_support = tuple(sorted(support, key=lambda item: item.atom_id))
            for weights in weight_grids[cardinality]:
                for strength in deployment_strengths:
                    candidates.append(evaluate(InterventionConfig(canonical_support, weights, strength)))
        beam = pareto_select(candidates, beam_width)
        all_beams.extend(beam)
        if cardinality == max_units:
            break
        next_supports = {}
        for evaluated in beam:
            selected_ids = {atom.atom_id for atom in evaluated.config.atoms}
            for atom in atoms:
                if atom.atom_id in selected_ids:
                    continue
                support = tuple(sorted((*evaluated.config.atoms, atom), key=lambda item: item.atom_id))
                if not _legal_support(support):
                    continue
                key = tuple(item.atom_id for item in support)
                next_supports[key] = support
        supports = list(next_supports.values())
        if not supports:
            break
    zero = EvaluatedConfig(
        config=InterventionConfig.zero(),
        metrics=baseline,
        gain=0.0,
        delta_acc_dis=0.0,
        direction_violation=0.0,
    )
    return [zero, *all_beams]


def select_without_certification(shortlist: list[EvaluatedConfig]) -> EvaluatedConfig:
    non_zero = [candidate for candidate in shortlist if candidate.config.cardinality > 0]
    if not non_zero:
        return shortlist[0]
    return min(
        non_zero,
        key=lambda item: (
            -item.gain,
            -item.delta_acc_dis,
            item.direction_violation,
            item.config.identifier,
        ),
    )
