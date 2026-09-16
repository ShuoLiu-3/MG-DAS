import numpy as np

from mgdas_v2.certification import certify_shortlist
from mgdas_v2.search import beam_search
from mgdas_v2.types import BehaviorMetrics, InterventionAtom, InterventionConfig


def make_atom(identifier: str, layer: int) -> InterventionAtom:
    return InterventionAtom(identifier, "Age", "centroid", 1, layer, np.asarray([1.0, 0.0]))


def search_evaluator(config: InterventionConfig) -> BehaviorMetrics:
    score = {0: 10.0, 1: 4.0, 2: 3.9}[config.cardinality]
    return BehaviorMetrics(score, 0.0, 70.0, 30.0)


def certification_evaluator(config: InterventionConfig) -> BehaviorMetrics:
    score = {0: 10.0, 1: 4.0, 2: 3.9}[config.cardinality]
    return BehaviorMetrics(score, 0.0, 70.0, 30.0, mmlu=49.0)


def test_beam_search_and_near_optimal_sparsity() -> None:
    atoms = [make_atom("a", 3), make_atom("b", 4)]
    shortlist = beam_search(
        atoms=atoms,
        baseline=BehaviorMetrics(10.0, 0.0, 70.0, 30.0),
        evaluation_function=search_evaluator,
        max_units=2,
        beam_width=10,
        deployment_strengths=(0.1,),
        weight_grids={1: ((1.0,),), 2: ((0.5, 0.5),)},
        epsilon_score=0.1,
    )
    selected, certified = certify_shortlist(
        shortlist,
        BehaviorMetrics(10.0, 0.0, 70.0, 30.0, mmlu=50.0),
        certification_evaluator,
        task_tolerance_pp=5.0,
        mmlu_tolerance_pp=3.0,
        gain_tolerance=0.02,
        epsilon_score=0.1,
    )
    assert selected.config.cardinality == 1
    assert any(item.config.cardinality == 0 and item.feasible for item in certified)


def test_certification_rejects_reversal_and_capability_loss() -> None:
    atom_value = make_atom("a", 3)
    shortlist = beam_search(
        atoms=[atom_value],
        baseline=BehaviorMetrics(10.0, 0.0, 70.0, 30.0),
        evaluation_function=lambda _config: BehaviorMetrics(2.0, 0.0, 70.0, 30.0),
        max_units=1,
        beam_width=5,
        deployment_strengths=(0.1,),
        weight_grids={1: ((1.0,),)},
        epsilon_score=0.1,
    )

    def failing_evaluator(_config: InterventionConfig) -> BehaviorMetrics:
        return BehaviorMetrics(-1.0, 0.0, 64.0, 30.0, mmlu=46.0)

    selected, certified = certify_shortlist(
        shortlist,
        BehaviorMetrics(10.0, 0.0, 70.0, 30.0, mmlu=50.0),
        failing_evaluator,
        task_tolerance_pp=5.0,
        mmlu_tolerance_pp=3.0,
        gain_tolerance=0.02,
        epsilon_score=0.1,
    )
    rejected = next(item for item in certified if item.config.cardinality == 1)
    assert set(rejected.rejection_reasons) == {"s_dis_reversal", "bbq_accuracy", "mmlu"}
    assert selected.config.cardinality == 0

