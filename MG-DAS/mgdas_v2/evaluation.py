from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .dataio import MultipleChoiceExample
from .metrics import compute_bbq_metrics
from .modeling import multiple_choice_accuracy, predictions_from_scores
from .types import BBQExample, BehaviorMetrics, InterventionConfig


BBQScorer = Callable[[list[BBQExample], InterventionConfig], np.ndarray]
MultipleChoiceScorer = Callable[[list[MultipleChoiceExample], InterventionConfig], np.ndarray]


def evaluate_bbq(
    examples: list[BBQExample], config: InterventionConfig, scorer: BBQScorer
) -> BehaviorMetrics:
    scores = scorer(examples, config)
    return compute_bbq_metrics(examples, predictions_from_scores(scores))


def evaluate_bbq_and_mmlu(
    bbq_examples: list[BBQExample],
    mmlu_examples: list[MultipleChoiceExample],
    config: InterventionConfig,
    bbq_scorer: BBQScorer,
    mmlu_scorer: MultipleChoiceScorer,
) -> BehaviorMetrics:
    bbq_metrics = evaluate_bbq(bbq_examples, config, bbq_scorer)
    mmlu_scores = mmlu_scorer(mmlu_examples, config)
    return BehaviorMetrics(
        s_dis=bbq_metrics.s_dis,
        s_amb=bbq_metrics.s_amb,
        acc_dis=bbq_metrics.acc_dis,
        acc_amb=bbq_metrics.acc_amb,
        mmlu=multiple_choice_accuracy(mmlu_scores, mmlu_examples),
        n_dis=bbq_metrics.n_dis,
        n_amb=bbq_metrics.n_amb,
    )


def attach_mmlu(metrics: BehaviorMetrics, mmlu: float) -> BehaviorMetrics:
    combined = BehaviorMetrics(
        s_dis=metrics.s_dis,
        s_amb=metrics.s_amb,
        acc_dis=metrics.acc_dis,
        acc_amb=metrics.acc_amb,
        mmlu=mmlu,
        n_dis=metrics.n_dis,
        n_amb=metrics.n_amb,
    )
    combined.validate()
    return combined
