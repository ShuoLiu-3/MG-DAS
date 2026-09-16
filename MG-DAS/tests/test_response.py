import numpy as np

from mgdas_v2.directions import DirectionFamily
from mgdas_v2.response import screen_behavioral_response
from mgdas_v2.types import BBQExample, InterventionConfig


def make_example(identifier: str) -> BBQExample:
    return BBQExample(
        example_id=identifier,
        group_id=identifier,
        attribute="Age",
        context="context",
        question="question",
        choices=("stereotype", "anti", "unknown"),
        label_index=2,
        stereotype_index=0,
        anti_stereotype_index=1,
        unknown_index=2,
        is_ambiguous=True,
    )


def fake_scorer(examples: list[BBQExample], config: InterventionConfig) -> np.ndarray:
    scores = np.zeros((len(examples), 3), dtype=np.float64)
    scores[:, 0] = 0.2
    if config.cardinality:
        atom = config.atoms[0]
        scores[:, 0] -= config.strength * atom.polarity * (atom.layer + 1)
    return scores


def test_response_nomination_is_direction_polarity_conditioned() -> None:
    examples = [make_example(f"e{index}") for index in range(4)]
    directions = [
        DirectionFamily("centroid", 1, np.asarray([1.0, 0.0]), None, None),
        DirectionFamily("centroid", 2, np.asarray([1.0, 0.0]), None, None),
    ]
    atoms, records = screen_behavioral_response(
        attribute="Age",
        examples=examples,
        directions=directions,
        score_function=fake_scorer,
        local_strengths=(0.0025, 0.005, 0.01, 0.02),
        epsilon_bias=0.05,
        near_neutral_abs_log_odds=0.5,
        max_layers_per_direction_polarity=1,
        excluded_layers=(),
    )
    positive_atom = next(atom for atom in atoms if atom.polarity == 1)
    negative_atom = next(atom for atom in atoms if atom.polarity == -1)
    assert positive_atom.layer == 2
    assert negative_atom.layer == 1
    assert len(records) == 4

