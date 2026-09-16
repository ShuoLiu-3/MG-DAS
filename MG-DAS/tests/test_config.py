from pathlib import Path

from mgdas_v2.config import load_protocol


def test_paper_configuration_is_valid() -> None:
    path = Path(__file__).parents[1] / "configs" / "paper_protocol.json"
    config = load_protocol(path)
    assert config.search.max_units == 3
    assert config.response.local_strengths == (0.0025, 0.005, 0.01, 0.02)
    assert config.certification.task_tolerance_pp == 5.0
    assert config.certification.mmlu_tolerance_pp == 3.0
