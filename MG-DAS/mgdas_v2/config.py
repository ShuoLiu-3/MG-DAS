from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _tuple_floats(values: list[Any], name: str) -> tuple[float, ...]:
    converted = tuple(float(value) for value in values)
    if not converted:
        raise ValueError(f"{name} must not be empty")
    return converted


@dataclass(frozen=True)
class ModelConfig:
    model_path: str
    dtype: str
    num_layers: int
    hidden_dim: int
    layer_module_template: str
    token_scope: str
    prompt_format: str


@dataclass(frozen=True)
class SplitConfig:
    manifest_path: str
    seed: int
    ratios: dict[str, float]


@dataclass(frozen=True)
class DataConfig:
    bbq_normalized_path: str
    mmlu_cert_path: str
    mmlu_test_path: str
    output_dir: str


@dataclass(frozen=True)
class DirectionConfig:
    probe_epochs: int
    probe_batch_size: int
    probe_learning_rate: float
    pgd_steps: int
    pgd_step_fraction: float
    pgd_radius_fraction: float
    min_valid_perturbations: int
    gamma_direction: float
    gamma_pc3: float
    gamma_pc5: float


@dataclass(frozen=True)
class ResponseConfig:
    local_strengths: tuple[float, ...]
    epsilon_bias: float
    near_neutral_abs_log_odds: float
    max_layers_per_direction_polarity: int
    excluded_layers: tuple[int, ...]


@dataclass(frozen=True)
class SearchConfig:
    max_units: int
    beam_width: int
    deployment_strengths: tuple[float, ...]
    weight_grids: dict[int, tuple[tuple[float, ...], ...]]
    epsilon_score: float


@dataclass(frozen=True)
class CertificationConfig:
    task_tolerance_pp: float
    mmlu_tolerance_pp: float
    gain_tolerance: float


@dataclass(frozen=True)
class ProtocolConfig:
    schema_version: int
    seed: int
    model: ModelConfig
    data: DataConfig
    split: SplitConfig
    direction: DirectionConfig
    response: ResponseConfig
    search: SearchConfig
    certification: CertificationConfig

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"Unsupported schema_version={self.schema_version}")
        if self.model.dtype not in {"fp16", "bf16", "fp32"}:
            raise ValueError("model.dtype must be fp16, bf16, or fp32")
        if self.model.token_scope not in {"all", "last"}:
            raise ValueError("model.token_scope must be all or last")
        if self.model.num_layers < 1 or self.model.hidden_dim < 1:
            raise ValueError("model dimensions must be positive")
        expected_pools = {"cal", "search", "cert", "test"}
        if set(self.split.ratios) != expected_pools:
            raise ValueError(f"split.ratios must contain exactly {sorted(expected_pools)}")
        if abs(sum(self.split.ratios.values()) - 1.0) > 1e-9:
            raise ValueError("split.ratios must sum to 1")
        if any(value <= 0 for value in self.split.ratios.values()):
            raise ValueError("all split ratios must be positive")
        direction = self.direction
        if not 0 <= direction.gamma_direction <= 1:
            raise ValueError("gamma_direction must be in [0, 1]")
        if not 0 <= direction.gamma_pc5 <= direction.gamma_pc3 <= 1:
            raise ValueError("require 0 <= gamma_pc5 <= gamma_pc3 <= 1")
        if direction.min_valid_perturbations < 2:
            raise ValueError("min_valid_perturbations must be at least 2")
        if direction.probe_epochs < 1 or direction.probe_batch_size < 1:
            raise ValueError("probe epochs and batch size must be positive")
        if direction.probe_learning_rate <= 0:
            raise ValueError("probe learning rate must be positive")
        if direction.pgd_steps < 1:
            raise ValueError("PGD steps must be positive")
        if direction.pgd_step_fraction <= 0 or direction.pgd_radius_fraction <= 0:
            raise ValueError("PGD step and radius fractions must be positive")
        if self.response.local_strengths != (0.0025, 0.005, 0.01, 0.02):
            raise ValueError("paper protocol requires local strengths 0.0025, 0.005, 0.01, 0.02")
        if self.response.max_layers_per_direction_polarity < 1:
            raise ValueError("max_layers_per_direction_polarity must be positive")
        if self.response.epsilon_bias <= 0 or self.response.near_neutral_abs_log_odds < 0:
            raise ValueError("response epsilon must be positive and threshold non-negative")
        if any(layer < 0 or layer >= self.model.num_layers for layer in self.response.excluded_layers):
            raise ValueError("excluded layers must be valid model layer indices")
        search = self.search
        if search.max_units != 3:
            raise ValueError("the paper protocol requires search.max_units=3")
        if search.beam_width < 1:
            raise ValueError("search.beam_width must be positive")
        if any(strength <= 0 for strength in search.deployment_strengths):
            raise ValueError("deployment strengths must be positive")
        if search.epsilon_score <= 0:
            raise ValueError("epsilon_score must be positive")
        for cardinality in range(1, search.max_units + 1):
            if cardinality not in search.weight_grids:
                raise ValueError(f"missing weight grid for cardinality {cardinality}")
            for weights in search.weight_grids[cardinality]:
                if len(weights) != cardinality:
                    raise ValueError(f"weight tuple {weights} has wrong cardinality")
                if any(weight < 0 for weight in weights):
                    raise ValueError("intervention weights must be non-negative")
                if abs(sum(weights) - 1.0) > 1e-8:
                    raise ValueError(f"weights must sum to one: {weights}")
        if self.certification.task_tolerance_pp < 0 or self.certification.mmlu_tolerance_pp < 0:
            raise ValueError("certification tolerances must be non-negative")
        if self.certification.gain_tolerance < 0:
            raise ValueError("gain_tolerance must be non-negative")


def load_protocol(path: str | Path) -> ProtocolConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    weight_grids = {
        int(cardinality): tuple(tuple(float(weight) for weight in row) for row in rows)
        for cardinality, rows in raw["search"]["weight_grids"].items()
    }
    config = ProtocolConfig(
        schema_version=int(raw["schema_version"]),
        seed=int(raw["seed"]),
        model=ModelConfig(**raw["model"]),
        data=DataConfig(**raw["data"]),
        split=SplitConfig(**raw["split"]),
        direction=DirectionConfig(**raw["direction"]),
        response=ResponseConfig(
            local_strengths=_tuple_floats(raw["response"]["local_strengths"], "local_strengths"),
            epsilon_bias=float(raw["response"]["epsilon_bias"]),
            near_neutral_abs_log_odds=float(raw["response"]["near_neutral_abs_log_odds"]),
            max_layers_per_direction_polarity=int(raw["response"]["max_layers_per_direction_polarity"]),
            excluded_layers=tuple(int(layer) for layer in raw["response"]["excluded_layers"]),
        ),
        search=SearchConfig(
            max_units=int(raw["search"]["max_units"]),
            beam_width=int(raw["search"]["beam_width"]),
            deployment_strengths=_tuple_floats(
                raw["search"]["deployment_strengths"], "deployment_strengths"
            ),
            weight_grids=weight_grids,
            epsilon_score=float(raw["search"]["epsilon_score"]),
        ),
        certification=CertificationConfig(**raw["certification"]),
    )
    config.validate()
    return config
