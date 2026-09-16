from __future__ import annotations

import json
import hashlib
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .artifacts import (
    evaluated_to_dict,
    file_sha256,
    implementation_sha256,
    load_direction_families,
    save_direction_families,
    save_stage,
)
from .certification import certify_shortlist
from .config import ProtocolConfig
from .dataio import load_multiple_choice, load_normalized_bbq
from .directions import (
    DirectionFamily,
    build_direction_families,
    neutralizing_pgd,
    train_linear_probe,
)
from .evaluation import attach_mmlu, evaluate_bbq, evaluate_bbq_and_mmlu
from .modeling import HuggingFaceRunner, multiple_choice_accuracy
from .reporting import aggregate_ablation, write_ablation_csv
from .response import nominate_by_probe_accuracy, screen_behavioral_response
from .search import beam_search, select_without_certification
from .splits import (
    build_grouped_split_manifest,
    load_manifest,
    manifest_digest,
    save_manifest,
    split_examples,
)
from .test_guard import FrozenTestGuard
from .types import BBQExample, BehaviorMetrics, EvaluatedConfig, InterventionConfig


ATTRIBUTES = (
    "Age",
    "Disability_status",
    "Gender_identity",
    "Nationality",
    "Physical_appearance",
    "Race_ethnicity",
    "Religion",
    "SES",
    "Sexual_orientation",
)


@dataclass
class PaperPipeline:
    config: ProtocolConfig
    runner: HuggingFaceRunner

    @property
    def output_dir(self) -> Path:
        return Path(self.config.data.output_dir)

    def prepare_splits(self) -> tuple[dict[str, list[BBQExample]], str]:
        examples = load_normalized_bbq(self.config.data.bbq_normalized_path)
        manifest_path = Path(self.config.split.manifest_path)
        if manifest_path.exists():
            manifest = load_manifest(manifest_path)
        else:
            manifest = build_grouped_split_manifest(
                examples, self.config.split.ratios, self.config.split.seed
            )
            save_manifest(manifest_path, manifest)
        pools = split_examples(examples, manifest)
        observed_attributes = {example.attribute for example in examples}
        missing = set(ATTRIBUTES) - observed_attributes
        if missing:
            raise ValueError(f"normalized BBQ data is missing attributes: {sorted(missing)}")
        return pools, manifest_digest(manifest)

    def _attribute_examples(
        self, pools: dict[str, list[BBQExample]], pool_name: str, attribute: str
    ) -> list[BBQExample]:
        examples = [example for example in pools[pool_name] if example.attribute == attribute]
        if not examples:
            raise ValueError(f"{attribute}: pool {pool_name} is empty")
        return examples

    def _direction_digest(self, split_digest: str) -> str:
        payload = {
            "schema_version": self.config.schema_version,
            "implementation_sha256": implementation_sha256(),
            "seed": self.config.seed,
            "model": asdict(self.config.model),
            "direction": asdict(self.config.direction),
            "bbq_sha256": file_sha256(self.config.data.bbq_normalized_path),
            "split_digest": split_digest,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _evaluation_digest(self, split_digest: str) -> str:
        payload = {
            "direction_digest": self._direction_digest(split_digest),
            "response": asdict(self.config.response),
            "search": asdict(self.config.search),
            "certification": asdict(self.config.certification),
            "mmlu_cert_sha256": file_sha256(self.config.data.mmlu_cert_path),
            "mmlu_test_sha256": file_sha256(self.config.data.mmlu_test_path),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _probe_partition(self, examples: list[BBQExample], attribute: str) -> tuple[np.ndarray, np.ndarray]:
        labelled_by_class = {
            probe_class: [
                index for index, example in enumerate(examples) if example.probe_label == probe_class
            ]
            for probe_class in (0, 1)
        }
        labelled = labelled_by_class[0] + labelled_by_class[1]
        if len(labelled) < self.config.direction.min_valid_perturbations * 2:
            raise ValueError(f"{attribute}: insufficient probe-labelled calibration examples")
        randomizer = random.Random(self.config.seed + sum(attribute.encode("utf-8")))
        train_values = []
        validation_values = []
        for probe_class in (0, 1):
            class_indices = labelled_by_class[probe_class]
            randomizer.shuffle(class_indices)
            validation_size = max(1, int(round(0.2 * len(class_indices))))
            validation_values.extend(class_indices[:validation_size])
            train_values.extend(class_indices[validation_size:])
        validation_indices = np.asarray(sorted(validation_values), dtype=np.int64)
        train_indices = np.asarray(sorted(train_values), dtype=np.int64)
        train_labels = {examples[index].probe_label for index in train_indices}
        validation_labels = {examples[index].probe_label for index in validation_indices}
        if train_labels != {0, 1} or validation_labels != {0, 1}:
            raise ValueError(f"{attribute}: probe train and validation partitions must contain both classes")
        return train_indices, validation_indices

    def build_directions(self, pools: dict[str, list[BBQExample]], split_digest: str) -> None:
        direction_dir = self.output_dir / "directions"
        direction_dir.mkdir(parents=True, exist_ok=True)
        protocol_path = direction_dir / "_protocol.json"
        direction_digest = self._direction_digest(split_digest)
        if protocol_path.exists():
            saved = json.loads(protocol_path.read_text(encoding="utf-8"))
            if saved.get("direction_digest") != direction_digest:
                raise RuntimeError("cached directions were built under a different protocol")
        else:
            cached_files = [path for path in direction_dir.iterdir() if path.name != protocol_path.name]
            if cached_files:
                raise RuntimeError("direction cache exists without protocol metadata; use a clean output directory")
            save_stage(protocol_path, {"schema_version": 1, "direction_digest": direction_digest})
        all_layers = list(range(self.config.model.num_layers))
        for attribute in ATTRIBUTES:
            direction_path = direction_dir / f"{attribute}.npz"
            probe_path = direction_dir / f"{attribute}_probe_accuracy.json"
            if direction_path.exists() and probe_path.exists():
                continue
            examples = self._attribute_examples(pools, "cal", attribute)
            train_indices, validation_indices = self._probe_partition(examples, attribute)
            activations = self.runner.extract_bbq_activations(examples, all_layers)
            labels = np.asarray(
                [-1 if example.probe_label is None else example.probe_label for example in examples],
                dtype=np.int64,
            )
            neutral_mask = np.asarray([example.neutral_reference for example in examples], dtype=bool)
            if not neutral_mask.any() or neutral_mask.all():
                raise ValueError(f"{attribute}: neutral_reference must identify both neutral and attribute rows")
            families: list[DirectionFamily] = []
            probe_accuracy: dict[int, float] = {}
            for layer in all_layers:
                layer_hidden = activations[layer]
                if layer_hidden.ndim != 2 or layer_hidden.shape[1] != self.config.model.hidden_dim:
                    raise ValueError(
                        f"{attribute}: layer {layer} activation shape {layer_hidden.shape} "
                        f"does not match hidden_dim={self.config.model.hidden_dim}"
                    )
                probe, accuracy = train_linear_probe(
                    layer_hidden[train_indices],
                    labels[train_indices],
                    layer_hidden[validation_indices],
                    labels[validation_indices],
                    epochs=self.config.direction.probe_epochs,
                    batch_size=self.config.direction.probe_batch_size,
                    learning_rate=self.config.direction.probe_learning_rate,
                    seed=self.config.seed + layer,
                    device="cpu",
                )
                probe_accuracy[layer] = accuracy
                perturbations, log_odds = neutralizing_pgd(
                    probe,
                    layer_hidden[train_indices],
                    radius_fraction=self.config.direction.pgd_radius_fraction,
                    step_fraction=self.config.direction.pgd_step_fraction,
                    steps=self.config.direction.pgd_steps,
                    device="cpu",
                )
                families.extend(
                    build_direction_families(
                        layer=layer,
                        attribute_hidden=layer_hidden[~neutral_mask],
                        neutral_hidden=layer_hidden[neutral_mask],
                        perturbations=perturbations,
                        probe_log_odds=log_odds,
                        min_valid=self.config.direction.min_valid_perturbations,
                        gamma_direction=self.config.direction.gamma_direction,
                        gamma_pc3=self.config.direction.gamma_pc3,
                        gamma_pc5=self.config.direction.gamma_pc5,
                    )
                )
            save_direction_families(direction_path, families)
            save_stage(
                probe_path,
                {
                    "attribute": attribute,
                    "probe_validation_accuracy": {
                        str(layer): probe_accuracy[layer] for layer in sorted(probe_accuracy)
                    },
                },
            )

    def _load_attribute_assets(self, attribute: str) -> tuple[list[DirectionFamily], dict[int, float]]:
        direction_dir = self.output_dir / "directions"
        families = load_direction_families(direction_dir / f"{attribute}.npz")
        payload = json.loads(
            (direction_dir / f"{attribute}_probe_accuracy.json").read_text(encoding="utf-8")
        )
        probe_accuracy = {
            int(layer): float(value)
            for layer, value in payload["probe_validation_accuracy"].items()
        }
        return families, probe_accuracy

    def _bbq_evaluator(self, examples: list[BBQExample]):
        return lambda intervention: evaluate_bbq(
            examples, intervention, self.runner.score_bbq
        )

    def _cert_evaluator(self, bbq_examples, mmlu_examples):
        cache: dict[str, BehaviorMetrics] = {}

        def evaluate(intervention: InterventionConfig) -> BehaviorMetrics:
            if intervention.identifier not in cache:
                cache[intervention.identifier] = evaluate_bbq_and_mmlu(
                    bbq_examples,
                    mmlu_examples,
                    intervention,
                    self.runner.score_bbq,
                    self.runner.score_multiple_choice,
                )
            return cache[intervention.identifier]

        return evaluate

    def _search(
        self,
        atoms,
        baseline: BehaviorMetrics,
        examples: list[BBQExample],
        max_units: int,
    ) -> list[EvaluatedConfig]:
        weight_grids = {
            cardinality: self.config.search.weight_grids[cardinality]
            for cardinality in range(1, max_units + 1)
        }
        return beam_search(
            atoms=atoms,
            baseline=baseline,
            evaluation_function=self._bbq_evaluator(examples),
            max_units=max_units,
            beam_width=self.config.search.beam_width,
            deployment_strengths=self.config.search.deployment_strengths,
            weight_grids=weight_grids,
            epsilon_score=self.config.search.epsilon_score,
        )

    def _certify(
        self,
        shortlist,
        baseline,
        evaluator,
    ) -> tuple[EvaluatedConfig, list[EvaluatedConfig]]:
        return certify_shortlist(
            shortlist=shortlist,
            baseline=baseline,
            evaluator=evaluator,
            task_tolerance_pp=self.config.certification.task_tolerance_pp,
            mmlu_tolerance_pp=self.config.certification.mmlu_tolerance_pp,
            gain_tolerance=self.config.certification.gain_tolerance,
            epsilon_score=self.config.search.epsilon_score,
        )

    def run_core_ablation(self, pools: dict[str, list[BBQExample]], split_digest: str) -> None:
        mmlu_cert = load_multiple_choice(self.config.data.mmlu_cert_path)
        mmlu_test = load_multiple_choice(self.config.data.mmlu_test_path)
        if Path(self.config.data.mmlu_cert_path).resolve() == Path(self.config.data.mmlu_test_path).resolve():
            raise ValueError("MMLU certification and frozen-test files must be disjoint")
        cert_ids = {example.example_id for example in mmlu_cert}
        test_ids = {example.example_id for example in mmlu_test}
        overlap = cert_ids & test_ids
        if overlap:
            raise ValueError(f"MMLU certification and test IDs overlap: {sorted(overlap)[:5]}")
        evaluation_digest = self._evaluation_digest(split_digest)
        result_path = self.output_dir / "ablation_attribute_results.json"
        existing = []
        if result_path.exists():
            result_payload = json.loads(result_path.read_text(encoding="utf-8"))
            if result_payload.get("evaluation_digest") != evaluation_digest:
                raise RuntimeError("existing ablation records belong to a different protocol")
            existing = result_payload["records"]
        completed = {(row["attribute"], row["variant"]) for row in existing}
        records = list(existing)
        guard = FrozenTestGuard(self.output_dir / "frozen_test_ledger.json")
        zero = InterventionConfig.zero()
        baseline_mmlu_cert = multiple_choice_accuracy(
            self.runner.score_multiple_choice(mmlu_cert, zero), mmlu_cert
        )
        baseline_test_path = self.output_dir / "baseline_mmlu_test.json"
        if baseline_test_path.exists():
            baseline_payload = json.loads(baseline_test_path.read_text(encoding="utf-8"))
            if baseline_payload.get("evaluation_digest") != evaluation_digest:
                raise RuntimeError("cached frozen-test MMLU belongs to a different protocol")
            baseline_mmlu_test = float(baseline_payload["mmlu"])
        else:
            guard.assert_available("__global__", "Original MMLU", zero.identifier, evaluation_digest)
            baseline_mmlu_test = multiple_choice_accuracy(
                self.runner.score_multiple_choice(mmlu_test, zero), mmlu_test
            )
            save_stage(
                baseline_test_path,
                {"schema_version": 1, "evaluation_digest": evaluation_digest, "mmlu": baseline_mmlu_test},
            )
            guard.record("__global__", "Original MMLU", zero.identifier, evaluation_digest)
        for attribute in ATTRIBUTES:
            cal_examples = self._attribute_examples(pools, "cal", attribute)
            search_examples = self._attribute_examples(pools, "search", attribute)
            cert_examples = self._attribute_examples(pools, "cert", attribute)
            test_examples = self._attribute_examples(pools, "test", attribute)
            families, probe_accuracy = self._load_attribute_assets(attribute)
            response_atoms, response_records = screen_behavioral_response(
                attribute=attribute,
                examples=cal_examples,
                directions=families,
                score_function=self.runner.score_bbq,
                local_strengths=self.config.response.local_strengths,
                epsilon_bias=self.config.response.epsilon_bias,
                near_neutral_abs_log_odds=self.config.response.near_neutral_abs_log_odds,
                max_layers_per_direction_polarity=self.config.response.max_layers_per_direction_polarity,
                excluded_layers=self.config.response.excluded_layers,
            )
            probe_atoms = nominate_by_probe_accuracy(
                attribute=attribute,
                directions=families,
                probe_accuracy_by_layer=probe_accuracy,
                max_layers_per_direction_polarity=self.config.response.max_layers_per_direction_polarity,
                excluded_layers=self.config.response.excluded_layers,
            )
            save_stage(
                self.output_dir / "response" / f"{attribute}.json",
                {
                    "attribute": attribute,
                    "records": [record.__dict__ for record in response_records],
                    "response_atom_ids": [atom.atom_id for atom in response_atoms],
                    "probe_atom_ids": [atom.atom_id for atom in probe_atoms],
                },
            )
            baseline_search = evaluate_bbq(search_examples, InterventionConfig.zero(), self.runner.score_bbq)
            probe_shortlist = self._search(probe_atoms, baseline_search, search_examples, max_units=1)
            response_single_shortlist = self._search(
                response_atoms, baseline_search, search_examples, max_units=1
            )
            full_shortlist = self._search(
                response_atoms,
                baseline_search,
                search_examples,
                max_units=self.config.search.max_units,
            )
            baseline_cert = attach_mmlu(
                evaluate_bbq(cert_examples, zero, self.runner.score_bbq),
                baseline_mmlu_cert,
            )
            cert_evaluator = self._cert_evaluator(cert_examples, mmlu_cert)
            probe_selected, probe_certified = self._certify(
                probe_shortlist, baseline_cert, cert_evaluator
            )
            response_selected, response_certified = self._certify(
                response_single_shortlist, baseline_cert, cert_evaluator
            )
            full_selected, full_certified = self._certify(
                full_shortlist, baseline_cert, cert_evaluator
            )
            no_certification_selected = select_without_certification(full_shortlist)
            selections = {
                "Original": InterventionConfig.zero(),
                "Probe-ranked, Kmax=1": probe_selected.config,
                "Response-ranked, Kmax=1": response_selected.config,
                "MG-DAS w/o certification": no_certification_selected.config,
                "Full MG-DAS, Kmax=3": full_selected.config,
            }
            save_stage(
                self.output_dir / "certification" / f"{attribute}.json",
                {
                    "attribute": attribute,
                    "probe": [evaluated_to_dict(item) for item in probe_certified],
                    "response_single": [evaluated_to_dict(item) for item in response_certified],
                    "full": [evaluated_to_dict(item) for item in full_certified],
                    "selected": {
                        variant: config.identifier for variant, config in selections.items()
                    },
                },
            )
            for variant, selected_config in selections.items():
                if (attribute, variant) in completed:
                    continue
                guard.assert_available(attribute, variant, selected_config.identifier, evaluation_digest)
                if variant == "Original":
                    test_metrics = attach_mmlu(
                        evaluate_bbq(test_examples, zero, self.runner.score_bbq),
                        baseline_mmlu_test,
                    )
                else:
                    test_metrics = evaluate_bbq_and_mmlu(
                        test_examples,
                        mmlu_test,
                        selected_config,
                        self.runner.score_bbq,
                        self.runner.score_multiple_choice,
                    )
                records.append(
                    {
                        "attribute": attribute,
                        "variant": variant,
                        "selected_k": selected_config.cardinality,
                        "config_identifier": selected_config.identifier,
                        "s_dis": test_metrics.s_dis,
                        "s_amb": test_metrics.s_amb,
                        "acc_dis": test_metrics.acc_dis,
                        "acc_amb": test_metrics.acc_amb,
                        "mmlu": test_metrics.mmlu,
                    }
                )
                save_stage(
                    result_path,
                    {
                        "schema_version": 1,
                        "evaluation_digest": evaluation_digest,
                        "records": records,
                    },
                )
                guard.record(attribute, variant, selected_config.identifier, evaluation_digest)
        table_rows = aggregate_ablation(records)
        write_ablation_csv(self.output_dir / "table_ii_core_ablation.csv", table_rows)
        save_stage(self.output_dir / "table_ii_core_ablation.json", table_rows)


def make_runner(config: ProtocolConfig) -> HuggingFaceRunner:
    return HuggingFaceRunner(
        model_path=config.model.model_path,
        dtype=config.model.dtype,
        layer_module_template=config.model.layer_module_template,
        token_scope=config.model.token_scope,
        prompt_format=config.model.prompt_format,
    )
