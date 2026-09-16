from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_protocol
from .dataio import load_multiple_choice, load_normalized_bbq
from .pipeline import ATTRIBUTES, PaperPipeline, make_runner
from .reporting import aggregate_ablation, write_ablation_csv
from .splits import build_grouped_split_manifest, save_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="MG-DAS paper experiment pipeline")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("validate", "preflight", "prepare-splits", "directions", "ablation", "aggregate"),
    )
    arguments = parser.parse_args()
    config = load_protocol(arguments.config)
    if arguments.stage == "validate":
        print("protocol configuration is valid")
        return
    if arguments.stage == "preflight":
        model_path = Path(config.model.model_path)
        required_paths = (
            Path(config.data.bbq_normalized_path),
            Path(config.data.mmlu_cert_path),
            Path(config.data.mmlu_test_path),
        )
        if not model_path.is_dir():
            raise FileNotFoundError(f"model directory not found: {model_path}")
        missing = [str(path) for path in required_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"required data files not found: {missing}")
        bbq = load_normalized_bbq(required_paths[0])
        observed_attributes = {example.attribute for example in bbq}
        if observed_attributes != set(ATTRIBUTES):
            raise ValueError(
                "BBQ attributes do not match the paper protocol: "
                f"observed={sorted(observed_attributes)}"
            )
        mmlu_cert = load_multiple_choice(required_paths[1])
        mmlu_test = load_multiple_choice(required_paths[2])
        overlap = {example.example_id for example in mmlu_cert} & {
            example.example_id for example in mmlu_test
        }
        if overlap:
            raise ValueError(f"MMLU certification/test IDs overlap: {sorted(overlap)[:5]}")
        print(
            "preflight passed: "
            f"BBQ={len(bbq)}, MMLU-cert={len(mmlu_cert)}, MMLU-test={len(mmlu_test)}"
        )
        return
    if arguments.stage == "prepare-splits":
        examples = load_normalized_bbq(config.data.bbq_normalized_path)
        manifest = build_grouped_split_manifest(examples, config.split.ratios, config.split.seed)
        save_manifest(config.split.manifest_path, manifest)
        print(f"saved frozen split manifest to {config.split.manifest_path}")
        return
    if arguments.stage == "aggregate":
        result_path = Path(config.data.output_dir) / "ablation_attribute_results.json"
        records = json.loads(result_path.read_text(encoding="utf-8"))["records"]
        rows = aggregate_ablation(records)
        write_ablation_csv(Path(config.data.output_dir) / "table_ii_core_ablation.csv", rows)
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    runner = make_runner(config)
    pipeline = PaperPipeline(config, runner)
    pools, split_digest = pipeline.prepare_splits()
    if arguments.stage == "directions":
        pipeline.build_directions(pools, split_digest)
    elif arguments.stage == "ablation":
        pipeline.build_directions(pools, split_digest)
        pipeline.run_core_ablation(pools, split_digest)


if __name__ == "__main__":
    main()
