# MG-DAS

Official experiment pipeline for **MG-DAS: Reliability-Aware Selection of Activation Interventions for LLM Debiasing**.

The implementation follows the method and core ablation protocol described in
the paper. It covers direction construction, behavioral response screening,
sparse multi-unit search, independent certification, frozen evaluation, and
automatic generation of the core ablation table.

## Method coverage

- Four disjoint pools: calibration, search, certification, and frozen test.
- Binary probes and norm-bounded PGD toward a uniform probe output.
- Per-sample sign alignment and unit normalization of PGD perturbations.
- Direction-consistency gating and intervention-concentration analysis.
- Centroid, A-PGD-Mean, Repaired-PC3, and Repaired-PC5 direction families.
- Direction-polarity-layer response screening at four local strengths.
- Pareto-aware beam search for `K=1,...,3` under one shared strength budget.
- Same-layer mutual exclusion and non-negative simplex weights.
- Certification of signed-bias magnitude, direction preservation, BBQ
  disambiguated accuracy, and MMLU accuracy.
- Explicit zero-intervention retention and near-optimal minimum-cardinality
  selection.
- Probe-ranked, response-ranked, uncertified, and full MG-DAS ablations.

For every selected layer, the intervention scale is measured by a separate
unperturbed forward pass. Later-layer scales therefore remain fixed and cannot
change because of earlier interventions.

## Repository layout

```text
MG-DAS_Code/
  configs/paper_protocol.json    Main experiment profile
  docs/DATA_FORMAT.md            Required BBQ and MMLU schemas
  docs/EXPERIMENT_PROTOCOL.md    Stage semantics and ablation definitions
  mgdas_v2/                      Implementation package
  tests/                         Synthetic unit tests
  PROTOCOL_AUDIT.md              Checklist before a full experiment
  RUN_GUIDE_CN.md                Chinese execution guide
  VALIDATION.md                  Delivered validation record
```

Model weights and benchmark data are not distributed in this package.

## Environment

- Python 3.10 or newer
- PyTorch 2.1 or newer
- Transformers 4.38 or newer
- NumPy 1.24--1.x

Create a clean environment and install the package:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Configuration

Edit `configs/paper_protocol.json` before execution:

1. Set `model.model_path` to the local Llama-2-Chat-7B or Llama-2-Chat-13B directory.
2. Set the normalized BBQ, MMLU certification, and MMLU test paths.
3. Select a new, empty output directory.
4. Review and freeze the complete profile before certification or testing.

The model profile expects 32 layers, hidden size 4096, and intervention hooks at
`model.layers.{layer}.mlp`. If a differently packaged checkpoint exposes a
different module path, update `layer_module_template` only after verifying the
resolved module and activation dimension.

## Execution

All commands are run from the repository root.

Validate configuration syntax without accessing data or loading the model:

```bash
python -m mgdas_v2 --config configs/paper_protocol.json --stage validate
```

Validate paths, schemas, attribute names, and MMLU split disjointness without
loading the model:

```bash
python -m mgdas_v2 --config configs/paper_protocol.json --stage preflight
```

Create and freeze the grouped four-pool BBQ manifest:

```bash
python -m mgdas_v2 --config configs/paper_protocol.json --stage prepare-splits
```

Build probes and direction families:

```bash
python -m mgdas_v2 --config configs/paper_protocol.json --stage directions
```

Run response screening, search, certification, and frozen evaluation:

```bash
python -m mgdas_v2 --config configs/paper_protocol.json --stage ablation
```

Regenerate the final table from completed attribute records without model
execution:

```bash
python -m mgdas_v2 --config configs/paper_protocol.json --stage aggregate
```

## Output integrity

The pipeline is designed to prevent accidental protocol mixing:

- BBQ base groups remain entirely inside one pool.
- MMLU certification and test IDs must be disjoint.
- Existing split manifests are never overwritten implicitly.
- Directions and ablation records carry data, configuration, and source-code
  digests.
- Incompatible cached outputs are rejected.
- Search never reads certification or frozen-test BBQ outcomes.
- MMLU is excluded from search and used only during certification and reporting.
- Frozen-test evaluations are recorded in an append-only consumption ledger.
- Attribute-level records are written incrementally before final aggregation.
- No target result is hard-coded into the implementation.

The configured output directory contains direction archives, response records,
certification decisions, frozen-test records, protocol digests, and the generated
`table_ii_core_ablation.csv` file.

## Tests

Run the synthetic test suite without model weights or benchmark data:

```bash
python -m pytest -q
```

The tests cover metric definitions, grouped splits, aligned direction geometry,
fixed unperturbed intervention scaling, multi-layer hooks, response ranking,
beam search, certification, reporting, and configuration validation.

