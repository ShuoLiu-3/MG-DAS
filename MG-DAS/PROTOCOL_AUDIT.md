# Protocol Audit Checklist

Complete this checklist before the first GPU run.

## Frozen inputs

- [ ] Exact Llama-2-Chat checkpoint and tokenizer revision recorded.
- [ ] Prompt format and continuation scoring checked against the prior protocol.
- [ ] Nine BBQ attributes use the expected names.
- [ ] Every BBQ base group remains inside one pool.
- [ ] Split manifest hash archived before search.
- [ ] MMLU certification and test files are disjoint.
- [ ] Experiment-profile hyperparameters reviewed and frozen.

## Direction construction

- [ ] Probe accuracy comes from the internal calibration validation partition.
- [ ] PGD perturbations minimize distance to the uniform probe output.
- [ ] Perturbations are aligned by probe log-odds sign before unit normalization.
- [ ] Failed direction-consistency gates discard PGD-derived families only.
- [ ] Centroid remains an independent candidate.
- [ ] IC is computed from centered aligned unit perturbations.
- [ ] Repaired PCs are oriented to the A-PGD-Mean reference before averaging.

## Search and certification

- [ ] Response slopes are fitted through the origin on the four local strengths.
- [ ] Candidate layers are ranked separately for each direction and polarity.
- [ ] Probe ablation changes only the layer nomination score.
- [ ] Single-unit ablations retain zero intervention as an option.
- [ ] Multi-unit weights sum to one and same-layer atoms are rejected.
- [ ] Every layer scale comes from the mean norm of its unperturbed forward state.
- [ ] Search evaluates every depth from one through three.
- [ ] Search does not inspect MMLU, certification, or test results.
- [ ] Uncertified selection is frozen from search results only.
- [ ] Certification enforces magnitude, sign, BBQ accuracy, and MMLU constraints.
- [ ] Near-optimal sparsity is applied only after feasibility filtering.

## Final evaluation

- [ ] Selected configuration identifiers archived before test execution.
- [ ] Frozen test is evaluated once per selected attribute-method configuration.
- [ ] Original and all variants use identical frozen BBQ examples.
- [ ] sDIS reversals are counted on signed attribute-level test scores.
- [ ] Mean absolute bias and accuracy are attribute macro averages.
- [ ] Mean K equals the integer total number of selected atoms divided by nine.
- [ ] No result is manually copied into the generated TABLE II files.
