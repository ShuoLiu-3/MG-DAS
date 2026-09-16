# Experiment Protocol

## Data isolation

BBQ base groups are assigned once to four disjoint pools:

- `cal`: probe training, PGD geometry, direction construction, and response slopes;
- `search`: structure, simplex-weight, and deployment-strength search;
- `cert`: final feasibility checking and minimum-cardinality selection;
- `test`: one frozen evaluation after the configuration is fixed.

The manifest is immutable unless explicitly removed by the operator. Test outcomes
never enter direction construction, response ranking, search, or certification.

## Candidate directions

For each attribute and layer, the pipeline trains a binary probe and obtains
norm-bounded perturbations that move its output toward the uniform distribution.
Non-zero perturbations are unit-normalized and oriented using the sign of the
probe log-odds before aggregation.

The direction-consistency statistic gates PGD-derived families. Intervention
concentration is computed from centered aligned perturbations and controls whether
Repaired-PC3 or Repaired-PC5 is added. The centroid remains an independent family.

## Response screening and search

Every valid direction-polarity-layer atom is evaluated on the local strength grid
`0.0025, 0.005, 0.01, 0.02`. The response slope is fitted through the origin, and
layers are ranked separately for every direction and polarity.

Beam search evaluates cardinalities one through three. Non-negative weights sum
to one, so all selected atoms share a single total strength. Two atoms at the same
layer are not allowed. Candidates are retained by Pareto comparison of signed-bias
gain, disambiguated-accuracy change, and sign-direction violation.

## Certification

Certification includes zero intervention and requires every accepted non-zero
configuration to:

- not increase absolute `sDIS`;
- remain on the original side of zero;
- lose no more than five percentage points of BBQ disambiguated accuracy;
- lose no more than three percentage points of MMLU accuracy.

Among feasible configurations within the fixed gain tolerance of the best gain,
the pipeline selects the smallest cardinality. Remaining ties use higher gain,
smaller task loss, and a deterministic identifier.

## Core ablation variants

- `Original`: unchanged model.
- `Probe-ranked, Kmax=1`: probe validation accuracy supplies layer nomination;
  search and certification remain unchanged, with at most one atom.
- `Response-ranked, Kmax=1`: behavioral response supplies layer nomination;
  search and certification remain unchanged, with at most one atom.
- `MG-DAS w/o certification`: full response-ranked multi-unit search; selection
  uses search outcomes without certification filtering.
- `Full MG-DAS, Kmax=3`: response-ranked search followed by independent
  certification and minimum-cardinality selection.

The generated table reports Mean K, mean absolute signed BBQ scores, disambiguated
sign reversals, BBQ disambiguated accuracy, and MMLU accuracy over nine attributes.

