# Data Format

All input files use UTF-8 JSON Lines: one JSON object per line.

## Normalized BBQ

Each row must contain:

```json
{
  "example_id": "unique-row-id",
  "group_id": "shared-id-for-the-same-BBQ-base-item",
  "attribute": "Age",
  "context": "Context text",
  "question": "Question text",
  "choices": ["choice 0", "choice 1", "choice 2"],
  "label_index": 2,
  "stereotype_index": 0,
  "anti_stereotype_index": 1,
  "unknown_index": 2,
  "is_ambiguous": true,
  "probe_label": 0,
  "neutral_reference": false
}
```

Required attributes are exactly:

- `Age`
- `Disability_status`
- `Gender_identity`
- `Nationality`
- `Physical_appearance`
- `Race_ethnicity`
- `Religion`
- `SES`
- `Sexual_orientation`

`example_id` must be unique. All ambiguous/disambiguated and question-polarity
variants derived from the same base item must use the same `group_id`; grouped
splitting then keeps them in one pool.

The three answer-role indices must be a permutation of `0,1,2`. Ambiguous rows
must label the UNKNOWN choice. The loader deliberately does not infer answer
roles from free text.

`probe_label` may be `0`, `1`, or `null`. Calibration data must contain enough
examples of both classes to train and validate the binary probe.
`neutral_reference` identifies the rows used as the neutral side of the centroid
direction; each attribute must include both neutral and attribute-sensitive rows.

## Prepared MMLU

Certification and frozen-test MMLU data use separate files:

```json
{
  "example_id": "subject/test-id",
  "prompt": "Five demonstrations followed by the test question and answer cue",
  "choices": [" A", " B", " C", " D"],
  "label_index": 1
}
```

The prompt must already contain the complete 5-shot context. Choice strings must
preserve the exact leading whitespace used by the scoring protocol. IDs must be
unique within each file and may not overlap between certification and test files.

## Input validation

After setting paths, run:

```bash
python -m mgdas_v2 --config configs/paper_protocol.json --stage preflight
```

This checks file availability, BBQ schema, exact attribute coverage, unique IDs,
and MMLU certification/test disjointness without loading the language model.

