# Validation Record

The delivered source tree was checked with:

```bash
python -m pytest -q
python -m compileall -q mgdas_v2 tests
python -m mgdas_v2 --config configs/paper_protocol.json --stage validate
```

Validation status at packaging:

- 15 synthetic unit tests passed;
- all package and test modules compiled successfully;
- the paper protocol configuration loaded successfully;
- no model weights or benchmark data are bundled;
- full GPU execution requires the paths and normalized inputs described in the
  README and data-format document.

The unit tests cover the formulas and control flow that can be validated without
external model or dataset files. A small isolated model/data smoke test is
recommended before the first full experiment.

