# Log

## 2026-02-27 (v1 pipeline implementation)

### Why this was implemented
- Build a reproducible v1 pipeline to test whether pretrained CNN layer representations encode **binary mean orientation** (CCW vs CW) on EPOriGabors.
- Keep outputs compatible with later psychometric-style analyses (PSE/precision) by saving trial-level and condition-level results.

### Key decisions locked during planning
- Decoder: `LinearSVC` (SVM only).
- Layer scope: stage-end layers only (not every conv op).
- Models: `resnet50`, `convnext_tiny`, `efficientnet_b0`, `vgg16_bn`.
- `mean=0`: excluded from training, evaluated separately as boundary cases.
- Preprocessing: deterministic only (`resize + ToTensor + normalize`), no crop/rotation/augmentation.
- Default subset: exclude `ss=1` and `sd=0` (overridable).

### What changed
- Updated dataset in `EPOriGabors.py`:
  - fixed `include_zerovar` bug (now properly excludes `sd=0` when False),
  - robust filename parsing with `instance`,
  - added explicit filters (`filter_mean`, `filter_sd`, `filter_ss`, `filter_instance`),
  - added optional metadata return (`file_name`, `condition_id`).
- Added `epgabor_v1_pipeline.py`:
  - model/layer map API,
  - feature extraction with GAP,
  - cross-condition split builders,
  - `LinearSVC` training/evaluation,
  - boundary (`mean=0`) evaluation via margin,
  - CSV save helpers + CLI runner.
- Added `tests/test_epgabors_v1.py` (+ `tests/__init__.py`) with core validation checks.
- Added `ResearchPlan.md` and updated `Description.md` to reflect v1 scope and constraints.

### Validation status
- Python compile checks passed.
- Test functions executed manually in `pip-torch-bayes` environment and passed.
- Smoke run completed and wrote outputs to `results_v1_smoke/`.

