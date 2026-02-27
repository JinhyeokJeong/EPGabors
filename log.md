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


## 2026-02-27 (v1.1 pipeline revision)

### Why this was implemented
- Make workstation runs easier to shard by model and layer.
- Add explicit train/test condition control.
- Save decoder artifacts for reproducibility.
- Add a lightweight notebook for model/layer inspection before long runs.

### What changed
- Updated `epgabor_v1_pipeline.py`:
  - added `selected_layer` support (`all` or one layer),
  - added `list_available_layers()` and CLI `--list-layers`,
  - added `ConditionFilter` and `build_controlled_splits()`,
  - added `explicit` split mode plus CLI train/test filter arguments,
  - added artifact saving (`decoder_bundle.joblib`, `split_manifest.csv`, `run_config.json`),
  - revised CLI to single-model / single-layer execution.
- Added notebook: `output/jupyter-notebook/model-layer-inspection.ipynb`.
- Expanded tests for layer listing, controlled splits, single-layer run, and artifact integrity.

### Expected use
- Use `--model-name` and `--layer-name` to run one layer at a time.
- Use `--list-layers` or the inspection notebook to decide layer names before large runs.

## 2026-02-27 (small-scale walkthrough notebook)

### Why this was implemented
- Provide a local, low-cost sanity-check notebook that demonstrates the full pipeline on a tiny subset.
- Make it easier to see how data selection, one-layer feature extraction, feature capping, and SVM training fit together before large-scale runs.

### What changed
- Added `output/jupyter-notebook/pipeline-smoke-demo.ipynb`.
- The notebook:
  - builds a 56-image subset,
  - uses `resnet50` / `stage2` by default,
  - uses an explicit split (train on `sd={4,8}`, test on `sd=16`),
  - optionally caps features at 256 dimensions,
  - reports both train and test accuracy,
  - optionally shows `mean=0` boundary behavior.

### Notes
- This notebook is for illustration only and does not save artifacts.
- Default `pretrained=False` avoids weight download/caching issues during local testing.

## 2026-02-27 (pipeline smoke notebook expansion)

### Why this was implemented
- Make the smoke-demo notebook more useful for research-oriented inspection by retaining per-sample condition metadata alongside predictions.
- Add a quick multi-layer extension so layerwise train/test accuracy can be compared in the same tiny demo.

### What changed
- Updated `pipeline-smoke-demo.ipynb` (root-level copy):
  - added a clarification that the dataset contains metadata while the decoder only uses the binary target,
  - added `train_results` and `test_results` tables with metadata + predictions + margins,
  - added a simple example aggregation of test accuracy by actual mean orientation,
  - appended a multi-layer section that loops over all mapped stage-end layers,
  - added a summary dataframe (`layer_summary_df`) and a simple train/test accuracy plot across layers.

### Notes
- This is still a notebook-only illustration; no pipeline code changed.
- The multi-layer section reuses the same 56-image subset and same explicit split as the single-layer demo.
