# train/

## Purpose
Offline training, feature extraction and dataset experiments. Not part of the live
pipeline. Each script computes ROOT from its depth and reads `data/` / `datasets/`
/ writes `models/` from the repo root. Run from the repo root, e.g.
`python train/results/states/states_wesad.py`.

Organised by role:

## `models/` - produce the live `.joblib` bundles
The compute container loads these from `models/`. Re-run only to retrain.

| File | Produces |
|---|---|
| `train_wesad.py` | `wesad_rf.joblib` - RandomForest rest-vs-stress (LOSO ~0.84) |
| `train_valence_wesad.py` | `valence_wesad.joblib` - WESAD valence (stress-vs-amusement) |
| `train_valence_eevr.py` | `valence_eevr.joblib` - EEVR-trained valence |
| `train_valence_case.py` | `valence_case.joblib` - CASE-trained valence |
| `deap_train_valence.py` | `deap_valence_fd.joblib` - DEAP frequency-domain valence |

## `extract/` - build the cached feature matrices
Run once per dataset; output `data/*.npz` (gitignored, regenerable).

| File | Builds |
|---|---|
| `case_extract.py` | CASE FD + morphological PPG features |
| `emowear_extract.py` | EmoWear (Empatica E4 BVP) features |
| `deap_extract_features.py` | DEAP features |

## `results/` - the dataset experiments behind the thesis numbers
LOSO + balanced accuracy throughout. The graded-valence conclusion (WESAD 80% ->
CASE 62% -> EmoWear/DEAP ~50%) comes from here. Grouped by the thesis question
each one answers:

### `results/arousal/` - does arousal work? (the spine)
| File | What it answers |
|---|---|
| `arousal_validate.py` | Our arousal estimator vs labelled arousal (CASE + WESAD) |
| `quadrant_classifier.py` | 4-quadrant Russell classifier on WESAD (honest best number) |

### `results/valence/` - why valence is weak (the rigorous negative result)
| File | What it answers |
|---|---|
| `wesad_explain.py` | SHAP: why the WESAD valence model decides as it does (HR in disguise) |
| `case_valence_stratified.py` | CASE valence, arousal-matched (the decisive test, ~62%) |
| `eevr_valence_stratified.py` | EEVR valence, arousal-stratified (~56%) |
| `case_analyze.py` | CASE valence + arousal LOSO with all refinements |
| `emowear_analyze.py` | EmoWear valence + arousal (~50% - dataset, not method) |

### `results/states/` - native states beat forced valence (the pivot)
| File | What it answers |
|---|---|
| `states_wesad.py` | WESAD native states: stress-vs-calm 86% vs forced valence 58% |
| `states_eevr_case.py` | Same view on EEVR + CASE (arousal collapses on weak clip stimuli) |

### `results/discomfort/` - the 2nd axis, and what transfers to the watch
| File | What it answers |
|---|---|
| `discomfort_features.py` | Where discomfort lives beyond arousal (pulse morphology, WESAD) |
| `my_domain_shift.py` | My real 100 Hz watch signal vs the training distribution |
| `discomfort_transferable.py` | The detector that survives on the wrist (HRV/RMSSD only) |

### `results/validation/` - honesty + methodological critique
| File | What it answers |
|---|---|
| `split_inflation.py` | Random split vs LOSO - the inflation WEARS/Nandini report as success |
| `feedback_validate.py` | The 3 valence models vs the user's OWN feedback labels (Phase A) |
| `galaxy_quality.py` | GalaxyPPG hardware diagnostic: Galaxy IBI vs Polar ECG (not valence) |
| `wesad_healthcheck.py` | Sanity: does the feature pipeline separate stress on WESAD |

## Note on cross-script imports
`results/valence/case_valence_stratified.py` imports from
`results/valence/eevr_valence_stratified.py` (both in `valence/`), referenced as
`train.results.valence.<name>`. `train_valence_case.py` (in `models/`) reuses the
CASE stratification from `train.results.valence.case_valence_stratified`.
