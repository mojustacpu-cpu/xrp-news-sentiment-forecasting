# Reproducibility

## Verified, portable saved-output reproduction
Run `python tools/reproduce_saved_metrics.py --output reproduced` with NumPy and pandas. It reads only relative repository paths, recomputes each saved forecast CSV, the canonical main validation and later price arrays, and all ten expected-integrated-gradient arrays. The generated verification JSON checks the published rounded main RMSE values and attribution percentage. It cannot reconstruct missing historical first-availability evidence or human validation.

## Actual source code and full fitting
Each named subproject contains its original Python implementation under `code/original/`, with architecture modules/references where applicable, and the archived feature/seed/training configurations. This is the original experimental code, not a new substitute fitted to reported results. Original scripts may reference the researcher's historical directory structure, external acquisition paths, bundled document tooling, or existing immutable stage folders. Those paths must be relocated and fresh output folders selected before full training. Do not blindly execute a stage pipeline: completed-run guards intentionally prevent overwriting or silently restarting archived results.

The prepared model datasets and train/validation origin arrays allow the forecasting inputs and splits to be inspected without acquiring headlines again. Trained checkpoints and scalers are supplied where available. Load only trusted checkpoint/pickle files; they are Python serialization formats. `PUBLICATION_FILE_MANIFEST.json` and `RELEASE_CHECKSUMS.json` record their identities. New fits may differ across platforms and library versions; no unexecuted end-to-end path-free retraining is claimed.

## Environment
The archived main experiments used Python3.9, PyTorch2.8, scikit-learn1.6.1, statsmodels0.14.6 and arch7.2, with NumPy/pandas/Matplotlib. The two requirements files distinguish the lightweight metric route from the fuller research environment. Training was on CPU with four intra-operation threads. The public saved-output verifier is tested independently of fitting.

## Main design
All principal news experiments use 30 completed hourly rows, an exact +24h log-return target, inverse target scaling followed by `origin_close * exp(predicted_return)`, training-only input/target scalers, chronological purges, validation checkpoint selection and seeds42,123,2026,7,99. Five seeds are optimisation repetitions, not five independent markets. Supporting technical architecture experiments use a different fitting target; GARCH forecasts variance and has separate metrics. No public-release step changes that design.

## Scope
Main validation uses2136 origins; later frozen XRP evaluation uses5262. Final historical exposure, repeated validation selection, news-availability uncertainty, no independent human audit and attribution limitations remain disclosed. The original configurations may describe planned stages that were not executed; plans are not results. Read the per-run COMPLETE metadata and saved prediction files to identify actual completed work.
