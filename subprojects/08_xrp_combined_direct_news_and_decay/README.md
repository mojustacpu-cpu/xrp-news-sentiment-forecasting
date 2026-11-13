# XRP compact direct news plus six-hour decay (E3)

Target: `log_return`.

- `code/original/`: preserved experimental implementation; local path constants may require relocation.
- `configs/`: archived settings and feature definitions, with a portable experiment description where available.
- `data/`: prepared numerical inputs, scalers and chronological origin indexes where used.
- `results/`: saved predictions, histories, metrics and selected checkpoints where available.

Run `python tools/reproduce_saved_metrics.py --output reproduced` from the repository root to reproduce saved price metrics without fitting. Read the root reproducibility guide before training: archived scripts preserve no-restart guards and historical paths; do not run stage pipelines blindly. No new models were fitted for this release.


## Source entry points

- [prepare_e3.py](code/original/prepare_e3.py)
- [verify_and_analyse.py](code/original/verify_and_analyse.py)
- [build_e3_reports.py](code/original/build_e3_reports.py)
- [train_candidate.py](code/original/train_candidate.py)
