# XRP relevance-weighted sentiment decay: 3-hour half-life

Target: `log_return`.

- `code/original/`: preserved experimental implementation; local path constants may require relocation.
- `configs/`: archived settings and feature definitions, with a portable experiment description where available.
- `data/`: prepared numerical inputs, scalers and chronological origin indexes where used.
- `results/`: saved predictions, histories, metrics and selected checkpoints where available.

Run `python tools/reproduce_saved_metrics.py --output reproduced` from the repository root to reproduce saved price metrics without fitting. Read the root reproducibility guide before training: archived scripts preserve no-restart guards and historical paths; do not run stage pipelines blindly. No new models were fitted for this release.


## Source entry points

- [verify_and_analyse.py](code/original/verify_and_analyse.py)
- [check_phase1.py](code/original/check_phase1.py)
- [prepare_phase1.py](code/original/prepare_phase1.py)
- [build_phase1_reports.py](code/original/build_phase1_reports.py)
- [pipeline.py](code/original/pipeline.py)
- [build_phase_runner.py](code/original/build_phase_runner.py)
- [train_candidate.py](code/original/train_candidate.py)
