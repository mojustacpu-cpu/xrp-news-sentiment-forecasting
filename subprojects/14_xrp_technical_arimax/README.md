# XRP technical ARIMAX benchmark

Target: `absolute_price_or_log_price`.

- `code/original/`: preserved experimental implementation; local path constants may require relocation.
- `configs/`: archived settings and feature definitions, with a portable experiment description where available.
- `data/`: prepared numerical inputs, scalers and chronological origin indexes where used.
- `results/`: saved predictions, histories, metrics and selected checkpoints where available.

Run `python tools/reproduce_saved_metrics.py --output reproduced` from the repository root to reproduce saved price metrics without fitting. Read the root reproducibility guide before training: archived scripts preserve no-restart guards and historical paths; do not run stage pipelines blindly. No new models were fitted for this release.

This supporting experiment uses a different target from E0/E1B. Do not pool its scores to infer the incremental value of news.
