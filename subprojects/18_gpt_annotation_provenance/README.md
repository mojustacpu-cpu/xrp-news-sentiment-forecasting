# GPT annotation provenance and uncompleted independent human-validation protocol

Target: `annotation_provenance_not_human_accuracy`.

- `code/original/`: preserved experimental implementation; local path constants may require relocation.
- `configs/`: archived settings and feature definitions, with a portable experiment description where available.
- `data/`: prepared numerical inputs, scalers and chronological origin indexes where used.
- `results/`: saved predictions, histories, metrics and selected checkpoints where available.

Run `python tools/reproduce_saved_metrics.py --output reproduced` from the repository root to reproduce saved price metrics without fitting. Read the root reproducibility guide before training: archived scripts preserve no-restart guards and historical paths; do not run stage pipelines blindly. No new models were fitted for this release.


## Source entry points

- [impact_audit.py](code/original/impact_audit.py)
- [prepare_audit.py](code/original/prepare_audit.py)
- [verify_legacy_json.py](code/original/verify_legacy_json.py)
- [verify_response_linkage.py](code/original/verify_response_linkage.py)
- [recover_provenance.py](code/original/recover_provenance.py)
- [score_human_audit.py](code/original/score_human_audit.py)
