# Generative-AI news sentiment and technical indicators for XRP forecasting

Research code and saved experimental evidence for the manuscript **Generative-AI News Sentiment and Technical Indicators for XRP Price Forecasting Using BiLSTM**. XRP is the principal study; HBAR is a separate cross-asset experiment. This repository preserves the completed 2023–2026 study, not the separate 2023–2025 manuscript.

## Start here

1. Browse the clearly named folders under `subprojects/`. Each has code, inputs/configurations and saved outputs.
2. Reproduce the saved results without training:

```bash
python -m pip install -r requirements-metrics.txt
python tools/reproduce_saved_metrics.py --output reproduced
```

The command checks matched validation and later-period metrics and recalculates the ten expected-integrated-gradient shares. It does not train, tune, call GPT or change saved results.

## Subprojects

| Folder | Purpose |
|---|---|
| [01_xrp_technical_bilstm](subprojects/01_xrp_technical_bilstm/) | XRP technical-only BiLSTM (E0) |
| [02_xrp_earlier_direct_news_bilstm](subprojects/02_xrp_earlier_direct_news_bilstm/) | XRP earlier eight-news-input BiLSTM (E1) |
| [03_xrp_compact_direct_news_bilstm](subprojects/03_xrp_compact_direct_news_bilstm/) | XRP compact direct-news BiLSTM (E1B) |
| [04_xrp_sentiment_decay_3h](subprojects/04_xrp_sentiment_decay_3h/) | XRP relevance-weighted sentiment decay: 3-hour half-life |
| [05_xrp_sentiment_decay_6h](subprojects/05_xrp_sentiment_decay_6h/) | XRP relevance-weighted sentiment decay: 6-hour half-life |
| [06_xrp_sentiment_decay_12h](subprojects/06_xrp_sentiment_decay_12h/) | XRP relevance-weighted sentiment decay: 12-hour half-life |
| [07_xrp_sentiment_decay_24h](subprojects/07_xrp_sentiment_decay_24h/) | XRP relevance-weighted sentiment decay: 24-hour half-life |
| [08_xrp_combined_direct_news_and_decay](subprojects/08_xrp_combined_direct_news_and_decay/) | XRP compact direct news plus six-hour decay (E3) |
| [09_xrp_gated_news_correction](subprojects/09_xrp_gated_news_correction/) | XRP gated GRU correction to frozen technical BiLSTM (E4) |
| [10_xrp_ungated_news_correction](subprojects/10_xrp_ungated_news_correction/) | XRP ungated GRU correction to frozen technical BiLSTM (E4) |
| [11_hbar_technical_bilstm](subprojects/11_hbar_technical_bilstm/) | HBAR technical-only BiLSTM |
| [12_hbar_transferred_news_bilstm](subprojects/12_hbar_transferred_news_bilstm/) | HBAR XRP-selected compact news transfer |
| [13_xrp_technical_architecture_comparison](subprojects/13_xrp_technical_architecture_comparison/) | XRP technical BiLSTM/LSTM/BTC ablation: supporting price-target study |
| [14_xrp_technical_arimax](subprojects/14_xrp_technical_arimax/) | XRP technical ARIMAX benchmark |
| [15_xrp_garch_volatility](subprojects/15_xrp_garch_volatility/) | XRP GARCH variance forecast (separate outcome) |
| [16_xrp_frozen_later_evaluation](subprojects/16_xrp_frozen_later_evaluation/) | Frozen later-period XRP evaluation: E0, E1B and persistence |
| [17_expected_integrated_gradients](subprojects/17_expected_integrated_gradients/) | Expected integrated gradients: five seeds, two backgrounds, completeness checks |
| [18_gpt_annotation_provenance](subprojects/18_gpt_annotation_provenance/) | GPT annotation provenance and uncompleted independent human-validation protocol |

## Main findings

Compact direct news reduced mean validation RMSE from 0.063935 to 0.063036 USDT (1.406%). The best tested decay half-life was six hours, with RMSE 0.063637. Adjusted validation intervals did not establish a robust improvement. In the later retrospective block, E1B was worse than E0 (0.042118 versus 0.039976); persistence achieved 0.039341. These are different historical periods and must not be pooled.

Expected integrated gradients gave an equal-pass mean news share of about 23.11%, with 52 retained completeness exceptions. This is model attribution, not exact SHAP, causal price impact, or an accuracy contribution. HBAR transfer showed no news benefit; the archive has sparse HBAR-specific coverage and the transferred semantic fields retain their XRP orientation.

## Structure and reproduction limits

- `subprojects/*/code/original/` preserves the actual source rather than replacing it with a new training design. Some archived scripts contain historical absolute paths; read `REPRODUCIBILITY.md` before full fitting.
- `data/` contains complete prepared numerical model input files, chronological index arrays and fitted scalers where available. Some archived training inputs deliberately redact final-period targets; actual final targets and predictions are supplied separately in subproject 16.
- `results/` contains saved per-seed predictions, configurations, histories and selected checkpoints. Supporting absolute-price architecture experiments are clearly separated from log-return news experiments; GARCH remains a volatility experiment.
- `analysis/` holds the stage-level comparisons, paired uncertainty outputs, source-verification tables and numerical evidence.
- `shared/news_feature_engineering/` contains the causal hourly feature and decay calculations.

## Data access and responsible reuse

The release includes prepared numerical market/news feature tables used by the models, not a redistribution grant over third-party source articles. Source text, private credentials and API response caches are not needed to reproduce the saved metrics and are not published as a blanket public news corpus. Obtain original article text from the relevant providers under their terms. The historical archive does not verify every first-seen time or text revision. No independent human-label accuracy is asserted. See `DATA_AVAILABILITY.md`.

The authors retain rights in their original work. No new blanket software/data licence has been assigned on behalf of the coauthors or third-party providers. Contact the corresponding author for permissions beyond applicable terms. The Elsevier CAS support retains its included LPPL notice.

No model was retrained for this public release.
