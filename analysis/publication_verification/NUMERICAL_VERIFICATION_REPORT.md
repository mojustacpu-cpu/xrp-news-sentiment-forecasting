# Numerical verification report

Status: all executed numerical checks pass. No forecast point estimate was changed.

| Period | Model | RMSE mean | Seed SD | MAE | R2 | Direction |
| --- | --- | --- | --- | --- | --- | --- |
| validation | E0 | 0.063935015 | 0.000416331 | 0.044513791 | 0.951957057 | 53.783% |
| validation | E1B | 0.063036174 | 0.000384950 | 0.043988668 | 0.953298591 | 55.871% |
| validation | E2_h3 | 0.063723756 | 0.000659773 | 0.044432475 | 0.952271553 | 52.275% |
| validation | E2_h6 | 0.063636943 | 0.000768105 | 0.044610675 | 0.952400042 | 54.045% |
| validation | E2_h12 | 0.064018777 | 0.001313861 | 0.044839393 | 0.951816494 | 53.736% |
| validation | E2_h24 | 0.064341014 | 0.000991960 | 0.044974644 | 0.951337355 | 50.721% |
| later | E0 | 0.039976091 | 0.000239361 | 0.027432393 | 0.935190554 | 49.605% |
| later | E1B | 0.042117907 | 0.004186198 | 0.028314130 | 0.927493399 | 49.437% |
| later | E2_h6 | 0.040546058 | 0.000974739 | 0.027840690 | 0.933300401 | 49.513% |

## Corrected primary risk intervals

Risk is the square root of mean-seed squared loss over matched origins. It differs from the arithmetic mean seed RMSE used for configuration ranking. The table is in USDT; a negative difference favours news. All use50,000 draws and72-hour paired calendar blocks.

| Period | Contrast | Risk difference | 95% interval | Stage-adjusted interval | Family |
| --- | --- | --- | --- | --- | --- |
| validation | E1B vs E0 | -0.000898985 | [-0.001688886, -0.000030277] | [-0.002030333, 0.000244868] | 10 |
| validation | E2_h6 vs E0 | -0.000295448 | [-0.001151365, 0.000627557] | [-0.001553512, 0.000989962] | 10 |
| later | E1B vs E0 | +0.002307346 | [0.000563161, 0.004304302] | [0.000563161, 0.004304302] | 1 |
| later | E2_h6 vs E0 | +0.000578766 | [0.000177835, 0.001032844] | [0.000177835, 0.001032844] | 1 |
| HBAR | HBAR_E1B vs technical | +0.000076519 | [0.000007090, 0.000149366] | [0.000007090, 0.000149366] | 1 |

## Multiplicity and numerical precision

32 exploratory subgroup/period contrasts imply99.84375% intervals, with tail probability0.00078125. At2,000 draws this tail has 1.5625expected observations; at 50,000it has 39.0625. Binomial order-statistic brackets approximate MonteCarlo uncertainty of the conditional bootstrap quantile. They are not confidence intervals for a second market sample. All raw draws, subgroup counts and endpoints are preserved. Borderline endpoints are not treated as sharp discoveries.

The later no-recent-XRP subgroup has a72-hour adjusted lower endpoint near0.0000587, but its endpoint MonteCarlo bracket reaches below zero. High-volatility later deterioration has a positive nominal95%interval but not an adjusted72-hour interval. Later no-current-hour-news and lower-frequency strata have positive adjusted endpoints across all three block choices and their recorded endpoint simulation brackets; they remain exploratory historical descriptions. No tested subgroup has an adjusted improvement at 24, 72 or 168 hours. Curated groups retain24 and 48 origins but insufficient nonempty bootstrap draws, so no interval is fabricated.

## Verification scope and files

- BOOTSTRAP_ALL_CORRECTED.csv: all 135contrast/block rows; paired differences,95%and adjusted endpoints, MonteCarlo brackets and usable counts.
- BOOTSTRAP_DRAWS_*: all50,000raw differences and subgroup counts for each period/block; seed2026.
- RAW_BOOTSTRAP_QUANTILE_VERIFICATION.csv: quantiles independently recalculated from the raw draw arrays and checked against summaries; empty-group intervals remain withheld.
- BOOTSTRAP_ALGORITHM_VERIFICATION.json: block-sum method checked against explicit sampled observations, including final-block truncation.
- PRIMARY_METRIC_REVERIFICATION.csv: primary mean/SD/error/direction recomputed from saved predictions.
- TRAINING_STABILITY_ALL_30_RUNS.csv: every parameter count, checkpoint hash, history hash, selected epoch and actual epoch count.
- SAVED_BATCH_RESPONSE_AUDIT.json:12,818 matching input/output IDs; no missing/extra/duplicate IDs; all12,818 primary schemas and ranges valid; no human semantic claim.
- ATTRIBUTION_REVERIFIED.csv: all10saved passes; the mean news share is23.1119%, not33.8%. Of10,000 explanations,52 remain outside completeness tolerance and are retained. No new gradients were run.
- ATTRIBUTION_EVENT_VERIFICATION.json: actual event-curve source hashes and checks; no retrospective timestamp adjustment.
- NUMERICAL_VERIFICATION.json: all 32 conditional table rows verified;95 previous source hashes unchanged; incomingPDFpreserved.

The inherited technical-control, persistence, event-effect and attribution tables retain their original verified point estimates. Their original source identifiers and metric verification are in the preserved evidence matrix; those files remain unchanged. This revision recalculates bootstrap endpoints and audits saved metadata, not historical training or LLM semantics.
