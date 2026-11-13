from pathlib import Path
import json,hashlib,shutil
import pandas as pd,numpy as np
B=Path(__file__).resolve().parents[2];A=B/'audit';O=A/'results';R=Path('/Users/mahmoudali/Disk/Master 2025/cursor projects/crypto-forecasting-research');P=R/'news-enhanced-crypto-research/phase1';OLD=B.parent/'publication_manuscript_20261108_story_enriched';x=pd.read_csv(O/'BOOTSTRAP_ALL_CORRECTED.csv');st=pd.read_csv(O/'TRAINING_STABILITY_ALL_30_RUNS.csv');summary=pd.read_csv(O/'TRAINING_STABILITY_SUMMARY.csv');primary=pd.read_csv(O/'PRIMARY_METRIC_REVERIFICATION.csv');anno=json.loads((O/'ANNOTATION_EVIDENCE_AUDIT.json').read_text());response=json.loads((O/'SAVED_BATCH_RESPONSE_AUDIT.json').read_text());num=json.loads((O/'NUMERICAL_VERIFICATION.json').read_text());attrs=json.loads((O/'ATTRIBUTION_EVENT_VERIFICATION.json').read_text())
def mdtable(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)+'\n'
issues=[
('C1','Critical','Independent final-period claim','Unresolved; candidly disclosed','Later period was exposed in broader research. Corrected resampling cannot create prospective independence. Retain retrospective label; a genuinely unexposed replication requires future data/protocol approval.'),
('C2','Critical','Independent semantic validation','Pending human work; usable package prepared','Both300-row reviewer files contain zero completed primary labels. Structural JSON validity and205batch/95legacy score consistency are not human accuracy.'),
('M1','Major','Extreme-quantile precision','Corrected from2,000to50,000draws','All32conditional contrasts, original10validation contrasts and later/HBAR comparisons recalculated at24/72/168h. Endpoint MonteCarlo brackets are supplied; no positive subgroup advantage appears.'),
('M2','Major','Information versus model-width advantage','Unresolved; controls specified, not trained','E1B adds6144parameters(1.089%); E2 adds2048(0.363%). No current matched retrained timing-only/shuffled-news/capacity control was found. Frozen masks are not such controls.'),
('M3','Major','Operational article availability','Unresolved; proxy stated','Publication timestamps do not establish first-seen collection or annotation completion. No timing was shifted backwards. Empty bins do not certify no relevant public information.'),
('M4','Major','Early selected epochs','Verified and explained','All30saved checkpoint epochs equal history minima and all histories follow patience15. Training continued16–21epochs; all30end with lower training and higher validation loss relative to the selected epoch. Consistent with post-selection overfitting, not proof of global convergence.'),
('M5','Major','Later negative findings and subgroup claims','Preserved and statistically rechecked','E1B and h6later deterioration have positive nominal95%intervals at all3block lengths. Subgroup strength varies; high-volatility deterioration is not72h multiplicity-adjusted evidence. No universal future/causal claim.'),
('M6','Major','Attribution and event interpretation','Verified; limitations retained','All10saved attribution arrays rechecked without gradients;23.1119%mean news share,52failed tolerance cases retained. All5event curves traceable; training-period charts are not forecast wins. No causal claims.'),
('m1','Minor','BiLSTM readout explanation','Clarified; architecture unchanged','Last output differs from both final directional hidden states; stacked lower-layer context remains. Do not relabel a retrained readout with old metrics.'),
('m2','Minor','Dissertation versus revised protocol','Clarified','Retained indicators distinguished from this realBTC/logreturn/exact24h/train-only/five-seed programme; unmatched historic scores not pooled.'),
('m3','Minor','Batch request/response counts','Rechecked and clarified','12818unique request IDs match12818unique response IDs;12816distinct normalised input headlines. All response contents parse and primary fields satisfy ranges. Repeat text is not duplicate response IDs.'),
('m4','Minor','Compilation and references','Checked in delivery QA','Relative figure paths and27retained references; no new invented reference or DOI. Internal CASfrontmatter box warning is documented after visual inspection.')]
review='# Final scientific review report\n\nClassification: **Requires major revisions** for a strong generalisable sentiment-enhancement claim. This is a critical assistant review, not external human peer review or journal acceptance advice. A transparent negative-result study remains scientifically useful; the outstanding issues concern scope and evidence rather than a requirement to manufacture a winning model.\n\n'+mdtable(['ID','Severity','Issue','Resolution status','Evidence and action'],issues)+'''
The corrected bootstrap preserves the published point forecasts, seeds, model selection and data. It improves simulation precision, not market independence. Bonferroni uses32stratum/period contrasts, each with alpha/32 two-sided error allocation; a union bound does not require independent groups. Approximate coverage still relies on valid marginal bootstrap intervals and a sufficiently stable historical process. The original10contrast validation adjustment is retained although the four half-lives are hyperparameter settings of one method. Different stage families do not correct accumulated validation reuse across the entire research programme.

No adjusted improvement is established at any checked block length. Some later negative conditional differences have positive adjusted lower endpoints, but sparse strata, overlap, endpoint MonteCarlo uncertainty and sensitivity to block length prevent universal regime claims. Current-hour absence is not absence throughout a30-hour sequence. Curated event intervals remain not estimable under the existing95%usable-resample rule.

The evidence directly supports the later error reversal, correct checkpoint selection, structural response validity, publication-proxy limitations and model sensitivity to news. It does not identify which mechanism caused the reversal. No model was retrained, no API was called and no recurring monitoring was created.

A strong submission should complete independent human semantic labels and strengthen independent chronological replication and semantic-versus-timing controls. If submitted before those tasks, the manuscript must retain its limited retrospective/negative-result framing and refrain from semantic accuracy or reliable forecasting-superiority claims. Author approval of declarations, data licensing and final journal format remains required.
'''
(B/'SCIENTIFIC_REVIEW_REPORT.md').write_text(review)
# Numerical report with exact source paths, preserved non-positive results, and Monte Carlo precision.
rows=[]
for _,r in primary.iterrows():rows.append([r.period,r.model,f'{r.rmse_mean:.9f}',f'{r.rmse_sd:.9f}',f'{r.mae:.9f}',f'{r.r2:.9f}',f'{100*r.direction:.3f}%'])
text='# Numerical verification report\n\nStatus: all executed numerical checks pass. No forecast point estimate was changed.\n\n'+mdtable(['Period','Model','RMSE mean','Seed SD','MAE','R2','Direction'],rows)
text+='\n## Corrected primary risk intervals\n\nRisk is the square root of mean-seed squared loss over matched origins. It differs from the arithmetic mean seed RMSE used for configuration ranking. The table is in USDT; a negative difference favours news. All use50,000draws and72-hour paired calendar blocks.\n\n'
rr=[]
for period,group in [('validation','E1B vs E0'),('validation','E2_h6 vs E0'),('later','E1B vs E0'),('later','E2_h6 vs E0'),('HBAR','HBAR_E1B vs technical')]:
 r=x[(x.period==period)&(x.group==group)&(x.block_hours==72)].iloc[0];rr.append([period,group,f'{r.difference:+.9f}',f'[{r.ci95_low:.9f}, {r.ci95_high:.9f}]',f'[{r.adjusted_low:.9f}, {r.adjusted_high:.9f}]',int(r.family)])
text+=mdtable(['Period','Contrast','Risk difference','95% interval','Stage-adjusted interval','Family'],rr)
text+='''
## Multiplicity and numerical precision

32exploratory subgroup/period contrasts imply99.84375%intervals, with tail probability0.00078125. At2,000draws this tail has1.5625expected observations; at50,000it has39.0625. Binomial order-statistic brackets approximate MonteCarlo uncertainty of the conditional bootstrap quantile. They are not confidence intervals for a second market sample. All raw draws, subgroup counts and endpoints are preserved. Borderline endpoints are not treated as sharp discoveries.

The later no-recent-XRP subgroup has a72h adjusted lower endpoint near0.0000587, but its endpoint MonteCarlo bracket reaches below zero. High-volatility later deterioration has a positive nominal95%interval but not an adjusted72h interval. Later no-current-hour-news and lower-frequency strata have positive adjusted endpoints across all3block choices and their recorded endpoint simulation brackets; they remain exploratory historical descriptions. No tested subgroup has an adjusted improvement at24,72or168h. Curated groups retain24and48origins but insufficient nonempty bootstrap draws, so no interval is fabricated.

## Verification scope and files

- BOOTSTRAP_ALL_CORRECTED.csv: all135contrast/block rows; paired differences,95%and adjusted endpoints, MonteCarlo brackets and usable counts.
- BOOTSTRAP_DRAWS_*: all50,000raw differences and subgroup counts for each period/block; seed2026.
- RAW_BOOTSTRAP_QUANTILE_VERIFICATION.csv: quantiles independently recalculated from the raw draw arrays and checked against summaries; empty-group intervals remain withheld.
- BOOTSTRAP_ALGORITHM_VERIFICATION.json: block-sum method checked against explicit sampled observations, including final-block truncation.
- PRIMARY_METRIC_REVERIFICATION.csv: primary mean/SD/error/direction recomputed from saved predictions.
- TRAINING_STABILITY_ALL_30_RUNS.csv: every parameter count, checkpoint hash, history hash, selected epoch and actual epoch count.
- SAVED_BATCH_RESPONSE_AUDIT.json:12818matching input/output IDs; no missing/extra/duplicate IDs; all12818primary schemas and ranges valid; no human semantic claim.
- ATTRIBUTION_REVERIFIED.csv: all10saved passes; the mean news share is23.1119%, not33.8%. Of10000explanations,52remain outside completeness tolerance and are retained. No new gradients were run.
- ATTRIBUTION_EVENT_VERIFICATION.json: actual event-curve source hashes and checks; no retrospective timestamp adjustment.
- NUMERICAL_VERIFICATION.json: all32conditional table rows verified;95previous source hashes unchanged; incomingPDFpreserved.

The inherited technical-control, persistence, event-effect and attribution tables retain their original verified point estimates. Their original source identifiers and metric verification are in the preserved evidence matrix; those files remain unchanged. This revision recalculates bootstrap endpoints and audits saved metadata, not historical training or LLM semantics.
'''
(B/'NUMERICAL_VERIFICATION_REPORT.md').write_text(text)
# Checklist is deliberately explicit about missing evidence.
items=[
('Source market data','Verified/reused','26337continuous hourly inputs;2023-10-2305UTC–2026-10-2413UTC; future-target archive25Oct2026','phase1/E0_E1_REUSE.json; previous source inventory'),
('Target and splits','Verified/reused','30completed hours; exact+24h logreturn;18861train/2136validation/5262later; chronological purges','phase1/main_*.npy; frozen origin CSVs'),
('RealBTC and indicators','Verified/reused','19technical/market inputs; real BTC and nine indicators unchanged','old full-hour/source verification; hashes unchanged'),
('Scalers','Verified/reused','Training-input rows and training targets only; no later refitting','frozen scaler manifests and previous inference verification'),
('Models and seeds','Verified','30primary checkpoints; parameters564353/570497/566401; seeds42,123,2026,7,99','TRAINING_STABILITY_ALL_30_RUNS.csv'),
('Checkpoint criterion','Verified','Every checkpoint matches validation MSE minimum and patience15','same history/metadata table'),
('Predictions','Verified/reused','No retraining; primary metrics recomputed from saved arrays','PRIMARY_METRIC_REVERIFICATION.csv'),
('Statistics','Corrected','50000paired resamples at24/72/168h; original contrast families retained','BOOTSTRAP_ALL_CORRECTED.csv; raw NPZ and script'),
('Subgroups','Verified','Definitions and sample counts unchanged; training-derived thresholds; no outcome-driven mask changes','bootstrap_audit.py; previous protocol'),
('News annotation structure','Verified within saved batch','12818matching unique request/response IDs; valid primary fields','SAVED_BATCH_RESPONSE_AUDIT.json'),
('300sample provenance','Recovered evidence, limited','205batch links;95legacyJSONmatches; legacy snapshot unknown','original provenance_v2 records'),
('Human semantic labels','Missing','Both reviewer files have zero completed primary labels','ANNOTATION_EVIDENCE_AUDIT.json; new blinded protocol'),
('Operational first-seen timestamps','Missing','Publication is a proxy; collection/annotation latency not fully recorded','original archive flags and event/source records'),
('Repeatability','Missing','Historical consistency is not repeated freshAPIcalls; none authorised','original repeatability status'),
('Matched retrained controls','Missing','No current timing-only/shuffled-news/near-capacity study','CONTROL_EVIDENCE_AUDIT.json; proposed configs'),
('Attribution','Verified with unresolved precision','10arrays,1000origins/pass,32training references;52residual failures retained','ATTRIBUTION_REVERIFIED.csv'),
('Event curves','Verified descriptively','5genuine cases; official exact times or date bands; no retiming','ATTRIBUTION_EVENT_VERIFICATION.json'),
('Prospective independence','Not established','Later historical period had prior exposure; no pristine deployment claim','manuscript limitation and previous exposure report'),
('Figures and references','Available; delivery QA','Corrected intervals in figures6/9; other verified graphics retained;27references','figure code and compiled project'),
('Public release and licensing','Author action required','Original data/source rights and repository release not invented','data availability declaration; author approval')]
pd.DataFrame(items,columns=['item','status','finding','evidence']).to_csv(B/'REPRODUCIBILITY_CHECKLIST.csv',index=False);(B/'REPRODUCIBILITY_CHECKLIST.md').write_text('# Reproducibility checklist\n\n'+mdtable(['Item','Status','Finding','Evidence'],items))
# Specific training proposals: no executable run launched and no implicit approval.
features=json.loads((P/'E1B/CONFIG.json').read_text())['features'];times=[float(pd.read_csv(p).elapsed_seconds.iloc[-1]) for p in st[st.model.eq('E1B')].history];estimate=float(np.median(times)*5/60)
common=dict(seeds=[42,123,2026,7,99],window_hours=30,horizon_hours=24,target='log(close[t+24]/close[t]);reconstruct close[t]*exp(predicted_return)',technical_features=features[:19],data=str(P/'E1B/training.csv'),data_sha256=hashlib.sha256((P/'E1B/training.csv').read_bytes()).hexdigest(),split_sources=[str(P/f'main_{k}.npy') for k in ['train','validation']],scaling='training-only RobustScaler inputs and MinMaxScaler return target',network='two-layer BiLSTM128perdirection; last-time output; dropout.2',optimizer='Adam lr.001 batch32',max_epochs=200,patience=15,epoch_selection='minimum validation scaled-return MSE',final_period='do not load for fitting or configuration selection; any later control comparison stays retrospective',API_calls=0)
controls=dict(status='PROPOSED_NOT_APPROVED_NOT_RUN',primary=dict(name='C1_timing_only',jobs=5,features=features[:19]+features[-2:],parameters=566401,dense_units=64,estimated_minutes_for_5=round(estimate,1),resource_basis='5times median saved E1B training duration; approximate machine-dependent estimate',purpose='separate four GPTsemantics from presence/recency; E0/E1B remain frozen comparison arms'),optional_capacity=dict(name='C2_near_capacity_technical',jobs=5,features=features[:19],dense_units=88,parameters=570545,difference_from_E1B=48,purpose='active technical-only head capacity within48parameters of E1B; not exact architecture matching'),common=common,rolling_replication='Requires separately frozen fold dates and nested selection before a new approval; existing historical exposure remains disclosed',shuffled_semantic_control='Requires a separate protocol for training-only permutations and repeated permutation seeds; no frozen mask is substituted for retraining')
(A/'PROPOSED_CONTROLS_REQUIRING_APPROVAL.json').write_text(json.dumps(controls,indent=2))
(B/'OUTSTANDING_WORK.md').write_text(f'''# Outstanding work — not completed or silently launched

1. Two real independent human annotators must complete the prepared300headline files. Preserve originals and adjudication; no labels can be supplied by the assistant as human work.
2. Approve the primary timing-only control before training: five fits, same frozen technical data/return target/windows/splits/seeds,21inputs,566401parameters. Estimated total approximately{estimate:.1f}minutes based on median saved E1B durations; actual hardware/early stopping varies. No paidAPIcalls. E0/E1B are reused, not retrained. Full specification is audit/PROPOSED_CONTROLS_REQUIRING_APPROVAL.json.
3. Optional near-capacity technical control: five fits with dense88and570545parameters, within48of E1B. Not approved or started.
4. Independent chronological replication needs a separately locked rolling/prospective plan. Earlier researcher exposure cannot be undone by relabelling the current later block. No such training was initiated.
5. Establish operational first-seen and annotation-completion timestamps for prospective use. Missing historical availability cannot be repaired by moving stories to price turns.
6. True annotation repeatability needs fresh pinned calls and separate cost approval. None were made.
7. Featurewise attribution convergence and52remaining residual failures are unresolved. They remain disclosed; no unnecessary gradient passes were started.
8. Authors must approve declarations, data licensing/release, journal fit and final submission. No public repository or human validation was invented.

Current classification: Requires major revisions. A bounded retrospective negative-result paper can still be useful, but a reliable generalisable sentiment-enhancement or human-accuracy claim remains unsupported.
''')
(B/'CHANGE_LOG.md').write_text('''# Precise change log relative to the incoming enriched manuscript

- Preserved incoming LaTeX/PDF and all original experiments in place; copied source into a separate final_review folder.
- Section4.5validation intervals, later primary/supplementary intervals and HBAR interval recomputed with50000draws. Point estimates, seeds, hyperparameter ranking, selected models and all adverse results unchanged.
- Section4.8now explains Bonferroni's alpha/32 union bound, dependent strata,50k tail count, endpoint MonteCarlo brackets, block sensitivity and sparse-group withholding. Tables17–18and figures6/9use corrected endpoints.
- Added an actual30history checkpoint audit: selected1–6epochs versus16–21completed; all patience/minimum checks verified; loss trends interpreted cautiously.
- Clarified revised-protocol versus dissertation scope without importing old numerical results.
- Clarified exact last-time BiLSTM readout and parameter deltas. No architecture was changed.
- Re-audited12818saved request/response pairs and clarified12816distinct-headline count. No semantic accuracy invented; legacy model provenance limitations retained.
- Supplied a four-trained-field-only blinded annotation protocol, two blank reviewer files, protected answer key, rubric, empty evaluation template and scoring code. Empty-label test correctly stays pending.
- Reverified ten saved attribution arrays and five event curves; no new gradients, inference/training or timestamp changes. Frozen masks explicitly distinguished from retraining.
- Added review/numerical/checklist/outstanding-work reports, reproducible audit scripts, all raw50000bootstrap draws and approval-required control specifications.
- Retained eight main sections,15main+2appendix figures,27references, original technical definitions, six-hour decay interpretation, negative later findings and limited HBAR transfer. No paidAPIor monitoring started.
''')
# Update provenance matrix while preserving original rows as original-evidence pointers.
trace=pd.read_csv(OLD/'EVIDENCE_TRACEABILITY_MATRIX.csv');extra=[]
for ident,claim,file in [('R01','All corrected CI endpoints and MC brackets',O/'BOOTSTRAP_ALL_CORRECTED.csv'),('R02','Every selected epoch and parameter count',O/'TRAINING_STABILITY_ALL_30_RUNS.csv'),('R03','Saved batch ID/schema/range audit',O/'SAVED_BATCH_RESPONSE_AUDIT.json'),('R04','Human label status',O/'ANNOTATION_EVIDENCE_AUDIT.json'),('R05','Primary metric point verification',O/'PRIMARY_METRIC_REVERIFICATION.csv'),('R06','Attribution/event verification',O/'ATTRIBUTION_EVENT_VERIFICATION.json')]:extra.append(dict(id=ident,claim=claim,source=str(file),source_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),reproduction='audit/code',scope='new low-cost verification; original model forecasts unchanged'))

for idx,r in trace.iterrows():
 if str(r['id']).startswith('fig'):
  pp=B/'figures'/(str(r['id'])+'.pdf')
  if pp.exists():trace.at[idx,'source']=str(pp);trace.at[idx,'source_sha256']=hashlib.sha256(pp.read_bytes()).hexdigest()
  if r['id'] in ['fig06_paired_intervals','fig09_conditional_forest']:trace.at[idx,'reproduction']='audit/code/figures_corrected.py; corrected50k draws'
 if r['id'] in ['C03','C05']:trace.at[idx,'scope']=str(r['scope'])+'; original interval endpoints superseded by R01'
pd.concat([trace,pd.DataFrame(extra)],ignore_index=True).to_csv(B/'EVIDENCE_TRACEABILITY_MATRIX.csv',index=False)
print('REPORTS COMPLETE; proposed C1 minutes',estimate)

# Improve spacing in prose without changing scientific values or machine-readable records.
replacements={'all32':'all 32','all30':'all 30','All30':'All 30','30saved':'30 saved','original10':'original 10','all135':'all 135','12818matching':'12,818 matching','12818unique':'12,818 unique','12816distinct':'12,816 distinct','12818primary':'12,818 primary','All response':'All response','50000paired':'50,000 paired','50000draws':'50,000 draws','50000bootstrap':'50,000 bootstrap','50,000draws':'50,000 draws','32exploratory':'32 exploratory','2,000draws':'2,000 draws','2,000to50,000draws':'2,000 to 50,000 draws','at2,000':'at 2,000','at50,000':'at 50,000','tail has1.5625':'tail has 1.5625','has39.0625':'has 39.0625','99.84375%intervals':'99.84375% intervals','at24,72or168h':'at 24, 72 or 168 hours','at24/72/168h':'at 24/72/168 hours','all3block':'all three block','72h adjusted':'72-hour adjusted','72h interval':'72-hour interval','72h multiplicity':'72-hour multiplicity','24and48origins':'24 and 48 origins','contains zero':'contains zero','95%usable':'95% usable','15epochs':'15 epochs','patience15':'patience 15','16–21epochs':'16–21 epochs','selected1–6epochs':'selected 1–6 epochs','versus16–21completed':'versus 16–21 completed','Both300-row':'Both 300-row','two300-row':'two 300-row','prepared300headline':'prepared 300-headline','four-trained-field-only':'four-field','21inputs':'21 inputs','566401parameters':'566,401 parameters','570545parameters':'570,545 parameters','dense88and':'dense 88 and ','within48of':'within 48 of','5times':'five times','E1Btraining':'E1B training','95previous':'95 previous','all32conditional':'all 32 conditional','12818saved':'12,818 saved','300headline':'300-headline','six-hour decay interpretation':'six-hour decay interpretation','15main+2appendix':'15 main and 2 appendix','27references':'27 references','realBTC':'real BTC','logreturn':'log return','exact24h':'exact +24-hour','E0/E1B/E2origins':'E0/E1B/E2 origins','10arrays,1000origins/pass,32training':'10 arrays, 1,000 origins/pass, 32 training','52residual':'52 residual','205batch':'205 batch','95legacy':'95 legacy','95legacyJSONmatches':'95 legacy JSON matches','12,818unique':'12,818 unique','No paidAPI':'No paid API','APIor':'API or','10000explanations':'10,000 explanations','52remain':'52 remain','23.1119%mean':'23.1119% mean','12818primary schemas':'12,818 primary schemas','24/72/168h':'24/72/168 hours','same300':'same 300'}
for name in ['SCIENTIFIC_REVIEW_REPORT.md','NUMERICAL_VERIFICATION_REPORT.md','REPRODUCIBILITY_CHECKLIST.md','CHANGE_LOG.md','OUTSTANDING_WORK.md']:
 pp=B/name;txt=pp.read_text()
 for a,b in sorted(replacements.items(),key=lambda kv:-len(kv[0])):txt=txt.replace(a,b)
 pp.write_text(txt)
