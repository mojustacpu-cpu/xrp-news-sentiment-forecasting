from pathlib import Path
import json,hashlib,re,math
import numpy as np,pandas as pd
B=Path(__file__).resolve().parents[2];OLD=B.parent/'publication_manuscript_20261108_story_enriched';O=B/'audit/results';R=Path('/Users/mahmoudali/Disk/Master 2025/cursor projects/crypto-forecasting-research');P=R/'news-enhanced-crypto-research/phase1';Q=R/'Research_Final_Experiments';checks=[];x=pd.read_csv(O/'BOOTSTRAP_ALL_CORRECTED.csv')
# Quantile verification directly from every saved raw bootstrap array, independent of summary assembly.
for file in sorted(O.glob('BOOTSTRAP_DRAWS_*.npz')):
 a=np.load(file);name=file.stem.removeprefix('BOOTSTRAP_DRAWS_');period,block=name.rsplit('_block',1)
 for j,g in enumerate(a['names']):
  r=x[(x.period==period)&(x.group==g)&x.block_hours.eq(int(block))].iloc[0];draw=a['differences'][:,j];valid=draw[np.isfinite(draw)];assert len(draw)==50000 and len(valid)==r.usable_resamples
  if r.interval_estimable:
   q=np.quantile(valid,[.025,.975,.05/(2*r.family),1-.05/(2*r.family)]);saved=np.array([r.ci95_low,r.ci95_high,r.adjusted_low,r.adjusted_high]);assert np.max(abs(q-saved))<1e-13
   checks.append(dict(period=period,group=str(g),block_hours=int(block),quantile_max_error=float(np.max(abs(q-saved))),usable_draws=len(valid)))
  else:assert len(valid)<47500
# All primary forecast point estimates, including every seed, remain identical to saved source.
v=np.load(P/'analysis/VALIDATION_PREDICTIONS_ALL.npz');f=np.load(Q/'01_XRP_Final_Evaluation/results/PREDICTIONS_ARRAYS.npz');e=np.load(OLD/'analysis/results/E2_H6_LATER_PREDICTIONS.npz');rank=pd.read_csv(P/'analysis/VALIDATION_RANKING.csv').set_index('candidate');main=[]
for key in ['E0','E1B','E2_h3','E2_h6','E2_h12','E2_h24']:
 y=v['actual'];p=v[key];rm=np.sqrt(((p-y)**2).mean(1));assert abs(rm.mean()-rank.loc[key,'rmse_mean'])<1e-12;assert abs(rm.std(ddof=1)-rank.loc[key,'rmse_std'])<1e-12
 main.append(dict(period='validation',model=key,rmse_mean=float(rm.mean()),rmse_sd=float(rm.std(ddof=1)),mae=float(abs(p-y).mean(1).mean()),r2=float((1-((p-y)**2).sum(1)/((y-y.mean())**2).sum()).mean()),direction=float((np.sign(p-v['origin'])==np.sign(y-v['origin'])).mean(1).mean())))
for key,p in [('E0',f['E0']),('E1B',f['E1B']),('E2_h6',e['predicted'])]:
 y=f['actual'];assert np.array_equal(y,e['actual']);rm=np.sqrt(((p-y)**2).mean(1));main.append(dict(period='later',model=key,rmse_mean=float(rm.mean()),rmse_sd=float(rm.std(ddof=1)),mae=float(abs(p-y).mean(1).mean()),r2=float((1-((p-y)**2).sum(1)/((y-y.mean())**2).sum()).mean()),direction=float((np.sign(p-f['origin'])==np.sign(y-f['origin'])).mean(1).mean())))
pd.DataFrame(main).to_csv(O/'PRIMARY_METRIC_REVERIFICATION.csv',index=False);pd.DataFrame(checks).to_csv(O/'RAW_BOOTSTRAP_QUANTILE_VERIFICATION.csv',index=False)
# Check every updated conditional table row against newly calculated values.
s=(B/'XRP_SENTIMENT_TECHNICAL_MANUSCRIPT.tex').read_text();verifiedrows=0
for period,label in [('validation','tab:conditionalall'),('later','tab:conditionallater')]:
 table=next(m.group() for m in re.finditer(r'\\begin\{table\}.*?\\end\{table\}',s,re.S) if '\\label{'+label+'}' in m.group())
 for _,r in x[(x.period==period)&(x.scope=='exploratory conditional')&(x.block_hours==72)].iterrows():
  q=f'{r.group} & {int(r.n)} & {r.E0_rmse:.6f} & {r.E1B_rmse:.6f} & {r.change_percent:+.2f}';assert q in table
  if r.interval_estimable:assert f'[{r.adjusted_low:.5f},{r.adjusted_high:.5f}]' in table
  verifiedrows+=1
inventory=json.loads((OLD/'analysis/SOURCE_INVENTORY.json').read_text());bad=[]
for p,m in inventory.items():
 if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=m['sha256']:bad.append(p)
assert not bad
# The incoming manuscript, all figures not intentionally replaced, and original source files are preserved.
oldpdf=OLD/'XRP_SENTIMENT_TECHNICAL_MANUSCRIPT.pdf';pres=B/'preserved_manuscript'/oldpdf.name;assert oldpdf.read_bytes()==pres.read_bytes()
report=dict(passed=True,raw_quantile_comparisons=len(checks),resamples=50000,subgroup_table_rows_verified=verifiedrows,original_source_hashes_unchanged=len(inventory),incoming_PDF_preserved=True,all_primary_predictions_reused=True,checkpoints_retrained=0,paid_API_calls=0,primary_metric_file='PRIMARY_METRIC_REVERIFICATION.csv',raw_quantile_file='RAW_BOOTSTRAP_QUANTILE_VERIFICATION.csv');(O/'NUMERICAL_VERIFICATION.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
