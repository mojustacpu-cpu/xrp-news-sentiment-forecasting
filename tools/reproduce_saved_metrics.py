"""Read-only metric reproduction; never trains or modifies archived results."""
from pathlib import Path
import argparse,json,re
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def metric(y,p,origin=None):
 e=p-y;sst=((y-y.mean())**2).sum()
 return dict(n=len(y),rmse=float(np.sqrt(np.mean(e*e))),mae=float(np.mean(abs(e))),r2=float(1-np.sum(e*e)/sst),directional_accuracy_percent=None if origin is None else float(100*np.mean(np.sign(p-origin)==np.sign(y-origin))))
def main():
 a=argparse.ArgumentParser();a.add_argument('--output',default='reproduced');args=a.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True);rows=[];skipped=[];variance_rows=[]
 for f in sorted((ROOT/'subprojects').rglob('*predictions.csv')):
  d=pd.read_csv(f).rename(columns={'actual_close':'actual','predicted_close':'predicted'})
  if {'actual_variance','predicted_variance'}.issubset(d.columns):
   y=d.actual_variance.to_numpy();v=d.predicted_variance.to_numpy();assert np.all(v>0)
   variance_rows.append(dict(file=f.relative_to(ROOT).as_posix(),n=len(y),variance_mae=float(abs(v-y).mean()),variance_rmse=float(np.sqrt(((v-y)**2).mean())),qlike=float((np.log(v)+y/v).mean())))
   continue
  if not {'actual','predicted'}.issubset(d.columns):skipped.append(f.relative_to(ROOT).as_posix());continue
  if '15_xrp_garch_volatility' in str(f):continue
  if d[['actual','predicted']].isna().any().any():continue
  origin=next((d[c].to_numpy() for c in ['origin_close','origin_price','close_at_origin','current_close','origin'] if c in d),None)
  r=metric(d.actual.to_numpy(),d.predicted.to_numpy(),origin)
  r.update(file=f.relative_to(ROOT).as_posix(),project=f.relative_to(ROOT).parts[1],seed=(re.search(r'seed(\d+)',str(f)).group(1) if re.search(r'seed(\d+)',str(f)) else None),partition='evaluation' if 'evaluation' in f.name else 'validation')
  rows.append(r)
 pd.DataFrame(rows).to_csv(out/'PER_FILE_PRICE_METRICS.csv',index=False)
 pd.DataFrame(variance_rows).to_csv(out/'SEPARATE_VARIANCE_METRICS.csv',index=False)
 # Canonical matched arrays reproduce all main validation candidates and later XRP metrics.
 matches=[('validation',ROOT/'analysis/phase1/VALIDATION_PREDICTIONS_ALL.npz'),('later',ROOT/'subprojects/16_xrp_frozen_later_evaluation/results/PREDICTIONS_ARRAYS.npz')]
 allrows=[]
 for period,path in matches:
  z=np.load(path,allow_pickle=False);y=z['actual'];origin=z['origin'];allrows.append(dict(period=period,model='Persistence',seed='deterministic',**metric(y,origin,None)))
  for name in z.files:
   v=z[name]
   if v.ndim==2 and v.shape[1]==len(y) and v.shape[0]==5 and np.issubdtype(v.dtype,np.number):
    for seed,p in zip([42,123,2026,7,99],v):allrows.append(dict(period=period,model=name,seed=str(seed),**metric(y,p,origin)))
 df=pd.DataFrame(allrows);df.to_csv(out/'MATCHED_PER_SEED_METRICS.csv',index=False)
 summary=[]
 for (period,model),g in df.groupby(['period','model'],sort=False):
  summary.append(dict(period=period,model=model,seeds=len(g),rmse_mean=float(g.rmse.mean()),rmse_sample_sd=None if len(g)==1 else float(g.rmse.std(ddof=1)),mae_mean=float(g.mae.mean()),r2_mean=float(g.r2.mean()),directional_accuracy_percent=None if model=='Persistence' else float(g.directional_accuracy_percent.mean())))
 pd.DataFrame(summary).to_csv(out/'MATCHED_SUMMARY.csv',index=False)
 # Check the unchanged published rounding, not an improvement claim.
 def check(period,model,value):
  x=next(x for x in summary if x['period']==period and x['model']==model);assert abs(x['rmse_mean']-value)<.00000051,(period,model,x['rmse_mean'],value)
 for row in [('validation','E0',.063935),('validation','E1B',.063036),('validation','E2_h6',.063637),('later','E0',.039976),('later','E1B',.042118),('later','Persistence',.039341)]:check(*row)
 # Independently recompute the saved attribution percentages from all ten arrays.
 eig=ROOT/'subprojects/17_expected_integrated_gradients/results';shares=[]
 news={'direct_sentiment','direct_specificity','direct_stance','direct_uncertainty','direct_news_present','hours_since_latest_news_capped48'}
 for f in sorted(eig.glob('E1B_seed*_background*.npz')):
  z=np.load(f,allow_pickle=False);v=z['values'];imp=abs(v).sum(1).mean(0);features=z['features'].astype(str);share=100*sum(a for a,n in zip(imp,features) if n in news)/imp.sum();failed=int((abs(z['residual'])>1e-5+.01*abs(z['output_difference'])).sum());shares.append(dict(file=f.name,news_share_percent=float(share),technical_share_percent=float(100-share),failed_completeness_tolerance=failed))
 assert len(shares)==10
 pd.DataFrame(shares).to_csv(out/'ATTRIBUTION_RECALCULATED.csv',index=False)
 mean=float(np.mean([x['news_share_percent'] for x in shares]));assert abs(mean-23.11)<.005
 report=dict(status='passed',prediction_csv_metrics=len(rows),separate_variance_rows=len(variance_rows),matched_seed_rows=len(df),matched_summaries=len(summary),attribution_passes=len(shares),news_attribution_share_percent=mean,attribution_tolerance_failures=sum(x['failed_completeness_tolerance'] for x in shares),training_runs=0,api_calls=0,original_absolute_paths_required=False,skipped_non_price_csvs=skipped)
 (out/'VERIFICATION.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
