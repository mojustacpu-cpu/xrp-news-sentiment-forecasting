import sys
from pathlib import Path
b=Path('/Users/mahmoudali/Disk/Master 2025/cursor projects/crypto-forecasting-research/subprojects/codex-xrp-indicators-recommendations');sys.path.insert(0,str(b/'code'))
from common import *
import pickle
p=json.loads((B/'configs/protocol.json').read_text());d=data();r=pd.read_csv(B/'reports/PRICE_RESULTS_ALL_RUNS.csv')
assert len(r)==32 and len(r[(r.fold=='main')&r.seed.notna()])==20 and len(r[(r.fold!='main')&r.seed.notna()])==6
assert sha(B/'data/predictors.csv')==p['source_sha256']==sha(Path(p['source'])/'data/review_candidate/XRP_predictors_through_25Oct_Sydney.csv')
for fold in p['folds']:
 ids=arrays(fold);assert ids['train'][-1]+24<ids['validation'][0] and ids['validation'][-1]+24<ids['evaluation'][0]
 lock=json.loads((B/'runs'/fold/'SELECTION_LOCK.json').read_text());assert lock['protocol_sha256']==sha(B/'configs/protocol.json')
 for path,h in lock['checkpoints_and_selections'].items():assert sha(B/path)==h
 for directory in (B/'runs'/fold).iterdir():
  if not directory.is_dir():continue
  f=pd.read_csv(directory/'evaluation_predictions.csv');ix=ids['evaluation'];assert np.array_equal(pd.to_datetime(f.origin_time,utc=True),d.time.iloc[ix]);assert np.all(pd.to_datetime(f.target_time,utc=True)-pd.to_datetime(f.origin_time,utc=True)==pd.Timedelta(hours=24))
  if directory.name=='garch':
   fit=pickle.load(open(directory/'model.pkl','rb'));returns=100*np.diff(np.log(d.close.to_numpy()));end=len(fit.resid);assert end==ids['train'][-1]+24
   actual=np.array([np.dot(returns[i:i+24],returns[i:i+24]) for i in ix]);assert np.allclose(actual,f.actual_variance)
   w,a,bb=fit.params.to_numpy();v=float(fit.conditional_volatility[-1])**2;pred=[]
   for k in range(end,ix[-1]+1):
    v=w+a*returns[k-1]**2+bb*v
    if k>=ix[0]:
     h=v;total=h
     for j in range(23):h=w+(a+bb)*h;total+=h
     pred.append(total)
   assert np.allclose(pred,f.predicted_variance)
  elif directory.name=='arimax':
   c=json.loads((directory/'COMPLETE.json').read_text());fit=pickle.load(open(directory/(c['selected']['key']+'.pkl'),'rb'));assert fit.nobs==int(ids['train'][-1]+24)-24+1
   candidates=json.loads((directory/'CANDIDATES.json').read_text());assert c['selected']['key']==min([v for v in candidates if v['converged']],key=lambda v:v['rmse'])['key']
  else:
   sc=pickle.load(open(directory/'scalers.pkl','rb'));assert np.allclose(sc['x'].center_,d[sc['features']].iloc[:ids['train'][-1]+1].median())
   hist=pd.read_csv(directory/'history.csv');c=json.loads((directory/'COMPLETE.json').read_text());assert int(hist.loc[hist.validation_mse.idxmin(),'epoch'])==c['selected_epoch']
   assert np.isclose(sc['y'].data_max_[0],d.close.iloc[ids['train']+24].max())
  if directory.name!='garch':
   m=metrics(d.close.iloc[ix+24].to_numpy(),f.predicted_close.to_numpy(),d.close.iloc[ix].to_numpy());saved=json.loads((directory/'EVALUATION.json').read_text());assert all(np.isclose(saved[k],v) for k,v in m.items())
 baseline=pd.read_csv(B/'reports'/f'{fold}_persistence_predictions.csv');assert np.allclose(baseline.predicted_close,d.close.iloc[ix])
save(B/'reports/INDEPENDENT_COMPLETION_AUDIT.json',dict(status='passed',price_rows=32,neural_runs=26,statistical_price_runs=3,persistence_runs=3,garch_runs=3,source_preserved=True,checkpoint_hashes_verified=True,labels_purged=True,exact24h=True,train_only_scalers=True,validation_epochs_and_orders_verified=True,garch_forecasts_independently_reproduced=True))
print('Independent checks passed')
