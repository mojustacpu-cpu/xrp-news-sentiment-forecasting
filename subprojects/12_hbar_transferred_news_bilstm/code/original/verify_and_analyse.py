from pathlib import Path
import json,hashlib,sys,math
import pandas as pd,numpy as np,joblib,torch
from sklearn.preprocessing import RobustScaler,MinMaxScaler
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P/'reference'));from neural import Model
from common import metrics
O=P/'analysis';O.mkdir(exist_ok=True);torch.set_num_threads(4)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,default=str,allow_nan=False))
frozen=json.loads((P/'FROZEN_MANIFEST.json').read_text());assert all(sha(p)==h for p,h in frozen['files'].items())
bv=json.loads((P/'BASELINE_REUSE_VERIFICATION.json').read_text());reuse={'HBAR_technical':dict(seeds=bv['seeds'])};base=pd.read_csv(bv['dataset'],float_precision='round_trip');cfg0=json.loads((P/'HBAR_E1B/CONFIG.json').read_text());tech=cfg0['features'][:19];ids={k:np.load(P/f'main_{k}.npy') for k in ['train','validation','evaluation']};v=ids['validation'];assert len(v)==2136
fit=np.unique((ids['train'][:,None]-np.arange(30)).ravel());z=base.iloc[v];actual=z.target_close_plus24h.to_numpy();origin=z.close.to_numpy();av=pd.to_datetime(base.available_at_utc,utc=True);ta=pd.to_datetime(base.target_available_at_utc,utc=True)
assert (ta-av).eq(pd.Timedelta(hours=24)).all() and av.diff().dropna().eq(pd.Timedelta(hours=1)).all();assert ta.iloc[ids['train'][-1]]<av.iloc[v[0]] and ta.iloc[v[-1]]<av.iloc[ids['evaluation'][0]]
news=pd.read_csv(P.parent/'phase1/E1B/training.csv',float_precision='round_trip');hd=pd.read_csv(P/'HBAR_E1B/training.csv',float_precision='round_trip');assert np.array_equal(hd[cfg0['news_features']].values,news[cfg0['news_features']].values);assert np.array_equal(hd.BTC_Close,news.BTC_Close)
rows=[];predictions={};checks=[];seedlist=[42,123,2026,7,99];candidates=['HBAR_E1B']
for name in ['HBAR_technical']+candidates:
 ps=[]
 if name in reuse:
  details=reuse[name]['seeds'];cfg=None
 else:
  cfg=json.loads((P/name/'CONFIG.json').read_text());d=pd.read_csv(cfg['dataset'],float_precision='round_trip');sc=joblib.load(cfg['scalers']);f=cfg['features'];assert np.array_equal(d[tech].values,base[tech].values);assert d.available_at_utc.equals(base.available_at_utc) and d.target_available_at_utc.equals(base.target_available_at_utc)
  assert d.iloc[ids['evaluation']].target_close_plus24h.isna().all();assert np.array_equal(d.iloc[v].target_close_plus24h,actual)
  sx=RobustScaler().fit(d.iloc[fit][f]);sy=MinMaxScaler().fit(np.log(base.iloc[ids['train']].target_close_plus24h/base.iloc[ids['train']].close).to_numpy()[:,None])
  for attr in ['center_','scale_']:assert np.array_equal(getattr(sx,attr),getattr(sc['input_scaler'],attr))
  for attr in ['scale_','min_']:assert np.array_equal(getattr(sy,attr),getattr(sc['target_scaler'],attr))
  assert np.array_equal(sc['fit_input_rows'],fit) and np.array_equal(sc['fit_target_origins'],ids['train']);x=sc['input_scaler'].transform(d[f]);details=[dict(seed=s,checkpoint=str(P/name/'runs'/f'{name}_seed{s}'/'best.pt'),predictions=str(P/name/'runs'/f'{name}_seed{s}'/'validation_predictions.csv')) for s in seedlist]
 for item in details:
  seed=item['seed'];path=Path(item['predictions']);r=path.parent;pred=pd.read_csv(path,float_precision='round_trip');assert pd.to_datetime(pred.origin_time,utc=True).reset_index(drop=True).equals(av.iloc[v].reset_index(drop=True));assert pd.to_datetime(pred.target_time,utc=True).reset_index(drop=True).equals(ta.iloc[v].reset_index(drop=True));assert np.array_equal(pred.actual,actual)
  saved=json.loads((r/'COMPLETE.json').read_text());m=metrics(actual,pred.predicted.to_numpy(),origin)
  for k in ['rmse','mae','r2','directional_accuracy']:assert np.isclose(m[k],saved[k],atol=1e-10,rtol=0)
  ck=torch.load(item['checkpoint'],map_location='cpu',weights_only=False);hist=pd.read_csv(r/'history.csv');assert int(hist.loc[hist.validation_mse.idxmin(),'epoch'])==ck['epoch']==saved['selected_epoch'];assert hist.epoch.tolist()==list(range(1,len(hist)+1))
  if cfg:
   assert ck['features']==f and ck['seed']==seed;model=Model(len(f),dict(hidden=128,bidirectional=True));model.load_state_dict(ck['model']);model.eval();out=[]
   with torch.no_grad():
    for j in range(0,len(v),256):out.extend(model(torch.tensor(np.stack([x[i-29:i+1] for i in v[j:j+256]]),dtype=torch.float32)).numpy().ravel())
   fresh=origin*np.exp(sc['target_scaler'].inverse_transform(np.array(out)[:,None]).ravel());err=float(np.max(abs(fresh-pred.predicted)));assert err<1e-7
   assert sum(p.numel() for p in model.parameters())==cfg['parameters'];assert ck['plan_sha256']==sha(r.parent/'PLAN.json')
   loglines=(P/('execution.log' if seed in [42,123,2026] else 'resume_execution_20261107.log')).read_text().splitlines();(r/'execution_seed.log').write_text('\n'.join(line for line in loglines if line.startswith(f'{name} {seed} '))+'\n')
   checks.append(dict(candidate=name,seed=seed,checkpoint_sha256=sha(item['checkpoint']),predictions_sha256=sha(path),max_inference_error=err,scalers_train_only=True,origins_equal=True,selected_epoch_verified=True))
  else:
   assert sha(item['checkpoint'])==item['checkpoint_sha256'] and sha(path)==item['predictions_sha256']
  rows.append(dict(candidate=name,seed=seed,**m,selected_epoch=saved['selected_epoch'],epochs=len(hist),parameters=saved['parameters'],training_seconds=saved['training_seconds']));ps.append(pred.predicted.to_numpy())
 predictions[name]=np.stack(ps);print('Verified',name,flush=True)
res=pd.DataFrame(rows);res.to_csv(O/'ALL_SEED_RESULTS.csv',index=False);summary=res.groupby('candidate').agg(**{f'{k}_{op}':(k,op) for k in ['rmse','mae','r2','directional_accuracy','training_seconds'] for op in ['mean','std']}).reset_index()
for ref in ['HBAR_technical']:summary[f'rmse_improvement_vs_{ref}_percent']=100*(1-summary.rmse_mean/summary.loc[summary.candidate.eq(ref),'rmse_mean'].iloc[0])
pers=metrics(actual,origin,origin);summary['rmse_improvement_vs_persistence_percent']=100*(1-summary.rmse_mean/pers['rmse']);summary=summary.sort_values('rmse_mean');summary.to_csv(O/'VALIDATION_RANKING.csv',index=False);save(O/'PERSISTENCE.json',pers)
# Frozen primary estimand: sqrt of seed-averaged squared error, paired circular blocks.
stat=[];n=len(v);rng=np.random.default_rng(2026);sq={k:np.mean((ps-actual)**2,axis=0) for k,ps in predictions.items()}
for block in [72,24,168]:
 starts=rng.integers(0,n,size=(2000,math.ceil(n/block)));sample=(starts[:,:,None]+np.arange(block))%n;sample=sample.reshape(2000,-1)[:,:n]
 for name in candidates:
  for ref in ['HBAR_technical']:
   delta=np.sqrt(sq[name][sample].mean(axis=1))-np.sqrt(sq[ref][sample].mean(axis=1));ci=np.quantile(delta,[.025,.975,.025,.975]);stat.append(dict(candidate=name,reference=ref,block_hours=block,rmse_risk_difference=float(np.sqrt(sq[name].mean())-np.sqrt(sq[ref].mean())),ci95_low=ci[0],ci95_high=ci[1],ci_adjusted_low=ci[2],ci_adjusted_high=ci[3],robust_improvement_adjusted=bool(ci[3]<0)))
pd.DataFrame(stat).to_csv(O/'PAIRED_BLOCK_INTERVALS.csv',index=False)
# Monthly consistency is descriptive, not additional selection.
month=pd.to_datetime(z.available_at_utc,utc=True).dt.strftime('%Y-%m').to_numpy();period=[]
for name,ps in predictions.items():
 for mon in sorted(set(month)):
  ix=month==mon;period.append(dict(candidate=name,month=mon,n=int(ix.sum()),mean_seed_rmse=float(np.mean(np.sqrt(np.mean((ps[:,ix]-actual[ix])**2,axis=1)))) ))
pd.DataFrame(period).to_csv(O/'MONTHLY_CONSISTENCY.csv',index=False)
np.savez_compressed(O/'VALIDATION_PREDICTIONS_ALL.npz',actual=actual,origin=origin,**predictions)
best=summary[summary.candidate.isin(candidates)].iloc[0].to_dict();save(O/'HBAR_RESULTS_LOCK.json',dict(best_new_configuration=best,ranking=summary.to_dict('records'),criterion='Five-seed mean validation price RMSE only',final_evaluation_performed=False,E4_started=False,requires_user_review_before_next_stage=True))
save(O/'NUMERICAL_VERIFICATION.json',dict(passed=True,new_fits=len(checks),checkpoint_checks=checks,frozen_hashes_unchanged=True,shared_news_values_verified=True,train_sequences=len(ids['train']),validation_origins=2136,window_hours=30,horizon_hours=24,real_btc_and_technical_values_unchanged=True,train_only_scalers_verified=True,final_labels_redacted=True,final_evaluation_performed=False,baseline_retrained=False))
print(summary.to_string(index=False))
