from pathlib import Path
import sys,json,hashlib,math
import numpy as np,pandas as pd,torch,joblib
from sklearn.preprocessing import RobustScaler,MinMaxScaler
from gated_model import NewsCorrection
P=Path(__file__).resolve().parents[1];OLD=P.parent/'phase2_e3';A=P/'analysis';A.mkdir(exist_ok=True)
sys.path.insert(0,str(P.parent/'phase1/reference'));from neural import Model
from common import metrics
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,default=str,allow_nan=False)+'\n')
for stage in [P,OLD,P.parent/'phase1']:
 assert all(sha(f)==v for f,v in json.loads((stage/'FROZEN_MANIFEST.json').read_text())['files'].items())
for stage in [OLD,P.parent/'phase1']:
 for report in json.loads((stage/'FINAL_VERIFICATION.json').read_text())['reports']:assert sha(report['path'])==report['sha256']
plan=json.loads((P/'PROTOCOL.json').read_text());reuse=json.loads((P/'E0_REUSE.json').read_text());d=pd.read_csv(P/'data/TRAINING_UNSCALED.csv',float_precision='round_trip');sc=joblib.load(P/'data/scalers.joblib');ids={k:np.load(P/f'data/main_{k}.npy') for k in ['train','validation','evaluation']};v=ids['validation'];t=ids['train'];fit=np.unique((t[:,None]-np.arange(30)).ravel());origin=d.close.iloc[v].to_numpy();actual=d.target_close_plus24h.iloc[v].to_numpy();mask=np.load(P/'data/eligible_news_mask.npy');assert np.array_equal(mask,1-d.no_relevant_news.to_numpy());assert len(v)==2136
av=pd.to_datetime(d.available_at_utc,utc=True);ta=pd.to_datetime(d.target_available_at_utc,utc=True);assert (ta-av).eq(pd.Timedelta(hours=24)).all();assert av.diff().dropna().eq(pd.Timedelta(hours=1)).all();assert ta.iloc[t[-1]]<av.iloc[v[0]] and ta.iloc[v[-1]]<av.iloc[ids['evaluation'][0]];assert d.target_close_plus24h.iloc[ids['evaluation']].isna().all()
sx=RobustScaler().fit(d.iloc[fit][sc['news_features']]);sy=MinMaxScaler().fit(np.log(d.target_close_plus24h.iloc[t]/d.close.iloc[t]).to_numpy()[:,None]);assert np.array_equal(sc['fit_input_rows'],fit) and np.array_equal(sc['fit_target_origins'],t)
for a in ['center_','scale_']:assert np.array_equal(getattr(sx,a),getattr(sc['news_scaler'],a))
for a in ['scale_','min_']:assert np.array_equal(getattr(sy,a),getattr(sc['target_scaler'],a))
tech=pd.read_csv(reuse['data'],float_precision='round_trip');assert np.array_equal(d[sc['technical_features']],tech[sc['technical_features']])
prior=np.load(OLD/'analysis/VALIDATION_PREDICTIONS_ALL.npz');assert np.array_equal(prior['actual'],actual) and np.array_equal(prior['origin'],origin);predictions={k:prior[k] for k in ['E0','E1','E1B','E2_h6','E3']};rows=pd.read_csv(OLD/'analysis/ALL_SEED_RESULTS.csv').to_dict('records');checks=[];groups=[]
torch.set_num_threads(4);torch.use_deterministic_algorithms(True);x=sx.transform(d[sc['news_features']]);xv=torch.tensor(np.stack([x[i-29:i+1] for i in v]),dtype=torch.float32);mv=torch.tensor(mask[v,None]);scale=float(sy.scale_[0]);e0sc=joblib.load(reuse['scaler_path']);xt=e0sc['input_scaler'].transform(d[sc['technical_features']]);basechecks=[]
for item in reuse['seeds']:
 seed=item['seed'];cache=np.load(P/f'cache/E0_seed{seed}.npz');m=Model(19,dict(hidden=128,bidirectional=True));m.load_state_dict(torch.load(item['checkpoint'],weights_only=False,map_location='cpu')['model']);m.eval()
 for split in ['train','validation']:
  fresh=[]
  with torch.no_grad():
   for j in range(0,len(ids[split]),256):fresh.extend(m(torch.tensor(np.stack([xt[i-29:i+1] for i in ids[split][j:j+256]]),dtype=torch.float32)).numpy().ravel())
  assert np.array_equal(np.array(fresh),cache[split+'_scaled'])
 bp=origin*np.exp(sy.inverse_transform(cache['validation_scaled'][:,None]).ravel());assert np.array_equal(bp,predictions['E0'][plan['seeds'].index(seed)]);basechecks.append(dict(seed=seed,cache_exact_fresh_inference=True,checkpoint_hash=sha(item['checkpoint'])))
print('All five frozen E0 caches verified on train and validation',flush=True)
for cfg in plan['models']:
 name=cfg['name'];ps=[]
 for seed in plan['seeds']:
  r=P/'runs'/name/f'seed{seed}';saved=json.loads((r/'COMPLETE.json').read_text());hist=pd.read_csv(r/'history.csv');ck=torch.load(r/'best.pt',weights_only=False,map_location='cpu');assert hist.epoch.tolist()==list(range(1,len(hist)+1));assert int(hist.loc[hist.validation_mse.idxmin(),'epoch'])==ck['epoch']==saved['selected_epoch'];assert ck['plan_sha256']==sha(P/'PROTOCOL.json');assert ck['seed']==seed and ck['news_features']==sc['news_features'];assert ck['frozen_E0_sha256']==next(x['checkpoint_sha256'] for x in reuse['seeds'] if x['seed']==seed)
  m=NewsCorrection(8,cfg['gated']);m.load_state_dict(ck['model']);m.eval();assert sum(p.numel() for p in m.parameters())==cfg['trainable_parameters'];cache=np.load(P/f'cache/E0_seed{seed}.npz');bv=torch.tensor(cache['validation_scaled'][:,None]);parts=[[],[],[],[]]
  with torch.no_grad():
   for j in range(0,len(v),256):
    sl=slice(j,j+256);out=m(xv[sl],bv[sl],mv[sl],scale);assert torch.equal(m(xv[sl],bv[sl],mv[sl],scale,disabled=True)[0],bv[sl]);assert torch.equal(m(xv[sl],bv[sl],torch.zeros_like(mv[sl]),scale)[0],bv[sl])
    for target,z in zip(parts,out):target.extend(z.numpy().ravel())
  vp,gate,delta,cor=map(np.array,parts);fresh=origin*np.exp(sy.inverse_transform(vp[:,None]).ravel());csv=pd.read_csv(r/'validation_predictions.csv',float_precision='round_trip');assert np.array_equal(csv.actual,actual);assert pd.to_datetime(csv.origin_time,utc=True).equals(av.iloc[v].reset_index(drop=True));assert pd.to_datetime(csv.target_time,utc=True).equals(ta.iloc[v].reset_index(drop=True));assert np.max(abs(fresh-csv.predicted))<1e-7
  for col,val in [('news_mask',mask[v]),('raw_gate',gate),('effective_gate',mask[v]*gate),('raw_correction_return',delta),('applied_correction_return',cor/scale)]:assert np.allclose(csv[col],val,atol=1e-7,rtol=0), (name,seed,col,float(np.max(abs(csv[col]-val))))
  assert np.array_equal(vp[mask[v]==0],cache['validation_scaled'][mask[v]==0]);bp=predictions['E0'][plan['seeds'].index(seed)];assert np.array_equal(csv.technical_predicted,bp);met=metrics(actual,fresh,origin)
  for k in ['rmse','mae','r2','directional_accuracy']:assert abs(met[k]-saved[k])<1e-10
  rows.append(dict(saved,parameters=saved['total_parameters']));ps.append(fresh);checks.append(dict(candidate=name,seed=seed,checkpoint_sha256=sha(r/'best.pt'),predictions_sha256=sha(r/'validation_predictions.csv'),max_inference_error=float(np.max(abs(fresh-csv.predicted))),disabled_equals_E0=True,inactive_equals_E0=True,selected_epoch_verified=True))
  for label,sel in [('active',mask[v]==1),('inactive',mask[v]==0),('new_article_hour',d.direct_news_present.iloc[v].to_numpy()==1),('no_new_article_hour',d.direct_news_present.iloc[v].to_numpy()==0)]:
   entry=dict(candidate=name,seed=seed,news_group=label,n=int(sel.sum()))
   if sel.any():entry.update(raw_gate_mean=float(gate[sel].mean()),raw_gate_min=float(gate[sel].min()),raw_gate_max=float(gate[sel].max()),effective_gate_mean=float((mask[v]*gate)[sel].mean()),mean_abs_applied_return=float(np.abs(cor[sel]/scale).mean()),mean_abs_price_correction=float(np.abs(fresh[sel]-bp[sel]).mean()),**{k:val for k,val in metrics(actual[sel],fresh[sel],origin[sel]).items() if k!='n'},technical_rmse=metrics(actual[sel],bp[sel],origin[sel])['rmse'])
   groups.append(entry)
 print('Verified',name,flush=True);predictions[name]=np.stack(ps)
res=pd.DataFrame(rows);res.to_csv(A/'ALL_SEED_RESULTS.csv',index=False);summary=res.groupby('candidate').agg(**{f'{k}_{op}':(k,op) for k in ['rmse','mae','r2','directional_accuracy','training_seconds'] for op in ['mean','std']}).reset_index();pers=metrics(actual,origin,origin);save(A/'PERSISTENCE.json',pers)
for ref in ['E0','E1B','E3']:summary[f'rmse_improvement_vs_{ref}_percent']=100*(1-summary.rmse_mean/summary.loc[summary.candidate.eq(ref),'rmse_mean'].iloc[0])
summary['rmse_improvement_vs_persistence_percent']=100*(1-summary.rmse_mean/pers['rmse']);summary=summary.sort_values('rmse_mean');summary.to_csv(A/'VALIDATION_RANKING.csv',index=False);pd.DataFrame(groups).to_csv(A/'NEWS_ACTIVITY_AND_GATES.csv',index=False)
contrasts=[(name,ref) for name in ['E4_gated','E4_ungated'] for ref in ['E0','E1B','E3']]+[('E4_gated','E4_ungated')];sq={k:np.mean((p-actual)**2,axis=0) for k,p in predictions.items()};rng=np.random.default_rng(2026);intervals=[];n=len(v)
for block in [72,24,168]:
 starts=rng.integers(0,n,size=(2000,math.ceil(n/block)));sample=((starts[:,:,None]+np.arange(block))%n).reshape(2000,-1)[:,:n]
 for name,ref in contrasts:
  diff=np.sqrt(sq[name][sample].mean(axis=1))-np.sqrt(sq[ref][sample].mean(axis=1));q=np.quantile(diff,[.025,.975,.05/(2*7),1-.05/(2*7)]);intervals.append(dict(candidate=name,reference=ref,block_hours=block,rmse_risk_difference=float(np.sqrt(sq[name].mean())-np.sqrt(sq[ref].mean())),ci95_low=q[0],ci95_high=q[1],ci_adjusted_low=q[2],ci_adjusted_high=q[3]))
pd.DataFrame(intervals).to_csv(A/'PAIRED_BLOCK_INTERVALS.csv',index=False);month=av.iloc[v].dt.strftime('%Y-%m').to_numpy();monthly=[]
for name,ps in predictions.items():
 for mon in sorted(set(month)):
  ix=month==mon;monthly.append(dict(candidate=name,month=mon,n=int(ix.sum()),mean_seed_rmse=float(np.mean(np.sqrt(np.mean((ps[:,ix]-actual[ix])**2,axis=1))))))
pd.DataFrame(monthly).to_csv(A/'MONTHLY_CONSISTENCY.csv',index=False);np.savez_compressed(A/'VALIDATION_PREDICTIONS_ALL.npz',actual=actual,origin=origin,**predictions)
save(A/'NUMERICAL_VERIFICATION.json',dict(passed=True,new_fits=10,checkpoint_checks=checks,E0_cache_checks=basechecks,train_only_scalers=True,window_hours=30,horizon_hours=24,validation_origins=2136,active_origins=int(mask[v].sum()),inactive_origins=int((mask[v]==0).sum()),previous_reports_unchanged=True,frozen_hashes_unchanged=True,final_evaluation_performed=False))
save(A/'RESULTS_LOCK.json',dict(best_E4=summary[summary.candidate.str.startswith('E4')].iloc[0].to_dict(),best_overall=summary.iloc[0].to_dict(),selection='Mean five-seed validation RMSE only',further_training_started=False,final_evaluation_performed=False));print(summary.to_string(index=False))
