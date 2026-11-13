from pathlib import Path
import sys,json,hashlib,datetime,math
import numpy as np,pandas as pd,torch,joblib
from sklearn.preprocessing import RobustScaler,MinMaxScaler
B=Path(__file__).resolve().parents[1];ROOT=B.parents[1];N=ROOT/'news-enhanced-crypto-research';P=N/'phase1';R=B/'results';R.mkdir(exist_ok=True)
sys.path.insert(0,str(P/'reference'));from neural import Model
from common import metrics
sys.path.insert(0,str(N/'shared/feature_engineering'));from causal_news import aggregate
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,default=str))
assert not (R/'XRP_FINAL_EVALUATION_RESULTS.csv').exists(),'Final evaluation already saved; do not repeat'
reuse=json.loads((P/'E0_E1_REUSE.json').read_text())['E0'];cfg=json.loads((P/'E1B/CONFIG.json').read_text());base=pd.read_csv(reuse['data'],float_precision='round_trip');news=pd.read_csv(cfg['dataset'],float_precision='round_trip');v=np.load(P/'main_evaluation.npy');tr=np.load(P/'main_train.npy');val=np.load(P/'main_validation.npy');seeds=[42,123,2026,7,99]
assert len(v)==5262 and news.iloc[v].target_close_plus24h.isna().all();assert np.array_equal(base[cfg['features'][:19]],news[cfg['features'][:19]])
av=pd.to_datetime(base.available_at_utc,utc=True);ta=pd.to_datetime(base.target_available_at_utc,utc=True);actual=base.iloc[v].target_close_plus24h.to_numpy();origin=base.iloc[v].close.to_numpy();assert np.isfinite(actual).all();assert (ta-av).eq(pd.Timedelta(hours=24)).all();raw_path=ROOT/'subprojects/xrp-2026-data-original-model/data/review_candidate/XRPUSDT_hourly_through_25Oct_Sydney.csv';raw=pd.read_csv(raw_path,float_precision='round_trip');raw.index=pd.to_datetime(raw.time,utc=True)+pd.Timedelta(hours=1);assert np.array_equal(actual,raw.close.reindex(ta.iloc[v]).to_numpy());assert ta.iloc[tr[-1]]<av.iloc[val[0]] and ta.iloc[val[-1]]<av.iloc[v[0]]
for stage in ['phase1','phase2_e3','phase3_e4']:
 f=json.loads((N/stage/'FROZEN_MANIFEST.json').read_text());assert all(sha(p)==h for p,h in f['files'].items())
models={'E0':{'features':reuse['feature_order'],'scalers':reuse['scaler_path'],'data':base,'checkpoints':[x['checkpoint'] for x in reuse['seeds']]},'E1B':{'features':cfg['features'],'scalers':cfg['scalers'],'data':news,'checkpoints':[str(P/f'E1B/runs/E1B_seed{s}/best.pt') for s in seeds]}}
fit=np.unique((tr[:,None]-np.arange(30)).ravel());files={str(raw_path):sha(raw_path),reuse['data']:sha(reuse['data']),cfg['dataset']:sha(cfg['dataset']),str(P/'analysis/PHASE1_SELECTION_LOCK.json'):sha(P/'analysis/PHASE1_SELECTION_LOCK.json')}
for name,m in models.items():
 sc=joblib.load(m['scalers']);m['sc']=sc;assert np.array_equal(sc['fit_input_rows'],fit) and np.array_equal(sc['fit_target_origins'],tr)
 sx=RobustScaler().fit(m['data'].iloc[fit][m['features']]);sy=MinMaxScaler().fit(np.log(base.iloc[tr].target_close_plus24h/base.iloc[tr].close).to_numpy()[:,None])
 for at in ['center_','scale_']:assert np.array_equal(getattr(sx,at),getattr(sc['input_scaler'],at))
 for at in ['scale_','min_']:assert np.array_equal(getattr(sy,at),getattr(sc['target_scaler'],at))
 for f in [m['scalers']]+m['checkpoints']:files[f]=sha(f)
config={'frozen_before_inference_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'seeds':seeds,'models':['E0','E1B','persistence'],'primary_contrast':'E1B minus E0','origins':len(v),'origin_start':str(av.iloc[v[0]]),'origin_end':str(av.iloc[v[-1]]),'target_start':str(ta.iloc[v[0]]),'target_end':str(ta.iloc[v[-1]]),'window':30,'horizon_hours':24,'target':'log return reconstructed to price','retraining':False,'scalers_verified_train_only':True,'checkpoint_selection':'Preserved historical validation MSE minimum','retrospective_status':'Reserved for this E1B evaluation, but 2026 history examined in other experiments; not a pristine unseen prospective test','news_availability':'Publication proxy, not verified first-public availability; cannot establish exclusion from all historical human design decisions','timing_sensitivities_hours':[1,6,24],'timing_policy':'Frozen model inference only using delayed news; diagnostic distribution shifts, no feature selection or retraining','bootstrap':{'replicates':2000,'seed':2026,'primary_block':72,'sensitivity_blocks':[24,168],'confidence':.95,'estimand':'sqrt(mean over seeds and origins squared errors); gain percent from same risk'},'HAC_lags':71,'files':files,'feature_schemas':{k:m['features'] for k,m in models.items()},'versions':{'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__}}
save(B/'XRP_FINAL_EXPERIMENT_CONFIG.json',config);pd.DataFrame({'origin_time':av.iloc[v].values,'target_time':ta.iloc[v].values,'row':v}).to_csv(R/'FROZEN_ORIGINS.csv',index=False)
torch.set_num_threads(4);preds={};rows=[];long=[];checks=[]
def infer(model,x,ii):
 out=[]
 with torch.no_grad():
  for j in range(0,len(ii),256):out.extend(model(torch.tensor(np.stack([x[i-29:i+1] for i in ii[j:j+256]]),dtype=torch.float32)).numpy().ravel())
 return np.asarray(out)
articles=pd.read_csv(ROOT/'subprojects/original-hybird-extended-oct-2026/data/final_news_v3/NEWS_ARTICLES_FINAL_CALCULATED.csv',float_precision='round_trip');delayed={}
for delay in [1,6,24]:delayed[delay]=aggregate(articles,av,delay=delay)[cfg['news_features']].to_numpy()
for name,m in models.items():
 x=m['sc']['input_scaler'].transform(m['data'][m['features']]);pp=[]
 for seed,cp in zip(seeds,m['checkpoints']):
  ck=torch.load(cp,map_location='cpu',weights_only=False);assert ck['features']==m['features'];model=Model(len(m['features']),dict(hidden=128,bidirectional=True));model.load_state_dict(ck['model']);model.eval();run=Path(cp).parent;hist=pd.read_csv(run/'history.csv');assert int(hist.loc[hist.validation_mse.idxmin(),'epoch'])==ck['epoch']
  vp=base.iloc[val].close.to_numpy()*np.exp(m['sc']['target_scaler'].inverse_transform(infer(model,x,val)[:,None]).ravel());saved=pd.read_csv(run/'validation_predictions.csv',float_precision='round_trip');err=float(np.max(abs(vp-saved.predicted)));assert err<1e-7
  y=origin*np.exp(m['sc']['target_scaler'].inverse_transform(infer(model,x,v)[:,None]).ravel());pp.append(y);rows.append(dict(model=name,seed=seed,selected_epoch=ck['epoch'],**metrics(actual,y,origin)));long.append(pd.DataFrame(dict(model=name,seed=seed,origin_time=av.iloc[v].to_numpy(),target_time=ta.iloc[v].to_numpy(),origin_close=origin,actual=actual,predicted=y,news_active=news.iloc[v].direct_news_present.to_numpy())));checks.append(dict(model=name,seed=seed,checkpoint_sha256=sha(cp),validation_inference_max_error=err))
  if name=='E1B':
   for delay,features in delayed.items():
    dd=news[m['features']].copy();dd.loc[:,cfg['news_features']]=features;xx=m['sc']['input_scaler'].transform(dd);yy=origin*np.exp(m['sc']['target_scaler'].inverse_transform(infer(model,xx,v)[:,None]).ravel());preds.setdefault(f'E1B_delay{delay}',[]).append(yy)
  print('Evaluated',name,seed,flush=True)
 preds[name]=np.stack(pp)
for k in list(preds):preds[k]=np.asarray(preds[k])
res=pd.DataFrame(rows);res.to_csv(R/'XRP_FINAL_SEED_RESULTS.csv',index=False);pd.concat(long).to_csv(R/'XRP_FINAL_PREDICTIONS.csv',index=False);summary=res.groupby('model').agg(**{f'{k}_{op}':(k,op) for k in ['rmse','mae','r2','directional_accuracy'] for op in ['mean','std']}).reset_index();pm=metrics(actual,origin,origin);summary['rmse_gain_vs_E0_percent']=100*(1-summary.rmse_mean/summary.loc[summary.model.eq('E0'),'rmse_mean'].iloc[0]);summary['rmse_gain_vs_persistence_percent']=100*(1-summary.rmse_mean/pm['rmse']);summary.to_csv(R/'XRP_FINAL_EVALUATION_RESULTS.csv',index=False);save(R/'PERSISTENCE.json',pm)
n=len(v);rng=np.random.default_rng(2026);e0=np.mean((preds['E0']-actual)**2,axis=0);e1=np.mean((preds['E1B']-actual)**2,axis=0);stats=[]
for block in [72,24,168]:
 sample=(rng.integers(0,n,size=(2000,math.ceil(n/block)))[:,:,None]+np.arange(block))%n;sample=sample.reshape(2000,-1)[:,:n];a=np.sqrt(e0[sample].mean(axis=1));b=np.sqrt(e1[sample].mean(axis=1));diff=b-a;gain=100*(1-b/a);dl,dh=np.quantile(diff,[.025,.975]);gl,gh=np.quantile(gain,[.025,.975]);stats.append(dict(block_hours=block,rmse_difference=np.sqrt(e1.mean())-np.sqrt(e0.mean()),difference_ci_low=dl,difference_ci_high=dh,gain_percent=100*(1-np.sqrt(e1.mean()/e0.mean())),gain_ci_low=gl,gain_ci_high=gh))
pd.DataFrame(stats).to_csv(R/'XRP_FINAL_BOOTSTRAP_RESULTS.csv',index=False)
import statsmodels.api as sm
hac=sm.OLS(e1-e0,np.ones(n)).fit(cov_type='HAC',cov_kwds={'maxlags':71});save(R/'PAIRED_LOSS_HAC.json',dict(loss='E1B minus E0 squared error averaged over seeds',mean=float(hac.params[0]),se=float(hac.bse[0]),pvalue=float(hac.pvalues[0]),ci95=hac.conf_int().tolist(),lags=71))
sub=[];month=av.iloc[v].dt.strftime('%Y-%m').to_numpy();active=news.iloc[v].direct_news_present.to_numpy().astype(bool)
for name,ps in preds.items():
 for group,mask in [('all',np.ones(n,dtype=bool)),('news_active_current_hour',active),('news_inactive_current_hour',~active)]+[(m,month==m) for m in sorted(set(month))]:
  if mask.any():
   for seed,y in zip(seeds,ps):sub.append(dict(model=name,seed=seed,group=group,**metrics(actual[mask],y[mask],origin[mask])))
pd.DataFrame(sub).to_csv(R/'SUBGROUP_AND_TIMING_RESULTS.csv',index=False);np.savez_compressed(R/'PREDICTIONS_ARRAYS.npz',actual=actual,origin=origin,**preds);pd.DataFrame(dict(origin_time=av.iloc[v].to_numpy(),squared_loss_difference=e1-e0)).to_csv(R/'PAIRED_ERROR_SERIES.csv',index=False)
assert all(sha(p)==h for p,h in files.items());save(R/'NUMERICAL_VERIFICATION.json',dict(passed=True,checkpoint_checks=checks,source_hashes_unchanged=True,train_only_scaling=True,exact_24h=True,origins=5262,final_inference_completed=True));print(summary.to_string(index=False));save(B/'STATUS.json',dict(status='evaluation_complete_reports_pending',retraining=False))
