from pathlib import Path
import sys,json,hashlib,shutil
import numpy as np,pandas as pd,torch,joblib
from sklearn.preprocessing import RobustScaler,MinMaxScaler
P=Path(__file__).resolve().parent;ROOT=P.parent;PH=ROOT/'phase1';HB=ROOT.parent/'subprojects/hbar-controlled-technical-bilstm';D=HB/'data/training_fixed12_v1';RUN=HB/'runs/controlled_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,default=str)+'\n')
assert not (P/'FROZEN_MANIFEST.json').exists() and not (P/'BASELINE_REUSE_VERIFICATION.json').exists()
for stage in [PH,ROOT/'phase2_e3',ROOT/'phase3_e4']:
 assert json.loads((stage/'FINAL_VERIFICATION.json').read_text())['passed'];assert all(sha(f)==h for f,h in json.loads((stage/'FROZEN_MANIFEST.json').read_text())['files'].items())
shutil.copytree(PH/'reference',P/'reference',dirs_exist_ok=True);sys.path.insert(0,str(P/'reference'));from neural import Model
from common import metrics
manifest=json.loads((D/'TRAINING_MANIFEST.json').read_text());assert sha(manifest['dataset'])==manifest['sha256'];d=pd.read_csv(manifest['dataset'],float_precision='round_trip');xrp=pd.read_csv(PH/'E1B/training.csv',float_precision='round_trip');cfg=json.loads((PH/'E1B/CONFIG.json').read_text());tech=manifest['technical_feature_order'];news=cfg['news_features'];assert tech==cfg['features'][:19]
assert d.available_at_utc.equals(xrp.available_at_utc) and d.target_available_at_utc.equals(xrp.target_available_at_utc);assert np.array_equal(d.BTC_Close,xrp.BTC_Close)
ids={k:np.load(D/f'main_{k}.npy') for k in ['train','validation','evaluation']}
for k,idx in ids.items():assert np.array_equal(idx,np.load(PH/f'main_{k}.npy'));np.save(P/f'main_{k}.npy',idx)
av=pd.to_datetime(d.available_at_utc,utc=True);ta=pd.to_datetime(d.target_available_at_utc,utc=True);v=ids['validation'];t=ids['train'];fit=np.unique((t[:,None]-np.arange(30)).ravel());assert len(v)==2136 and len(d)==26337;assert (ta-av).eq(pd.Timedelta(hours=24)).all() and av.diff().dropna().eq(pd.Timedelta(hours=1)).all();assert ta.iloc[t[-1]]<av.iloc[v[0]] and ta.iloc[v[-1]]<av.iloc[ids['evaluation'][0]]
sc=joblib.load(RUN/'log_return_scalers.joblib');assert sc['feature_order']==tech;assert np.array_equal(sc['fit_input_rows'],fit) and np.array_equal(sc['fit_target_origins'],t)
sx=RobustScaler().fit(d.iloc[fit][tech]);sy=MinMaxScaler().fit(np.log(d.target_close_plus24h.iloc[t]/d.close.iloc[t]).to_numpy()[:,None])
for attr in ['center_','scale_']:assert np.array_equal(getattr(sx,attr),getattr(sc['input_scaler'],attr))
for attr in ['scale_','min_']:assert np.array_equal(getattr(sy,attr),getattr(sc['target_scaler'],attr))
assert sha(HB/'reference/neural.py')==sha(PH/'reference/neural.py')
torch.set_num_threads(4);torch.use_deterministic_algorithms(True);x=sx.transform(d[tech]);checks=[]
for seed in manifest['seeds']:
 r=RUN/f'fixed_seed{seed}';ck=torch.load(r/'best.pt',weights_only=False,map_location='cpu');assert ck['features']==tech and ck['seed']==seed;assert ck['plan_sha256']==sha(RUN/'PLAN.json');hist=pd.read_csv(r/'history.csv');saved=json.loads((r/'COMPLETE.json').read_text());assert ck['epoch']==saved['selected_epoch']==int(hist.loc[hist.validation_mse.idxmin(),'epoch'])
 m=Model(19,dict(hidden=128,bidirectional=True));m.load_state_dict(ck['model']);m.eval();assert sum(p.numel() for p in m.parameters())==564353;vals=[]
 with torch.no_grad():
  for j in range(0,len(v),256):vals.extend(m(torch.tensor(np.stack([x[i-29:i+1] for i in v[j:j+256]]),dtype=torch.float32)).numpy().ravel())
 pred=d.close.iloc[v].to_numpy()*np.exp(sy.inverse_transform(np.array(vals)[:,None]).ravel());orig=pd.read_csv(r/'validation_predictions.csv',float_precision='round_trip');assert np.max(abs(pred-orig.predicted))<1e-7;assert np.array_equal(orig.actual,d.target_close_plus24h.iloc[v]);assert pd.to_datetime(orig.origin_time,utc=True).equals(av.iloc[v].reset_index(drop=True));assert pd.to_datetime(orig.target_time,utc=True).equals(ta.iloc[v].reset_index(drop=True))
 result=metrics(orig.actual.to_numpy(),pred,d.close.iloc[v].to_numpy())
 for key in ['rmse','mae','r2','directional_accuracy']:assert np.isclose(result[key],saved[key],atol=1e-10,rtol=0)
 checks.append(dict(seed=seed,checkpoint=str(r/'best.pt'),checkpoint_sha256=sha(r/'best.pt'),predictions=str(r/'validation_predictions.csv'),predictions_sha256=sha(r/'validation_predictions.csv'),prediction_max_error=float(np.max(abs(pred-orig.predicted))),metrics=result,selected_epoch=ck['epoch']))
 print('Verified HBAR technical seed',seed,flush=True)
save(P/'BASELINE_REUSE_VERIFICATION.json',dict(passed=True,project=str(HB),dataset=manifest['dataset'],dataset_sha256=manifest['sha256'],scaler=str(RUN/'log_return_scalers.joblib'),scaler_sha256=sha(RUN/'log_return_scalers.joblib'),seeds=checks,train_only_scalers=True,parameters=564353,validation_origins=2136,architecture_identical_to_XRP=True,previous_final_evaluation_exists=True))
name='HBAR_E1B';out=P/name;out.mkdir(exist_ok=True);z=d[['available_at_utc','target_available_at_utc','target_close_plus24h']+tech].copy()
for c in news:z[c]=xrp[c]
used=np.r_[t,v];z.loc[~z.index.isin(used),'target_close_plus24h']=np.nan;z.to_csv(out/'training.csv',index=False);features=tech+news;sxnew=RobustScaler().fit(z.iloc[fit][features]);assert np.array_equal(sxnew.center_[:19],sx.center_) and np.array_equal(sxnew.scale_[:19],sx.scale_);assert np.isfinite(sxnew.transform(z[features])).all();joblib.dump(dict(input_scaler=sxnew,target_scaler=sy,feature_order=features,fit_input_rows=fit,fit_target_origins=t),out/'scalers.joblib')
newcfg=dict(name=name,half_life_hours=None,delay_hours=0,news_features=news,features=features,parameters=570497,dataset=str(out/'training.csv'),scalers=str(out/'scalers.joblib'),news_source=str(PH/'E1B/training.csv'),news_source_sha256=sha(PH/'E1B/training.csv'),news_interpretation='XRP-scored cross-asset signals, not HBAR-specific scores');save(out/'CONFIG.json',newcfg)
z.iloc[v][['available_at_utc','target_available_at_utc','close','target_close_plus24h']].to_csv(P/'VALIDATION_ORIGINS.csv',index=False)
protocol=json.loads((PH/'PROTOCOL.json').read_text());protocol.update(status='approved_HBAR_E1B_only_after_resume',approved_new_fits=5,candidates=[newcfg],stop_after_HBAR=True,selection='E1B selected using XRP validation before this HBAR transfer. No HBAR feature or half-life tuning. Each seed checkpoint chosen by HBAR validation scaled-return MSE; fixed five-seed mean reporting.',authorization='User resumed after E4 completion. Continue next planned HBAR transfer stage with five new fits and reused compatible HBAR baseline only.',HBAR_statistics='One primary contrast HBAR_E1B vs matched reused HBAR technical BiLSTM. Paired circular moving-block bootstrap on seed-averaged squared errors,72h primary24/168sensitivity,2000replicatesseed2026,95% intervals. Persistence contextual. Five seeds are not independent markets. Historical HBAR period and baseline were examined previously; no pristine prospective claim.',final_evaluation_performed=False,limitations='Publication-time availability proxies; XRP-specific stance and specificity reused as cross-asset signals. HBAR baseline has a previously completed final evaluation in the archive, which is not read or used for selection in this stage. HBAR own scalers and prices are used.');protocol.pop('phase1_statistics',None);protocol.pop('stop_after_phase1',None);save(P/'PROTOCOL.json',protocol)
runner=(PH/'train_candidate.py').read_text().replace("assert name in ['E1B','E2_h3','E2_h6','E2_h12','E2_h24']","assert name == 'HBAR_E1B'").replace('Technical-only log-return control; epochs chosen by validation scaled-return MSE','Fixed XRP-selected compact news transfer to HBAR; epochs chosen by HBAR validation scaled-return MSE');(P/'train_candidate.py').write_text(runner)
torch.manual_seed(42);m=Model(25,dict(hidden=128,bidirectional=True));xx=torch.tensor(np.stack([sxnew.transform(z[features])[i-29:i+1] for i in t[:2]]),dtype=torch.float32);y=m(xx);assert torch.isfinite(y).all();y.square().mean().backward();assert all(torch.isfinite(p.grad).all() for p in m.parameters());assert sum(p.numel() for p in m.parameters())==570497
save(P/'PREFLIGHT_VERIFICATION.json',dict(passed=True,baseline_five_checkpoints_reused=True,finite_forward_backward=True,features=25,parameters=570497,rows=26337,validation_origins=2136,news_values_exact_XRP_E1B=True,BTC_values_exact_XRP=True,technical_values_exact_archived_HBAR=True,HBAR_train_only_scalers=True,final_targets_redacted=True))
(P/'README.md').write_text('# HBAR compact direct news transfer\n\nFive new seeds only; HBAR technical baseline reused after independent checkpoint inference verification. Same30hinputs and24hlogreturntarget,19HBARtechnical/BTCfeatures plus6compactnewsfields from XRP E1B. Own HBAR scalers fit only training. XRP stance/specificity are cross-asset signals, not HBAR-specific annotations. Earlier HBAR final evaluation exists and was not used in this stage; no fresh prospective claim. Run train_candidate.py HBAR_E1B with research ML Python. Stop after reports, no final evaluation or additional stages.\n')
files={str(f):sha(f) for f in P.rglob('*') if f.is_file() and '__pycache__' not in str(f)}
for f in [Path(manifest['dataset']),RUN/'log_return_scalers.joblib',HB/'code/train.py',PH/'E1B/training.csv',PH/'analysis/VALIDATION_RANKING.csv',ROOT/'phase3_e4/analysis/VALIDATION_RANKING.csv']:files[str(f)]=sha(f)
for c in checks:
 for k in ['checkpoint','predictions']:files[c[k]]=sha(c[k])
save(P/'FROZEN_MANIFEST.json',dict(created_utc=pd.Timestamp.now(tz='UTC'),files=files));print('HBAR baseline verified and new five-fit configuration frozen')
