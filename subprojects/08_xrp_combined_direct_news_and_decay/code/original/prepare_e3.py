from pathlib import Path
import json,shutil,hashlib,sys
import pandas as pd,numpy as np,joblib,torch
from sklearn.preprocessing import RobustScaler
root=Path('/Users/mahmoudali/Disk/Master 2025/cursor projects/crypto-forecasting-research/news-enhanced-crypto-research');src=root/'phase1';out=root/'phase2_e3'
assert not out.exists(),'Preserve existing E3 work'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,default=str)+'\n')
old=json.loads((src/'FROZEN_MANIFEST.json').read_text())
assert all(sha(p)==h for p,h in old['files'].items())
ranks=pd.read_csv(src/'analysis/VALIDATION_RANKING.csv');best=ranks[ranks.candidate.str.startswith('E2_')].sort_values('rmse_mean').iloc[0].candidate;assert best=='E2_h6'
out.mkdir();(out/'E3').mkdir();shutil.copytree(src/'reference',out/'reference')
for f in ['main_train.npy','main_validation.npy','main_evaluation.npy','VALIDATION_ORIGINS.csv']:shutil.copy2(src/f,out/f)
shutil.copy2('/tmp/prepare_xrp_e3.py',out/'prepare_e3.py')
a=json.loads((src/'E1B/CONFIG.json').read_text());b=json.loads((src/best/'CONFIG.json').read_text())
d=pd.read_csv(a['dataset'],float_precision='round_trip');eq=pd.read_csv(b['dataset'],float_precision='round_trip')
assert d.available_at_utc.equals(eq.available_at_utc)
for f in b['news_features']:d[f]=eq[f]
features=a['features']+b['news_features'];assert len(features)==27 and len(set(features))==27
oldsc=joblib.load(a['scalers']);fit=oldsc['fit_input_rows'];sx=RobustScaler().fit(d.iloc[fit][features]);assert np.array_equal(sx.center_[:25],oldsc['input_scaler'].center_);assert np.array_equal(sx.scale_[:25],oldsc['input_scaler'].scale_)
bundle=dict(oldsc,input_scaler=sx,feature_order=features);joblib.dump(bundle,out/'E3/scalers.joblib');d.to_csv(out/'E3/training.csv',index=False)
sys.path.insert(0,str(out/'reference'));from neural import Model
params=sum(x.numel() for x in Model(27,dict(hidden=128,bidirectional=True)).parameters())
cfg=dict(name='E3',half_life_hours=6,delay_hours=0,news_features=a['news_features']+b['news_features'],features=features,parameters=params,dataset=str(out/'E3/training.csv'),scalers=str(out/'E3/scalers.joblib'),equation_selected_from='Phase 1 validation mean RMSE only',source_configs=[str(src/'E1B/CONFIG.json'),str(src/best/'CONFIG.json')]);save(out/'E3/CONFIG.json',cfg)
p=json.loads((src/'PROTOCOL.json').read_text());p.update(status='approved_E3_only_after_user_resume',approved_new_fits=5,candidates=[cfg],stop_after_E3=True,final_evaluation_performed=False,authorization='User resumed after assistant identified E3 as next stage. No E4, limited-addition E2, rolling, HBAR or final evaluation authorized by this continuation.',E3_statistics='Rank by mean validation price RMSE. Paired circular moving-block bootstrap on seed-averaged squared errors, blocks72 primary and24/168 sensitivity,2000 replicates seed2026. Four predeclared comparisons versus E0,E1,E1B,E2_h6:95% intervals and Bonferroni98.75% intervals. Disclose accumulated validation selection optimism; not independent confirmation.');p.pop('phase1_statistics',None);p.pop('stop_after_phase1',None);save(out/'PROTOCOL.json',p)
runner=(src/'train_candidate.py').read_text().replace("assert name in ['E1B','E2_h3','E2_h6','E2_h12','E2_h24']","assert name == 'E3'").replace('Technical-only log-return control; epochs chosen by validation scaled-return MSE','Combined direct and equation news; epochs chosen by validation scaled-return MSE');(out/'train_candidate.py').write_text(runner)
ids={k:np.load(out/f'main_{k}.npy') for k in ['train','validation','evaluation']};v=ids['validation'];t=ids['train'];orig=pd.read_csv(out/'VALIDATION_ORIGINS.csv',float_precision='round_trip');assert len(v)==2136;assert d.iloc[v][orig.columns].reset_index(drop=True).equals(orig)
assert d.iloc[ids['evaluation']].target_close_plus24h.isna().all()
av=pd.to_datetime(d.available_at_utc,utc=True);assert av.diff().dropna().eq(pd.Timedelta(hours=1)).all();assert (pd.to_datetime(d.target_available_at_utc,utc=True)-av).eq(pd.Timedelta(hours=24)).all();assert av.iloc[t[-1]]+pd.Timedelta(hours=24)<av.iloc[v[0]]
x=sx.transform(d[features]);assert np.isfinite(x).all();torch.set_num_threads(4);torch.manual_seed(42);m=Model(27,dict(hidden=128,bidirectional=True));xx=torch.tensor(np.stack([x[i-29:i+1] for i in t[:2]]),dtype=torch.float32);yp=m(xx);assert torch.isfinite(yp).all();yp.square().mean().backward();assert all(torch.isfinite(p.grad).all() for p in m.parameters())
save(out/'PREFLIGHT_VERIFICATION.json',dict(passed=True,validation_origins=2136,features=27,parameters=params,train_only_scalers=True,original_25_features_unchanged=True,final_targets_redacted=True,exact24h=True,window30=True,finite_forward_backward=True,phase1_hashes_unchanged=True))
files={str(f):sha(f) for f in out.rglob('*') if f.is_file() and '__pycache__' not in str(f)}
for f in [src/'FROZEN_MANIFEST.json',src/'analysis/VALIDATION_RANKING.csv',src/'E1B/training.csv',src/best/'training.csv']:files[str(f)]=sha(f)
save(out/'FROZEN_MANIFEST.json',dict(created_utc=pd.Timestamp.now(tz='UTC'),files=files))
(out/'README.md').write_text('# E3 combined direct and equation news\n\nFive seeds only:42,123,2026,7,99. Uses six E1B news inputs plus six-hour equation sentiment and no-relevant-news flag. Half-life selected from completed Phase1 validation results. Same27-input BiLSTM design (input width changes),30-hour windows,+24-hour log-return target,training-only scaling and2136 matched validation origins. Final evaluation remains redacted. Stop after E3.\n\nRun once using existing research Python: `python train_candidate.py E3`. STARTED lock prevents automatic restarts. Frozen hashes are checked before training.\n')
print(json.dumps({'path':str(out),'parameters':params,'preflight':'passed','fits':5}))
